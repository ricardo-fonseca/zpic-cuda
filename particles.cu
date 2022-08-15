#include "particles.cuh"
#include <iostream>
#include <string>

#include "util.cuh"

#include "timer.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg=cooperative_groups;


__global__
/**
 * @brief CUDA Kernel for initializing t_part_tile array
 * 
 * @param d_tiles       Tile information array
 * @param ntiles        Number of tiles in each direction
 * @param max_np_tile   Maximum number of particles per tile
 */
void _init_tiles_kernel( t_part_tile * const __restrict__ d_tiles, 
    uint2 const ntiles, unsigned int const max_np_tile) {

    auto grid = cg::this_grid();

    for( unsigned int i = grid.thread_rank(); i < ntiles.y * ntiles.x; i += grid.num_threads() ) {
        t_part_tile t = {
            .tile = make_uint2( i % ntiles.x, i / ntiles.x ),
            .pos = i * max_np_tile,
            .n = 0,
            .nb = 0
        };
        d_tiles[i] = t;
    } 
}

/**
 * @brief Construct a new Particle Buffer:: Particle Buffer object
 * 
 * @param ntiles        Number of tiles (x,y)
 * @param nx            Tile size
 * @param max_np_tile   Maximum number of particles per tile
 */
__host__
Particles::Particles(uint2 const ntiles, uint2 const nx, unsigned int const max_np_tile ) :
    ntiles( ntiles ), nx( nx ), max_np_tile( max_np_tile )
{    
    size_t size = ntiles.x * ntiles.y * max_np_tile;
    malloc_dev( ix, size );
    malloc_dev( x, size );
    malloc_dev( u, size );

    // Allocate tile information array on device and initialize using a CUDA kernel
    malloc_dev( tiles, ntiles.x * ntiles.y );
    _init_tiles_kernel <<< 1, 256 >>> ( tiles, ntiles, max_np_tile );
};


__global__
/**
 * @brief CUDA Kernel for getting total number of particles
 * 
 * Note that the kernel does not reset the output total value
 * 
 * @param d_tiles   Tile information
 * @param ntiles    total number of tiles
 * @param total     (out) total number of particles
 */
void _np_kernel( t_part_tile const * const __restrict__ d_tiles, 
    unsigned int const ntiles, unsigned int * const __restrict__ total) {
    auto group = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(group);

    unsigned int v = 0;
    for( int i = group.thread_rank(); i < ntiles; i += group.num_threads() )
        v += d_tiles[i].n;
    
    v = cg::reduce( warp, v, cg::plus<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( total, v );
}

__host__
/**
 * @brief Gets total number of particles on device
 * 
 * @return unsigned long long   Total number of particles
 */
unsigned int Particles::np() {
    _dev_np = 0;
    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    _np_kernel <<< grid, block >>> ( tiles, size, _dev_np.ptr() );
    return _dev_np.get();
}


__global__
void _np_exscan_kernel( 
    unsigned int * const __restrict__ idx,
    t_part_tile const * const __restrict__ tiles, unsigned int const ntiles,
    unsigned int * const __restrict__ total) {

    __shared__ unsigned int tmp[ 32 ];
    __shared__ unsigned int prev;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    prev = 0;

    for( unsigned int i = block.thread_rank(); i < ntiles; i += block.num_threads() ) {
        unsigned int s = tiles[i].n;

        unsigned int v = cg::exclusive_scan( warp, s, cg::plus<unsigned int>());
        if ( warp.thread_rank() == warp.num_threads() - 1 ) tmp[ warp.meta_group_rank() ] = v + s;
        block.sync();

        if ( warp.meta_group_rank() == 0 ) {
            auto t = tmp[ warp.thread_rank() ];
            t = cg::exclusive_scan( warp, t, cg::plus<unsigned int>());
            tmp[ warp.thread_rank() ] = t + prev;
        }
        block.sync();

        v += tmp[ warp.meta_group_rank() ];
        idx[i] = v;

        if ((block.thread_rank() == block.num_threads() - 1) || ( i + 1 == ntiles ) )
            prev = v + s;
        block.sync();
    }

    if ( block.thread_rank() == 0 ) *total = prev;

}

/**
 * @brief Exclusive scan of number of particles per tile
 * 
 * This is used for compacting operations
 * 
 * @param d_offset          Output array on device, must be of size ntiles.x * ntiles.y
 * @return unsigned int     Total number of particles
 */
unsigned int Particles::np_exscan( unsigned int * __restrict__ d_offset ) {

    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    _dev_np = 0;
    _np_exscan_kernel <<< 1, block >>> ( d_offset, tiles, size, _dev_np.ptr() );
    return _dev_np.get();
}

/**
 * @brief CUDA kernel for gathering particle data
 * 
 * @tparam quant        Quantiy to gather
 * @param d_ix          Particle data (cells)
 * @param d_x           Particle data (positions)
 * @param d_u           Particle data (generalized velocity)
 * @param d_tiles       Particle tile information
 * @param tile_nx       Size of tile grid
 * @param d_out_offset  Output array offsets
 * @param d_data        Output data
 */
template < part::quant quant >
__global__
void _gather_quant( 
    int2 const * const __restrict__ d_ix, 
    float2 const * const __restrict__ d_x, 
    float3 const * const __restrict__ d_u, 
    t_part_tile const * const __restrict__ d_tiles, uint2 const tile_nx,
    unsigned int const * const __restrict__ d_out_offset, 
    float * const __restrict__ d_data )
{    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;

    int2   __restrict__ const * const ix = &d_ix[ offset ];
    float2 __restrict__ const * const x  = &d_x[ offset ];
    float3 __restrict__ const * const u  = &d_u[ offset ];
    
    unsigned int const out_offset = d_out_offset[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        float val;
        if ( quant == part::x )  val = (blockIdx.x * tile_nx.x + ix[idx].x) + (0.5f + x[idx].x);
        if ( quant == part::y )  val = (blockIdx.y * tile_nx.y + ix[idx].y) + (0.5f + x[idx].y);
        if ( quant == part::ux ) val = u[idx].x;
        if ( quant == part::uy ) val = u[idx].y;
        if ( quant == part::uz ) val = u[idx].z;
        d_data[ out_offset + idx ] = val;
    }
};


__host__
/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * @param quant         Quantity to gather
 * @param h_data        Output data host buffer, assumed to have size >= np
 * @param np            Number of particles
 * @param d_data_offset Data offset in output array for each tile
 */
void Particles::gather( part::quant quant, float * const __restrict__ h_data, 
        float * const __restrict__ d_data, 
        unsigned int const np, unsigned int const * const __restrict__ d_out_offset ) {

    if ( np > 0 ) {
        dim3 grid( ntiles.x, ntiles.y );
        dim3 block( 1024 );

        // Gather data on device
        switch (quant) {
        case part::x : 
            _gather_quant<part::x> <<<grid,block>>>( ix, x, u, tiles, nx, d_out_offset, d_data );
            break;
        case part::y:
            _gather_quant<part::y> <<<grid,block>>>( ix, x, u, tiles, nx, d_out_offset, d_data );
            break;
        case part::ux:
            _gather_quant<part::ux> <<<grid,block>>>( ix, x, u, tiles, nx, d_out_offset, d_data );
            break;
        case part::uy:
            _gather_quant<part::uy> <<<grid,block>>>( ix, x, u, tiles, nx, d_out_offset, d_data );
            break;
        case part::uz:
            _gather_quant<part::uz> <<<grid,block>>>( ix, x, u, tiles, nx, d_out_offset, d_data );
            break;
        }

        // Copy to host
        devhost_memcpy( h_data, d_data, np );
    }
}

__host__
/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * This version will first do an exscan on the number of particles per tile to
 * determine the data offset on the outout buffer for each tile and call the 
 * above version.
 * 
 * @param quant     Quantity to gather
 * @param h_data    Output data host buffer, assumed to have size >= np
 */
void Particles::gather( part::quant quant, float * const __restrict__ h_data ) {
        
    unsigned int * d_out_offset;
    malloc_dev( d_out_offset, ntiles.x * ntiles.y );
    unsigned int np = np_exscan( d_out_offset );
    
    if ( np > 0 ) {
        float * d_data;
        malloc_dev( d_data, np );
        gather( quant, h_data, d_data, np, d_out_offset );
        free_dev( d_data );
    }
    
    free_dev( d_out_offset );
}

__host__
/**
 * @brief Save particle data to disk
 * 
 * @param info  Particle metadata (name, labels, units, etc.). Information is used to set file name
 * @param iter  Iteration metadata
 * @param path  Path where to save the file
 */
void Particles::save( zdf::part_info &info, zdf::iteration &iter, std::string path ) {

    // Get number of particles and data offsets
    unsigned int * d_out_offset;
    malloc_dev( d_out_offset, ntiles.x * ntiles.y );
    unsigned int np = np_exscan( d_out_offset );
    info.np = np;

    // Open file
    zdf::file part_file;
    zdf::open_part_file( part_file, info, iter, path+"/"+info.name );

    // Gather and save each quantity
    float *h_data = nullptr, *d_data = nullptr;
    if( np > 0 ) {
        malloc_host( h_data, np );
        malloc_dev( d_data, np );
    }

    gather( part::quant::x, h_data, d_data, np, d_out_offset );
    zdf::add_quant_part_file( part_file, "x", h_data, np );

    gather( part::quant::y, h_data, d_data, np, d_out_offset );
    zdf::add_quant_part_file( part_file, "y", h_data, np );

    gather( part::quant::ux, h_data, d_data, np, d_out_offset );
    zdf::add_quant_part_file( part_file, "ux", h_data, np );

    gather( part::quant::uy, h_data, d_data, np, d_out_offset );
    zdf::add_quant_part_file( part_file, "uy", h_data, np );

    gather( part::quant::uz, h_data, d_data, np, d_out_offset );
    zdf::add_quant_part_file( part_file, "uz", h_data, np );

    // Close the file
    zdf::close_file( part_file );

    // Cleanup
    if ( np > 0 ) {
        free_dev( d_data );
        free_host( h_data );
    }
    free_dev( d_out_offset );
}

/**
 * @brief CUDA kernel for copying particles out of the tile to a temp buffer
 * 
 * @tparam dir          Direction to check `coord::x` or `coord::y`
 * @param lim           Tile size along chosen direction
 * @param d_tiles       Tile information (main buffer)
 * @param d_ix          Particle cells (main buffer)
 * @param d_x           Particle positions (main buffer)
 * @param d_u           Particle momenta (main buffer)
 * @param tmp_d_tiles   Tile information (temp buffer)
 * @param tmp_d_ix      Particle cells (temp buffer)
 * @param tmp_d_x       Particle positions (temp buffer)
 * @param tmp_d_u       Particle momenta (temp buffer)
 */
template < coord::cart dir >
__global__
void _bnd_out( int const lim, 
    t_part_tile * const __restrict__ d_tiles,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u,
    t_part_tile * const __restrict__ tmp_d_tiles,
    int2 * __restrict__ tmp_d_ix, float2 * __restrict__ tmp_d_x, float3 * __restrict__ tmp_d_u )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const size = d_tiles[ tid ].n;
    unsigned int const offset =  d_tiles[ tid ].pos;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    unsigned int const tmp_offset =  tmp_d_tiles[ tid ].pos;
    int2   __restrict__ *tmp_ix = &tmp_d_ix[ tmp_offset ];
    float2 __restrict__ *tmp_x  = &tmp_d_x[ tmp_offset ];
    float3 __restrict__ *tmp_u  = &tmp_d_u[ tmp_offset ];


    __shared__ unsigned int _n1;
    __shared__ unsigned int _n2;

    _n1 = 0;
    _n2 = 0;

    block.sync();

    // Count number of particles with keys = 1, 2
    unsigned int n2 = 0, n1 = 0;
    for( unsigned int i = block.thread_rank(); i < size; i+= block.num_threads() ) {
        int key;

        // Template
        if ( dir == coord::x ) key = ( ix[i].x < 0 ) + 2 * ( ix[i].x >= lim );
        if ( dir == coord::y ) key = ( ix[i].y < 0 ) + 2 * ( ix[i].y >= lim );

        n2 += ( key == 2 );
        n1 += ( key == 1 );
    }

    n2 = cg::reduce( warp, n2, cg::plus<unsigned int>());
    n1 = cg::reduce( warp, n1, cg::plus<unsigned int>());

    if ( warp.thread_rank() == 0 ) {
        atomicAdd( &_n2, n2 );
        atomicAdd( &_n1, n1 );
    }

    block.sync();

    n1 = _n1;
    n2 = _n2;

    unsigned int const nmove = n2 + n1;
    unsigned int const n0 = size - nmove;

    __shared__ unsigned int _k0;
    __shared__ unsigned int _k1;
    __shared__ unsigned int _k2;

    _k2 = n1;
    _k1 = 0;
    _k0 = nmove;

    block.sync();

    // Copy values that are moving to temp buffer
    {
        int k2 , k1, k0;
        for( int i = block.thread_rank(); i < size; i+= block.num_threads() ) {
            int key;
            
            // Template
            if ( dir == coord::x ) key = ( ix[i].x < 0 ) + 2 * ( ix[i].x >= lim );
            if ( dir == coord::y ) key = ( ix[i].y < 0 ) + 2 * ( ix[i].y >= lim );

            int const c2 = ( key == 2 );
            int const c1 = ( key == 1 );
            int const c0 = ( key == 0 ) && ( i >= n0 ); // Used to fill holes

            if ( c2 ) k2 = atomicAdd( &_k2, 1 );
            if ( c1 ) k1 = atomicAdd( &_k1, 1 );
            if ( c0 ) k0 = atomicAdd( &_k0, 1 );

            // Data prefetch using coalesced access
            int2   pre_ix = ix[i];
            float2 pre_x  = x[i];
            float3 pre_u  = u[i];

            if ( key || c0 ) {
                int const idx = c2 * k2 + c1 * k1 + c0 * k0;
                tmp_ix[idx] = pre_ix;
                tmp_x[idx]  = pre_x;
                tmp_u[idx]  = pre_u;
            }
        }
    }

    __shared__ unsigned int _k;
    _k = nmove;

    block.sync();

    // Fill holes left behind
    {
        int k;
        for( int i = block.thread_rank(); i < n0; i+= block.num_threads() ) {
            int c;

            // Template
            if ( dir == coord::x ) c = ( ix[i].x < 0 ) || ( ix[i].x >= lim );
            if ( dir == coord::y ) c = ( ix[i].y < 0 ) || ( ix[i].y >= lim );

            if ( c ) {
                k = atomicAdd( &_k, 1 );
                ix[i] = tmp_ix[ k ];
                x[i] = tmp_x[ k ];
                u[i] = tmp_u[ k ];
            }
        }
    }

    // Store new values on tile information
    if ( block.thread_rank() == 0 ) {
        d_tiles[ tid ].n  = n0;
        tmp_d_tiles[ tid ].n  = n1;
        tmp_d_tiles[ tid ].nb = n2;
    }
}


#if 0

// __UNDER DEVELOPMENT__ do not remove!

template < coord::cart dir >
__global__
void _bnd_out_r2( int const lim, 
    t_part_tile * const __restrict__ d_tiles,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u,
    t_part_tile * const __restrict__ tmp_d_tiles,
    int2 * __restrict__ tmp_d_ix, float2 * __restrict__ tmp_d_x, float3 * __restrict__ tmp_d_u )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const size = d_tiles[ tid ].n;
    unsigned int const offset =  d_tiles[ tid ].pos;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    unsigned int const tmp_offset =  tmp_d_tiles[ tid ].pos;
    int2   __restrict__ *tmp_ix = &tmp_d_ix[ tmp_offset ];
    float2 __restrict__ *tmp_x  = &tmp_d_x[ tmp_offset ];
    float3 __restrict__ *tmp_u  = &tmp_d_u[ tmp_offset ];

    __shared__ int2   stage_ix[ 1024 ];
    __shared__ float2 stage_x[ 1024 ];
    __shared__ float3 stage_u[ 1024 ];

    unsigned int total_n1, total_n2;

    {   // Get total_n1 and total_n2

        __shared__ unsigned int _n1;
        __shared__ unsigned int _n2;
        _n1 = 0;
        _n2 = 0;

        block.sync();

        // Count number of particles with keys = 1, 2
        unsigned int n2 = 0, n1 = 0;
        for( unsigned int i = block.thread_rank(); i < size; i+= block.num_threads() ) {
            int key;

            // Template
            if ( dir == coord::x ) key = ( ix[i].x < 0 ) + 2 * ( ix[i].x >= lim );
            if ( dir == coord::y ) key = ( ix[i].y < 0 ) + 2 * ( ix[i].y >= lim );

            n2 += ( key == 2 );
            n1 += ( key == 1 );
        }

        n2 = cg::reduce( warp, n2, cg::plus<unsigned int>());
        n1 = cg::reduce( warp, n1, cg::plus<unsigned int>());

        if ( warp.thread_rank() == 0 ) {
            atomicAdd( &_n2, n2 );
            atomicAdd( &_n1, n1 );
        }

        block.sync();
        total_n1 = _n1;
        total_n2 = _n2;
    }

    // Process particles one block.num_threads() at a time 
    for( int i = 0; i < size; i+= block.num_threads() ) {

        // particle index
        int j = i + block.thread_rank();
        int key;
        unsigned int n1, n2;
        
        // Get number of particles leaving this chunk
        __shared__ unsigned int _n1;
        __shared__ unsigned int _n2;
        _n1 = 0; _n2= 0;

        block.sync();

        if ( j < size ) {

            // Template
            if ( dir == coord::x ) key = ( ix[j].x < 0 ) + 2 * ( ix[j].x >= lim );
            if ( dir == coord::y ) key = ( ix[j].y < 0 ) + 2 * ( ix[j].y >= lim );

            n2 = ( key == 2 );
            n1 = ( key == 1 );
        } else {
            n2 = 0;
            n1 = 0;
            key = 0;
        }

        n2 = cg::reduce( warp, n2, cg::plus<unsigned int>());
        n1 = cg::reduce( warp, n1, cg::plus<unsigned int>());
        if ( warp.thread_rank() == 0 ) {
            atomicAdd( &_n2, n2 );
            atomicAdd( &_n1, n1 );
        }

        // Compact outgoing data into staging buffers

        __shared__ unsigned int _k2;
        __shared__ unsigned int _k1;
        __shared__ unsigned int _k0;
        _k2 = 0; _k1 = 0; _k0 = 0;

        block.sync();
        n2 = _n2; n1 = _n1;

        unsigned int k0, k1, k2;

        if ( j < size ) {
            int const c2 = ( key == 2 );
            int const c1 = ( key == 1 );
            int const c0 = ( key == 0 ) && ( j >= n0 ); // Used to fill holes

            if ( c2 ) k2 = atomicAdd( &_k2, 1 );
            if ( c1 ) k1 = atomicAdd( &_k1, 1 );
            if ( c0 ) k0 = atomicAdd( &_k0, 1 );

            if ( key || c0 ) {
                int const idx = c2 * ( k2 + n1 ) + c1 * k1 + c0 * ( k0 + n2 + n1 );
                stage_ix[idx] = ix[j];
                stage_x[idx]  = x[j];
                stage_u[idx]  = u[j];
            }
        }

        block.sync();
        k2 = _k2; k1 = _k1; k0 = k0;

        // Copy data to external temporary memory
        int const l = block.thread_rank();
        if ( l < k0 + k1 + k2 ) {
            
            int const idx;
            if ( l > k1 + k2 ) idx = total_n1 + total_n2 + total_k0;
            else if ( l > k1 ) idx = total_n1 + total_k2;
            else idx = 
            tmp_ix[ idx ] = stage_ix[l];
            tmp_x[ idx ]  = stage_x[l];
            tmp_u[ idx ]  = stage_u[l];
        }

        if ( l == 0 ) {
            total_k0 += k0;
            total_k1 += k1;
            total_k2 += k2;
        }

    }


    // Fill holes left behind
    {
        int k;
        for( int i = block.thread_rank(); i < n0; i+= block.num_threads() ) {
            int c;

            // Template
            if ( dir == coord::x ) c = ( ix[i].x < 0 ) || ( ix[i].x >= lim );
            if ( dir == coord::y ) c = ( ix[i].y < 0 ) || ( ix[i].y >= lim );

            if ( c ) {
                k = atomicAdd( &_k, 1 );
                ix[i] = tmp_ix[ k ];
                x[i] = tmp_x[ k ];
                u[i] = tmp_u[ k ];
            }
        }
    }

    // Store new values on tile information
    if ( block.thread_rank() == 0 ) {
        d_tiles[ tid ].n  = n0;
        tmp_d_tiles[ tid ].n  = n1;
        tmp_d_tiles[ tid ].nb = n2;
    }
}

#endif

/**
 * @brief CUDA kernel for copying in particles that moved out of neighboring
 * tiles into the local tile.
 * 
 * @tparam dir          Direction to check `coord::x` or `coord::y`
 * @param lim           Tile size along chosen direction
 * @param d_tiles       Tile information (main buffer)
 * @param d_ix          Particle cells (main buffer)
 * @param d_x           Particle positions (main buffer)
 * @param d_u           Particle momenta (main buffer)
 * @param tmp_d_tiles   Tile information (temp buffer)
 * @param tmp_d_ix      Particle cells (temp buffer)
 * @param tmp_d_x       Particle positions (temp buffer)
 * @param tmp_d_u       Particle momenta (temp buffer)
 */
template < coord :: cart dir > 
__global__
void _bnd_in( int const lim,
    t_part_tile * const __restrict__ d_tiles,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u,
    t_part_tile * const __restrict__ tmp_d_tiles,
    int2 * __restrict__ tmp_d_ix, float2 * __restrict__ tmp_d_x, float3 * __restrict__ tmp_d_u )
{

    auto grid  = cg::this_grid(); 
    auto block = cg::this_thread_block();

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int n0  = d_tiles[ tid ].n;
    const int offset =  d_tiles[ tid ].pos;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    // Copy from upper neighbour
    {
        int uid = tid;
        
        if ( dir == coord::x ) uid +=  1 - (( blockIdx.x < gridDim.x - 1 ) ? 0 : gridDim.x );
        if ( dir == coord::y ) uid += (1 - (( blockIdx.y < gridDim.y - 1 ) ? 0 : gridDim.y )) * gridDim.x;

        unsigned int nu = tmp_d_tiles[ uid ].n;
        const int upper_offset =  tmp_d_tiles[ uid ].pos;
        int2   __restrict__ *upper_ix = &tmp_d_ix[ upper_offset ];
        float2 __restrict__ *upper_x  = &tmp_d_x[ upper_offset ];
        float3 __restrict__ *upper_u  = &tmp_d_u[ upper_offset ];

        for( int i = block.thread_rank(); i < nu; i+= block.num_threads() ) {
            int2 t = upper_ix[i];

            if ( dir == coord::x ) t.x += lim;
            if ( dir == coord::y ) t.y += lim;

            ix[ n0 + i ] = t;
            x[ n0 + i ]  = upper_x[i];
            u[ n0 + i ]  = upper_u[i];
        }
        n0 += nu;
    }

    // Copy from lower neighbour
    {
        int lid = tid;
        
        if ( dir == coord::x ) lid +=  (( blockIdx.x > 0 ) ? 0 : gridDim.x) - 1;
        if ( dir == coord::y ) lid += ((( blockIdx.y > 0 ) ? 0 : gridDim.y) - 1) * gridDim.x;

        unsigned int k  = tmp_d_tiles[ lid ].n;
        unsigned int nl = tmp_d_tiles[ lid ].nb;
        const int lower_offset =  tmp_d_tiles[ lid ].pos;
        int2   __restrict__ *lower_ix = &tmp_d_ix[ lower_offset ];
        float2 __restrict__ *lower_x  = &tmp_d_x[ lower_offset ];
        float3 __restrict__ *lower_u  = &tmp_d_u[ lower_offset ];

        for( int i = block.thread_rank(); i < nl; i+= block.num_threads() ) {
            int2 t = lower_ix[k+i];
            
            if ( dir == coord::x ) t.x -= lim;
            if ( dir == coord::y ) t.y -= lim;

            ix[ n0 + i ] = t;
            x[ n0 + i ]  = lower_x[k+i];
            u[ n0 + i ]  = lower_u[k+i];
        }
        n0 += nl;
    }

    if ( block.thread_rank() == 0 ) d_tiles[ tid ].n = n0;
}


void Particles::tile_sort( Particles &tmp ) {
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block = dim3( 1024 );

    _bnd_out< coord::x > <<< grid, block >>> ( 
        nx.x, 
        tiles, ix, x, u,
        tmp.tiles, tmp.ix, tmp.x, tmp.u
    );

    _bnd_in< coord::x >  <<< grid, block >>> ( 
        nx.x,
        tiles, ix, x, u,
        tmp.tiles, tmp.ix, tmp.x, tmp.u
    );

    _bnd_out< coord::y > <<< grid, block >>> ( 
        nx.y, 
        tiles, ix, x, u,
        tmp.tiles, tmp.ix, tmp.x, tmp.u
     );

    _bnd_in< coord::y >  <<< grid, block >>> ( 
        nx.y,
        tiles, ix, x, u,
        tmp.tiles, tmp.ix, tmp.x, tmp.u
    );
}

__host__
/**
 * @brief Moves particles to the correct tiles
 * 
 * Note that particles are only expected to have moved no more than 1 tile
 * in each direction
 *
 */
void Particles::tile_sort() {

    Particles tmp( ntiles, nx, max_np_tile );
    tile_sort( tmp );
}


#define __ULIM __FLT_MAX__

__global__
/**
 * @brief Checks particle buffer data for error
 * 
 * WARNING: This routine is meant for debug only and should not be called 
 *          for production code.
 * 
 * The routine will check for:
 *      1. Invalid cell data (out of tile bounds)
 *      2. Invalid position data (out of [-0.5,0.5[)
 *      3. Invalid momenta (nan, inf or above __ULIM macro value)
 * 
 * If there are any errors found the routine will exit the code.
 * 
 * @param tiles 
 * @param d_ix 
 * @param d_x 
 * @param d_u 
 * @param nx 
 * @param over 
 * @param out 
 */
void _validate( t_part_tile * tiles, 
    int2   const * const __restrict__ d_ix,
    float2 const * const __restrict__ d_x,
    float3 const * const __restrict__ d_u,
    uint2 const nx, int const over, int * out ) {

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const offset = tiles[ tid ].pos;
    int const np     = tiles[ tid ].n;
    int2   const * const __restrict__ ix = &d_ix[ offset ];
    float2 const * const __restrict__ x  = &d_x[ offset ];
    float3 const * const __restrict__ u  = &d_u[ offset ];

    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( nx.x + over, nx.y + over ); 

    for( int i = threadIdx.x; i < np; i += blockDim.x) {
        int err = 0;

        if ( (ix[i].x < lb.x) || (ix[i].x >= ub.x )) {
            printf("(*error*) Invalid ix[%d].x position (%d), range = [%d,%d]\n", i, ix[i].x, lb.x, ub.x );
            err = 1;
        }
        if ( (ix[i].y < lb.y) || (ix[i].y >= ub.y )) {
            printf("(*error*) Invalid ix[%d].y position (%d), range = [%d,%d]\n", i, ix[i].y, lb.y, ub.y );
            err = 1;
        }

        if ( isnan(u[i].x) || isinf(u[i].x) || fabsf(u[i].x) >= __ULIM ) {
            printf("(*error*) Invalid u[%d].x gen. velocity (%f)\n", i, u[i].x );
            err = 1;
        }

        if ( isnan(u[i].y) || isinf(u[i].y) || fabsf(u[i].x) >= __ULIM ) {
            printf("(*error*) Invalid u[%d].y gen. velocity (%f)\n", i, u[i].y );
            err = 1;
        }

        if ( isnan(u[i].z) || isinf(u[i].z) || fabsf(u[i].x) >= __ULIM ) {
            printf("(*error*) Invalid u[%d].z gen. velocity (%f)\n", i, u[i].z );
            err = 1;
        }

        if ( x[i].x < -0.5f || x[i].x >= 0.5f ) {
            printf("(*error*) Invalid x[%d].x position (%f), range = [-0.5,0.5[\n", i, x[i].x );
            err = 1;
        }
        if ( x[i].y < -0.5f || x[i].y >= 0.5f ) {
            printf("(*error*) Invalid x[%d].y position (%f), range = [-0.5,0.5[\n", i, x[i].y );
            err = 1;
        }

        if ( err ) {
            atomicAdd( out, 1 );
            break;
        }
    }
}

#undef __ULIM

__host__
/**
 * @brief Checks if particle buffer data is correct
 * 
 */
void Particles::validate( std::string msg, int const over ) {

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 32 );

    device::Var<int> err(0);

    _validate <<< grid, block >>> ( tiles, ix, x, u, nx, over, err.ptr() );

    unsigned int nerr = err.get();
    if ( nerr > 0 ) {
        std::cerr << "(*error*) " << msg << "\n";
        std::cerr << "(*error*) invalid particle, aborting...\n";
        exit(1);
    }
}

void Particles::validate( std::string msg ) {

    validate( msg, 0 );

}

void Particles::check_tiles() {

    ParticlesTile * h_tiles;
    malloc_host( h_tiles, ntiles.x * ntiles.y );

    devhost_memcpy( h_tiles, tiles, ntiles.x * ntiles.y );

    int np = 0;
    int max = 0;

    for( int i = 0; i < ntiles.x * ntiles.y; i++ ) {
        np += h_tiles[i].n;
        if ( h_tiles[i].n > max ) max = h_tiles[i].n;
    }

    printf("(*info*) #part tile: %g (avg), %d (max), %d (lim)\n", 
        float(np) / (ntiles.x * ntiles.y), max, max_np_tile );

    if ( max >= 0.9 * max_np_tile ) {
        printf("(*critical*) Buffer almost full!\n");
        exit(1);
    }

    free_host( h_tiles );
}