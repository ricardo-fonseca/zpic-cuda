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
 * @brief Gather x quantity data
 * 
 * @param d_ix 
 * @param d_x 
 * @param d_tile 
 * @param d_out_offset 
 * @param d_data 
 */
__global__
void _gather_x_kernel( int2 * const __restrict__ d_ix, float2 * const __restrict__ d_x, 
    t_part_tile const * const __restrict__ d_tiles, unsigned int const tile_nx,
    unsigned int const * const __restrict__ d_out_offset, float * const __restrict__ d_data ) {
    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;

    int2   __restrict__ * const ix = &d_ix[ offset ];
    float2 __restrict__ * const x  = &d_x[ offset ];
    
    unsigned int const out_offset = d_out_offset[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = (blockIdx.x * tile_nx + ix[idx].x) + (0.5f + x[idx].x);
    }
};

__global__
void _gather_y_kernel( int2 * const __restrict__ d_ix, float2 * const __restrict__ d_x, 
    t_part_tile const * const __restrict__ d_tiles, unsigned int const tile_ny,
    unsigned int const * const __restrict__ d_out_offset, float * const __restrict__ d_data ) {
    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;

    int2   __restrict__ * const ix = &d_ix[ offset ];
    float2 __restrict__ * const x  = &d_x[ offset ];
    
    unsigned int const out_offset = d_out_offset[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = ( blockIdx.y * tile_ny + ix[idx].y ) + ( 0.5f + x[idx].y);
    }
};

__global__
void _gather_ux_kernel( float3 * const __restrict__ d_u, 
    t_part_tile const * const __restrict__ d_tiles,
    unsigned int const * const __restrict__ d_out_offset, float * const __restrict__ d_data ) {
    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;

    float3 __restrict__ * const u = &d_u[ offset ];
    unsigned int const out_offset = d_out_offset[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = u[idx].x;
    }
};

__global__
void _gather_uy_kernel( float3 * const __restrict__ d_u, 
    t_part_tile const * const __restrict__ d_tiles,
    unsigned int const * const __restrict__ d_out_offset, float * const __restrict__ d_data ) {
    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;

    float3 __restrict__ * const u = &d_u[ offset ];
    unsigned int const out_offset = d_out_offset[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = u[idx].y;
    }
};

__global__
void _gather_uz_kernel( float3 * const __restrict__ d_u, 
    t_part_tile const * const __restrict__ d_tiles,
    unsigned int const * const __restrict__ d_out_offset, float * const __restrict__ d_data ) {
    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;

    float3 __restrict__ * const u = &d_u[ offset ];
    unsigned int const out_offset = d_out_offset[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = u[idx].z;
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
        dim3 block( 64 );

        // Gather data on device
        switch (quant) {
        case part::quant::x :
            _gather_x_kernel <<< grid, block >>> ( ix, x, tiles, nx.x, d_out_offset, d_data );
            break;
        case part::quant::y:
            _gather_y_kernel <<< grid, block >>> ( ix, x, tiles, nx.y, d_out_offset, d_data );
            break;
        case part::quant::ux:
            _gather_ux_kernel <<< grid, block >>> ( u, tiles, d_out_offset, d_data );
            break;
        case part::quant::uy:
            _gather_uy_kernel <<< grid, block >>> ( u, tiles, d_out_offset, d_data );
            break;
        case part::quant::uz:
            _gather_uz_kernel <<< grid, block >>> ( u, tiles, d_out_offset, d_data );
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

    float * d_data = nullptr;
    if ( np > 0 ) malloc_dev( d_data, np );

    gather( quant, h_data, d_data, np, d_out_offset );

    free_dev( d_data );
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
    float *h_data, *d_data;
    malloc_host( h_data, np );
    malloc_dev( d_data, np );

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
    free_dev( d_data );
    free_host( h_data );
    free_dev( d_out_offset );
}

__global__
/**
 * @brief CUDA kernel for copying particles out of the tile (x) to a temp buffer
 * 
 * This should be called with 1 block per tile and 1 warp per block
 * 
 * @param lim           Tile size along x direction
 * @param d_tiles       Tile information (main buffer)
 * @param d_ix          Particle cells (main buffer)
 * @param d_x           Particle positions (main buffer)
 * @param d_u           Particle momenta (main buffer)
 * @param tmp_d_tiles   Tile information (temp buffer)
 * @param tmp_d_ix      Particle cells (temp buffer)
 * @param tmp_d_x       Particle positions (temp buffer)
 * @param tmp_d_u       Particle momenta (temp buffer)
 */
void _bnd_out_x( int const lim, 
    t_part_tile * const __restrict__ d_tiles,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u,
    t_part_tile * const __restrict__ tmp_d_tiles,
    int2 * __restrict__ tmp_d_ix, float2 * __restrict__ tmp_d_x, float3 * __restrict__ tmp_d_u )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    unsigned int smask = (0x7FFFFFFF >> (0x1F - warp.thread_rank()));

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

    // Count number of particles with keys = 1, 2
    // When sorting multiple datasets with the same key this can (should)
    // be cached, mainly to avoid reading d_key[] one more time per dataset
    unsigned int n2 = 0, n1 = 0;
    for( unsigned int i = warp.thread_rank(); i < size; i+= warp.num_threads() ) {
        int const key = ( ix[i].x < 0 ) + 2 * ( ix[i].x >= lim );
        n2 += ( key == 2 );
        n1 += ( key == 1 );
    }
    n2 = cg::reduce( warp, n2, cg::plus<unsigned int>());
    n1 = cg::reduce( warp, n1, cg::plus<unsigned int>());

    unsigned int const nmove = n2 + n1;
    unsigned int const n0 = size - nmove;

    // Copy values that are moving to temp buffer
    {
        int k2 = n1, k1 = 0, k0 = nmove;
        for( int i = warp.thread_rank(); i < size; i+= warp.num_threads() ) {
            int const key = ( ix[i].x < 0 ) + 2 * ( ix[i].x >= lim );
            
            int const c2 = ( key == 2 );
            int const c1 = ( key == 1 );
            int const c0 = ( key == 0 ) && ( i >= n0 ); // Used to fill holes

            auto const b2 = warp.ballot( c2 );
            auto const b1 = warp.ballot( c1 );
            auto const b0 = warp.ballot( c0 );

            auto const s2 = __popc( b2 & smask );
            auto const s1 = __popc( b1 & smask );
            auto const s0 = __popc( b0 & smask );

            if ( c2 || c1 || c0 ) {
                int const idx = c2 * ( k2 + s2 ) + 
                                c1 * ( k1 + s1 ) + 
                                c0 * ( k0 + s0 );
                tmp_ix[idx] = ix[i];
                tmp_x[idx]  = x[i];
                tmp_u[idx]  = u[i];
            }

            k2 += __popc(b2);
            k1 += __popc(b1);
            k0 += __popc(b0);
        }
    }

    // Fill holes left behind
    {
        warp.sync();
        int k = 0;
        for( int i = warp.thread_rank(); i < n0; i+= warp.num_threads() ) {
            int  const c = ( ix[i].x < 0 ) || ( ix[i].x >= lim );
            auto const b = warp.ballot( c );
            auto const s = __popc( b & smask );
            if ( c ) {
                ix[i] = tmp_ix[ nmove + k + s ];
                x[i] = tmp_x[ nmove + k + s ];
                u[i] = tmp_u[ nmove + k + s ];
            }
            k += __popc(b);
        }
    }

    // Store new values on tile information
    if ( warp.thread_rank() == 0 ) {
        d_tiles[ tid ].n  = n0;
        tmp_d_tiles[ tid ].n  = n1;
        tmp_d_tiles[ tid ].nb = n2;
    }
}


__global__
/**
 * @brief CUDA kernel for copying in particles that moved out of neighboring
 * tiles into the local tile.
 * 
 * This should be called with 1 block per tile and 1 warp per block
 * 
 * @param self  Particle buffer
 * @param tmp   Particle buffer holding particles that have left their tile (x)
 */
void _bnd_in_x( int const lim,
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
        int uid = tid + 1 - (( blockIdx.x < gridDim.x -1 ) ? 0 : gridDim.x );

        unsigned int nu = tmp_d_tiles[ uid ].n;
        const int upper_offset =  tmp_d_tiles[ uid ].pos;
        int2   __restrict__ *upper_ix = &tmp_d_ix[ upper_offset ];
        float2 __restrict__ *upper_x  = &tmp_d_x[ upper_offset ];
        float3 __restrict__ *upper_u  = &tmp_d_u[ upper_offset ];

        for( int i = block.thread_rank(); i < nu; i+= block.num_threads() ) {
            int2 t = upper_ix[i];
            t.x += lim;
            ix[ n0 + i ] = t;
            x[ n0 + i ]  = upper_x[i];
            u[ n0 + i ]  = upper_u[i];
        }
        n0 += nu;
    }

    // Copy from lower neighbour
    {
        int lid = tid - 1 + (( blockIdx.x > 0 ) ? 0 : gridDim.x);

        unsigned int k  = tmp_d_tiles[ lid ].n;
        unsigned int nl = tmp_d_tiles[ lid ].nb;
        const int lower_offset =  tmp_d_tiles[ lid ].pos;
        int2   __restrict__ *lower_ix = &tmp_d_ix[ lower_offset ];
        float2 __restrict__ *lower_x  = &tmp_d_x[ lower_offset ];
        float3 __restrict__ *lower_u  = &tmp_d_u[ lower_offset ];

        for( int i = block.thread_rank(); i < nl; i+= block.num_threads() ) {
            int2 t = lower_ix[k+i];
            t.x -= lim;
            ix[ n0 + i ] = t;
            x[ n0 + i ]  = lower_x[k+i];
            u[ n0 + i ]  = lower_u[k+i];
        }
        n0 += nl;
    }

    if ( block.thread_rank() == 0 ) d_tiles[ tid ].n = n0;
}

__global__
/**
 * @brief CUDA kernel for copying particles out of the tile (y) to a temp buffer
 * 
 * This should be called with 1 block per tile and 1 warp per block
 * 
 * @param lim           Tile size along y direction
 * @param d_tiles       Tile information (main buffer)
 * @param d_ix          Particle cells (main buffer)
 * @param d_x           Particle positions (main buffer)
 * @param d_u           Particle momenta (main buffer)
 * @param tmp_d_tiles   Tile information (temp buffer)
 * @param tmp_d_ix      Particle cells (temp buffer)
 * @param tmp_d_x       Particle positions (temp buffer)
 * @param tmp_d_u       Particle momenta (temp buffer)
 */
void _bnd_out_y( int const lim, 
    t_part_tile * const __restrict__ d_tiles,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u,
    t_part_tile * const __restrict__ tmp_d_tiles,
    int2 * __restrict__ tmp_d_ix, float2 * __restrict__ tmp_d_x, float3 * __restrict__ tmp_d_u )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    unsigned int smask = (0x7FFFFFFF >> (0x1F - warp.thread_rank()));

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

    // Count number of particles with keys = 1, 2
    // When sorting multiple datasets with the same key this can (should)
    // be cached, mainly to avoid reading d_key[] one more time per dataset
    unsigned int n2 = 0, n1 = 0;
    for( unsigned int i = warp.thread_rank(); i < size; i+= warp.num_threads() ) {
        int const key = ( ix[i].y < 0 ) + 2 * ( ix[i].y >= lim );
        n2 += ( key == 2 );
        n1 += ( key == 1 );
    }
    n2 = cg::reduce( warp, n2, cg::plus<unsigned int>());
    n1 = cg::reduce( warp, n1, cg::plus<unsigned int>());

    unsigned int const nmove = n2 + n1;
    unsigned int const n0 = size - nmove;

    // Copy values that are moving to temp buffer
    {
        int k2 = n1, k1 = 0, k0 = nmove;
        for( int i = warp.thread_rank(); i < size; i+= warp.num_threads() ) {
            int const key = ( ix[i].y < 0 ) + 2 * ( ix[i].y >= lim );
            
            int const c2 = ( key == 2 );
            int const c1 = ( key == 1 );
            int const c0 = ( key == 0 ) && ( i >= n0 ); // Used to fill holes

            auto const b2 = warp.ballot( c2 );
            auto const b1 = warp.ballot( c1 );
            auto const b0 = warp.ballot( c0 );

            auto const s2 = __popc( b2 & smask );
            auto const s1 = __popc( b1 & smask );
            auto const s0 = __popc( b0 & smask );

            if ( c2 || c1 || c0 ) {
                int const idx = c2 * ( k2 + s2 ) + 
                                c1 * ( k1 + s1 ) + 
                                c0 * ( k0 + s0 );
                tmp_ix[idx] = ix[i];
                tmp_x[idx]  = x[i];
                tmp_u[idx]  = u[i];
            }

            k2 += __popc(b2);
            k1 += __popc(b1);
            k0 += __popc(b0);
        }
    }

    // Fill holes left behind
    {
        warp.sync();
        int k = 0;
        for( int i = warp.thread_rank(); i < n0; i+= warp.num_threads() ) {
            int  const c = ( ix[i].y < 0 ) || ( ix[i].y >= lim );
            auto const b = warp.ballot( c );
            auto const s = __popc( b & smask );
            if ( c ) {
                ix[i] = tmp_ix[ nmove + k + s ];
                x[i] = tmp_x[ nmove + k + s ];
                u[i] = tmp_u[ nmove + k + s ];
            }
            k += __popc(b);
        }
    }

    // Store new values on tile information
    if ( warp.thread_rank() == 0 ) {
        d_tiles[ tid ].n  = n0;
        tmp_d_tiles[ tid ].n  = n1;
        tmp_d_tiles[ tid ].nb = n2;
    }
}


__global__
/**
 * @brief CUDA kernel for copying in particles that moved out of neighboring
 * tiles into the local tile.
 * 
 * This should be called with 1 block per tile and 1 warp per block
 * 
 * @param self  Particle buffer
 * @param tmp   Particle buffer holding particles that have left their tile (x)
 */
void _bnd_in_y( int const lim,
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
        int const uid = tid + (1 - ((blockIdx.y < ( gridDim.y - 1 )) ? 0 : gridDim.y)) * gridDim.x;

        unsigned int nu = tmp_d_tiles[ uid ].n;
        const int upper_offset =  tmp_d_tiles[ uid ].pos;
        int2   __restrict__ *upper_ix = &tmp_d_ix[ upper_offset ];
        float2 __restrict__ *upper_x  = &tmp_d_x[ upper_offset ];
        float3 __restrict__ *upper_u  = &tmp_d_u[ upper_offset ];

        for( int i = block.thread_rank(); i < nu; i+= block.num_threads() ) {
            int2 t = upper_ix[i];
            t.y += lim;
            ix[ n0 + i ] = t;
            x[ n0 + i ]  = upper_x[i];
            u[ n0 + i ]  = upper_u[i];
        }
        n0 += nu;
    }

    // Copy from lower neighbour
    {
        int lid = tid + (((blockIdx.y > 0) ? 0 : gridDim.y)-1) * gridDim.x;

        unsigned int k  = tmp_d_tiles[ lid ].n;
        unsigned int nl = tmp_d_tiles[ lid ].nb;
        const int lower_offset =  tmp_d_tiles[ lid ].pos;
        int2   __restrict__ *lower_ix = &tmp_d_ix[ lower_offset ];
        float2 __restrict__ *lower_x  = &tmp_d_x[ lower_offset ];
        float3 __restrict__ *lower_u  = &tmp_d_u[ lower_offset ];

        for( int i = block.thread_rank(); i < nl; i+= block.num_threads() ) {
            int2 t = lower_ix[k+i];
            t.y -= lim;
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
    dim3 block( 32 );

    _bnd_out_x <<< grid, block >>> ( 
        nx.x, 
        tiles, ix, x, u,
        tmp.tiles, tmp.ix, tmp.x, tmp.u
    );

    _bnd_in_x  <<< grid, block >>> ( 
        nx.x,
        tiles, ix, x, u,
        tmp.tiles, tmp.ix, tmp.x, tmp.u
    );

    _bnd_out_y <<< grid, block >>> ( 
        nx.y, 
        tiles, ix, x, u,
        tmp.tiles, tmp.ix, tmp.x, tmp.u
     );

    _bnd_in_y  <<< grid, block >>> ( 
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