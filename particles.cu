#include "particles.cuh"
#include <iostream>
#include <string>

#include "util.cuh"

#include "timer.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg=cooperative_groups;


__global__
void _init_tiles_kernel( 
    int * const __restrict__ d_tile_offset, 
    int * const __restrict__ d_tile_np, 
    int * const __restrict__ d_tile_np2, 
    unsigned int const max_np_tile ) {

    const int i = blockIdx.y * gridDim.x + blockIdx.x;

    d_tile_offset[i] = i * max_np_tile;
    d_tile_np[i]  = 0;
    d_tile_np2[i] = 0;
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
    ntiles( ntiles ), nx( nx ), max_np_tile( max_np_tile ), periodic( make_int2(1,1) )
{    
    size_t size = ntiles.x * ntiles.y * max_np_tile;
    malloc_dev( ix, size );
    malloc_dev( x, size );
    malloc_dev( u, size );

    malloc_dev( idx, size );

    // Allocate tile information array on device and initialize using a CUDA kernel

    malloc_dev( tile_offset, ntiles.x * ntiles.y );
    malloc_dev( tile_np, ntiles.x * ntiles.y );
    malloc_dev( tile_np2, ntiles.x * ntiles.y );

    dim3 grid( ntiles.x, ntiles.y );
    _init_tiles_kernel <<< grid, 1 >>> ( tile_offset, tile_np, tile_np2, max_np_tile );
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
void _np_kernel( int const * const __restrict__ d_tile_np, 
    unsigned int const ntiles, unsigned int * const __restrict__ total) {
    auto group = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(group);

    unsigned int np = 0;
    for( int i = group.thread_rank(); i < ntiles; i += group.num_threads() )
        np += d_tile_np[i];
    
    np = cg::reduce( warp, np, cg::plus<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( total, np );
}

__host__
/**
 * @brief Gets total number of particles on device
 * 
 * @return unsigned long long   Total number of particles
 */
unsigned int Particles::np() {
    _dev_tmp_uint = 0;
    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    _np_kernel <<< grid, block >>> ( tile_np, size, _dev_tmp_uint.ptr() );
    return _dev_tmp_uint.get();
}

__global__
void _np_max_tile( int const * const __restrict__ d_tile_np, 
    unsigned int const ntiles, unsigned int * const __restrict__ max) {
    auto group = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(group);

    unsigned int v = 0;
    for( int i = group.thread_rank(); i < ntiles; i += group.num_threads() ) {
        int tile_np = d_tile_np[i];
        if ( tile_np > v ) v = tile_np;
    }
    
    v = cg::reduce( warp, v, cg::greater<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicMax( max, v );
}

__host__
/**
 * @brief Gets maximum number of particles per tile
 * 
 * @return unsigned int 
 */
unsigned int Particles::np_max_tile() {
    _dev_tmp_uint = 0;
    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    _np_max_tile <<< grid, block >>> ( tile_np, size, _dev_tmp_uint.ptr() );
    return _dev_tmp_uint.get();
}

__global__
void _np_min_tile( int const * const __restrict__ d_tile_np, 
    unsigned int const ntiles, unsigned int * const __restrict__ max) {
    auto group = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(group);

    unsigned int v = 0;
    for( int i = group.thread_rank(); i < ntiles; i += group.num_threads() ) {
        int tile_np = d_tile_np[i];
        if ( tile_np > v ) v = tile_np;
    }
    
    v = cg::reduce( warp, v, cg::less<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicMin( max, v );
}

__host__
/**
 * @brief Gets minimum number of particles per tile
 * 
 * @return unsigned int 
 */
unsigned int Particles::np_min_tile() {
    _dev_tmp_uint = 0;
    auto size = ntiles.x*ntiles.y;
    auto block = ( size < 1024 ) ? size : 1024 ;
    auto grid = (size-1)/block + 1;
    _np_max_tile <<< grid, block >>> ( tile_np, size, _dev_tmp_uint.ptr() );
    return _dev_tmp_uint.get();
}

__global__
void _np_exscan_kernel( 
    unsigned int * const __restrict__ idx,
    int const * const __restrict__ d_tile_np, unsigned int const ntiles,
    unsigned int * const __restrict__ total) {

    __shared__ unsigned int tmp[ 32 ];
    __shared__ unsigned int prev;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    prev = 0;

    for( unsigned int i = block.thread_rank(); i < ntiles; i += block.num_threads() ) {
        unsigned int s = d_tile_np[i];

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
    _dev_tmp_uint = 0;
    _np_exscan_kernel <<< 1, block >>> ( d_offset, tile_np, size, _dev_tmp_uint.ptr() );
    return _dev_tmp_uint.get();
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
    int const * const __restrict__ d_tile_offset, 
    int const * const __restrict__ d_tile_np, 
    uint2 const tile_nx,
    unsigned int const * const __restrict__ d_out_offset, 
    float * const __restrict__ d_data )
{    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tile_offset[tid];
    const int np     = d_tile_np[tid];

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
            _gather_quant<part::x> <<<grid,block>>>( ix, x, u, tile_offset, tile_np, nx, d_out_offset, d_data );
            break;
        case part::y:
            _gather_quant<part::y> <<<grid,block>>>( ix, x, u, tile_offset, tile_np, nx, d_out_offset, d_data );
            break;
        case part::ux:
            _gather_quant<part::ux> <<<grid,block>>>( ix, x, u, tile_offset, tile_np, nx, d_out_offset, d_data );
            break;
        case part::uy:
            _gather_quant<part::uy> <<<grid,block>>>( ix, x, u, tile_offset, tile_np, nx, d_out_offset, d_data );
            break;
        case part::uz:
            _gather_quant<part::uz> <<<grid,block>>>( ix, x, u, tile_offset, tile_np, nx, d_out_offset, d_data );
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
 * @param tmp_d_u       Particle generalized velocity (temp buffer)
 */
template < coord::cart dir >
__global__
void _bnd_out( int const lim, 
    int * const __restrict__ d_tile_np, int * const __restrict__ d_tile_offset,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u, int * __restrict__ d_idx,
    int * const __restrict__ d_tmp_tile_np, int * const __restrict__ d_tmp_tile_offset, int * const __restrict__ d_tmp_tile_np2, 
    int2 * __restrict__ tmp_d_ix, float2 * __restrict__ tmp_d_x, float3 * __restrict__ tmp_d_u )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const np = d_tile_np[ tid ];
    unsigned int const offset =  d_tile_offset[ tid ];
    int2   * __restrict__ ix  = &d_ix[ offset ];
    float2 * __restrict__ x   = &d_x[ offset ];
    float3 * __restrict__ u   = &d_u[ offset ];

    int * __restrict__ idx = &d_idx[ offset ];

    unsigned int const tmp_offset =  d_tmp_tile_offset[ tid ];
    int2   * __restrict__ tmp_ix = &tmp_d_ix[ tmp_offset ];
    float2 * __restrict__ tmp_x  = &tmp_d_x[ tmp_offset ];
    float3 * __restrict__ tmp_u  = &tmp_d_u[ tmp_offset ];


    // Total number of particles leaving
    __shared__ int _nmove;

    // Number of particle leaving to the left neighbour
    __shared__ int _n1;

    _nmove = 0;
    _n1 = 0;

    block.sync();

    int n1 = 0;

    for( int i = block.thread_rank(); i < np; i+= block.num_threads() ) {
        int ipos;
        if ( dir == coord::x ) ipos = ix[i].x;
        if ( dir == coord::y ) ipos = ix[i].y;

        n1 += ( ipos < 0 );

        if ( ( ipos < 0 ) || ( ipos >= lim ) ) {
            int k = atomicAdd( &_nmove, 1 );
            idx[k] = i;
        }
    }

    n1 = cg::reduce( warp, n1, cg::plus<int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( &_n1, n1 );

    block.sync();

    // Number of particles staying in node
    int const _n0 = np - _nmove;

    // Number of particles leaving to the right neighbour
    int const _n2 = _nmove - _n1;

    // Indices for tmp partilce buffer ( 0 - lower, 1 - upper )
    __shared__ int _k[2];
    _k[0] =   0;
    _k[1] = _n1;

    // Index for copying back particles to fill holes
    __shared__ int _c;
    _c = _n0;

    block.sync();

    for( int i = block.thread_rank(); i < _nmove; i+= block.num_threads() ) {
        int src = idx[i];

        int ipos;
        if ( dir == coord::x ) ipos = ix[src].x;
        if ( dir == coord::y ) ipos = ix[src].y;

        int bnd = ( ipos >= lim );

        int k = atomicAdd( & _k[bnd], 1 );

        tmp_ix[k] = ix[src];
        tmp_x[k]  = x[src];
        tmp_u[k]  = u[src];
    }

    block.sync();

    for( int i = block.thread_rank(); i < _nmove; i+= block.num_threads() ) {

        int tgt = idx[i];

        if ( tgt < _n0 ) {
            int c, ipos;

            do {
                c = atomicAdd( &_c, 1 );
                if ( dir == coord::x ) ipos = ix[c].x;
                if ( dir == coord::y ) ipos = ix[c].y;
            } while (( ipos < 0 ) || ( ipos >= lim ));

            ix[ tgt ] = ix[c];
            x[ tgt ]  = x[c];
            u[ tgt ]  = u[c];
        }
    }

    // Store new values on tile information
    if ( block.thread_rank() == 0 ) {
        d_tile_np[ tid ]         = _n0;
        d_tmp_tile_np[ tid ]     = _n1;
        d_tmp_tile_np2[ tid ]    = _n2;
    }

}


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
template < coord::cart dir > 
__global__
void _bnd_in( int const lim,
    int * const __restrict__ d_tile_np, int * const __restrict__ d_tile_offset,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u,
    int * const __restrict__ d_tmp_tile_np, int * const __restrict__ d_tmp_tile_offset, int * const __restrict__ d_tmp_tile_np2,
    int2 * __restrict__ tmp_d_ix, float2 * __restrict__ tmp_d_x, float3 * __restrict__ tmp_d_u,
    int const periodic )
{

    auto grid  = cg::this_grid(); 
    auto block = cg::this_thread_block();

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int n0  = d_tile_np[ tid ];
    const int offset =  d_tile_offset[ tid ];
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    // Copy from upper neighbour
    int x_ucoord = blockIdx.x;
    int x_lcoord = blockIdx.x;

    if ( dir == coord::x ) {
        x_lcoord -= 1;
        x_ucoord += 1;
        if ( periodic ) {
            if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
            if ( x_ucoord >= gridDim.x ) x_ucoord -= gridDim.x;
        }
    }

    int y_ucoord = blockIdx.y;
    int y_lcoord = blockIdx.y;

    if ( dir == coord::y ) {
        y_lcoord -= 1;
        y_ucoord += 1;
        if ( periodic ) {
            if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
            if ( y_ucoord >= gridDim.y ) y_ucoord -= gridDim.y;
        }
    }

    if (( x_ucoord < gridDim.x ) && 
        ( y_ucoord < gridDim.y )) {

        int uid = y_ucoord * gridDim.x + x_ucoord;

        unsigned int nu = d_tmp_tile_np[ uid ];
        const int upper_offset =  d_tmp_tile_offset[ uid ];
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
    if (( x_lcoord >= 0 ) && 
        ( y_lcoord >= 0 )) {

        int lid = y_lcoord * gridDim.x + x_lcoord;;
        
        unsigned int k  = d_tmp_tile_np[ lid ];
        unsigned int nl = d_tmp_tile_np2[ lid ];
        const int lower_offset =  d_tmp_tile_offset[ lid ];
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

    if ( block.thread_rank() == 0 ) d_tile_np[ tid ] = n0;
}

/**
 * @brief Moves particles to the correct tiles
 * 
 * Note that particles are only expected to have moved no more than 1 tile
 * in each direction
 * 
 * @param tmp   Temporary buffer to hold particles moving out of tiles. This
 *              buffer *MUST* be big enough to hold all the particles moving
 *              out of the tiles. It's size is not checked.
 */
__host__
void Particles::tile_sort( Particles &tmp ) {
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    _bnd_out< coord::x > <<< grid, block >>> ( 
        nx.x, tile_np, tile_offset,
        ix, x, u, idx,
        tmp.tile_np, tmp.tile_offset, tmp.tile_np2,
        tmp.ix, tmp.x, tmp.u
    );

    _bnd_in< coord::x >  <<< grid, block >>> ( 
        nx.x, tile_np, tile_offset, ix, x, u,
        tmp.tile_np, tmp.tile_offset, tmp.tile_np2, tmp.ix, tmp.x, tmp.u,
        periodic.x
    );

    _bnd_out< coord::y > <<< grid, block >>> ( 
        nx.y, tile_np, tile_offset,
        ix, x, u, idx,
        tmp.tile_np, tmp.tile_offset, tmp.tile_np2,
        tmp.ix, tmp.x, tmp.u
     );

    _bnd_in< coord::y >  <<< grid, block >>> ( 
        nx.y, tile_np, tile_offset,
        ix, x, u,
        tmp.tile_np, tmp.tile_offset, tmp.tile_np2, tmp.ix, tmp.x, tmp.u,
        periodic.y
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

    // Create temporary buffer
    Particles tmp( ntiles, nx, max_np_tile );

    tile_sort( tmp );
}

__global__
void _cell_shift( int * __restrict__ d_tile_np, int * __restrict__ d_tile_offset, 
    int2 * const __restrict__ d_ix,
    int2 const shift )
{
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const offset = d_tile_offset[ tid ];
    int const np     = d_tile_np[ tid ];
    int2 * const __restrict__ ix = &d_ix[ offset ];

    for( int i = threadIdx.x; i < np; i += blockDim.x) {
        int2 cell = ix[i];
        cell.x += shift.x;
        cell.y += shift.y;
        ix[i] = cell;
    }
}

/**
 * @brief Shifts particle cells by the required amount
 * 
 * Cells are shited by adding the parameter `shift` to the particle cell
 * indexes.
 * 
 * Note that this routine does not check if the particles are still inside the
 * tile.
 * 
 * @param shift     Cell shift in both directions
 */
void Particles::cell_shift( int2 const shift ) {

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    _cell_shift <<< grid, block >>> ( tile_np, tile_offset, ix, shift );
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
void _validate( 
    int const * const __restrict__ d_tile_np, 
    int const * const __restrict__ d_tile_offset, 
    int2   const * const __restrict__ d_ix,
    float2 const * const __restrict__ d_x,
    float3 const * const __restrict__ d_u,
    uint2 const nx, int const over, unsigned int * out ) {

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const offset = d_tile_offset[ tid ];
    int const np     = d_tile_np[ tid ];
    int2   const * const __restrict__ ix = &d_ix[ offset ];
    float2 const * const __restrict__ x  = &d_x[ offset ];
    float3 const * const __restrict__ u  = &d_u[ offset ];

    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( nx.x + over, nx.y + over ); 

    for( int i = threadIdx.x; i < np; i += blockDim.x) {
        int err = 0;

        if ( (ix[i].x < lb.x) || (ix[i].x >= ub.x )) {
            printf("(*error*) Invalid ix[%d].x position (%d), range = [%d,%d[\n", i, ix[i].x, lb.x, ub.x );
            err = 1;
        }
        if ( (ix[i].y < lb.y) || (ix[i].y >= ub.y )) {
            printf("(*error*) Invalid ix[%d].y position (%d), range = [%d,%d[\n", i, ix[i].y, lb.y, ub.y );
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


template < coord::cart dir >
__global__
void _validate_dir( 
    int    const * const __restrict__ d_tile_np, 
    int    const * const __restrict__ d_tile_offset, 
    int2   const * const __restrict__ d_ix,
    float2 const * const __restrict__ d_x,
    float3 const * const __restrict__ d_u,
    uint2 const nx, int const over, unsigned int * out ) {

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const offset = d_tile_offset[ tid ];
    int const np     = d_tile_np[ tid ];
    int2   const * const __restrict__ ix = &d_ix[ offset ];
    float2 const * const __restrict__ x  = &d_x[ offset ];
    float3 const * const __restrict__ u  = &d_u[ offset ];

    int2 const lb = make_int2( -over, -over );
    int2 const ub = make_int2( nx.x + over, nx.y + over ); 

    for( int i = threadIdx.x; i < np; i += blockDim.x) {
        int err = 0;

        if ( dir == coord::x ) {
            if ( (ix[i].x < lb.x) || (ix[i].x >= ub.x )) {
                printf("(*error*) Invalid ix[%d].x position (%d), range = [%d,%d[\n", i, ix[i].x, lb.x, ub.x );
                err = 1;
            }
            if ( x[i].x < -0.5f || x[i].x >= 0.5f ) {
                printf("(*error*) Invalid x[%d].x position (%f), range = [-0.5,0.5[\n", i, x[i].x );
                err = 1;
            }
        }

        if ( dir == coord::x ) {
            if ( (ix[i].y < lb.y) || (ix[i].y >= ub.y )) {
                printf("(*error*) Invalid ix[%d].y position (%d), range = [%d,%d[\n", i, ix[i].y, lb.y, ub.y );
                err = 1;
            }
            if ( x[i].y < -0.5f || x[i].y >= 0.5f ) {
                printf("(*error*) Invalid x[%d].y position (%f), range = [-0.5,0.5[\n", i, x[i].y );
                err = 1;
            }
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

        if ( err ) {
            atomicAdd( out, 1 );
            break;
        }
    }
}

#undef __ULIM

__host__
/**
 * @brief Validates particle data in buffer
 * 
 * Routine checks for valid positions (both cell index and cell position) and
 * for valid velocities
 * 
 * @param msg       Message to print in case error is found
 * @param over      Amount of extra cells indices beyond limit allowed. Used
 *                  when checking the buffer before tile_sort()
 */
void Particles::validate( std::string msg, int const over ) {

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 32 );

    _validate <<< grid, block >>> ( tile_np, tile_offset, ix, x, u, nx, over, _dev_tmp_uint.ptr() );

    unsigned int nerr = _dev_tmp_uint.get();
    if ( nerr > 0 ) {
        std::cerr << "(*error*) " << msg << "\n";
        std::cerr << "(*error*) invalid particle, aborting...\n";
        exit(1);
    }
}

void Particles::validate( std::string msg ) {

    validate( msg, 0 );

}

/**
 * @brief Check if tiles are full
 * 
 * OBSOLETE: To be removed
 * 
 */
void Particles::check_tiles() {

    int * h_tile_np;
    malloc_host( h_tile_np, ntiles.x * ntiles.y );

    devhost_memcpy( h_tile_np, tile_np, ntiles.x * ntiles.y );

    int np = 0;
    int max = 0;

    for( int i = 0; i < ntiles.x * ntiles.y; i++ ) {
        np += h_tile_np[i];
        if ( h_tile_np[i] > max ) max = h_tile_np[i];
    }

    printf("(*info*) #part tile: %g (avg), %d (max), %d (lim)\n", 
        float(np) / (ntiles.x * ntiles.y), max, max_np_tile );

    if ( max >= 0.9 * max_np_tile ) {
        printf("(*critical*) Buffer almost full!\n");
        exit(1);
    }

    free_host( h_tile_np );
}