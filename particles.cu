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
    t_part_tiles const tiles, 
    unsigned int const max_np_tile ) {

    const int i = blockIdx.y * gridDim.x + blockIdx.x;

    tiles.offset[i] = i * max_np_tile;
    tiles.np[i]  = 0;
    tiles.np2[i] = 0;
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

    std::cerr << "(*development*) Particle buffer size = " << size << "\n";

    malloc_dev( data.ix, size );
    malloc_dev( data.x, size );
    malloc_dev( data.u, size );

    malloc_dev( idx, size );

    // Allocate tile information array on device and initialize using a CUDA kernel

    malloc_dev( tiles.offset, ntiles.x * ntiles.y );
    malloc_dev( tiles.np, ntiles.x * ntiles.y );
    malloc_dev( tiles.np2, ntiles.x * ntiles.y );

    malloc_dev( tiles.nidx, ntiles.x * ntiles.y );

    malloc_dev( tiles.npt, 9 * ntiles.x * ntiles.y );

    dim3 grid( ntiles.x, ntiles.y );
    _init_tiles_kernel <<< grid, 1 >>> ( tiles, max_np_tile );
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
void _np_kernel( 
    int const * const __restrict__ d_tile_np, 
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
    _np_kernel <<< grid, block >>> ( tiles.np, size, _dev_tmp_uint.ptr() );
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
    _np_max_tile <<< grid, block >>> ( tiles.np, size, _dev_tmp_uint.ptr() );
    return _dev_tmp_uint.get();
}

__global__
void _np_min_tile( int const * const __restrict__ d_tiles_np, 
    unsigned int const ntiles, unsigned int * const __restrict__ max) {
    auto group = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(group);

    unsigned int v = 0;
    for( int i = group.thread_rank(); i < ntiles; i += group.num_threads() ) {
        int tile_np = d_tiles_np[i];
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
    _np_max_tile <<< grid, block >>> ( tiles.np, size, _dev_tmp_uint.ptr() );
    return _dev_tmp_uint.get();
}

__global__
void _np_exscan_kernel( 
    unsigned int * const __restrict__ idx,
    int const * const __restrict__ d_tiles_np, unsigned int const ntiles,
    unsigned int * const __restrict__ total) {

    __shared__ unsigned int tmp[ 32 ];
    __shared__ unsigned int prev;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    prev = 0;

    for( unsigned int i = block.thread_rank(); i < ntiles; i += block.num_threads() ) {
        unsigned int s = d_tiles_np[i];

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
    _np_exscan_kernel <<< 1, block >>> ( d_offset, tiles.np, size, _dev_tmp_uint.ptr() );
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
    t_part_data const data,
    t_part_tiles const tiles,
    uint2 const tile_nx,
    unsigned int const * const __restrict__ d_out_offset, 
    float * const __restrict__ d_data )
{    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = tiles.offset[tid];
    const int np     = tiles.np[tid];

    int2   __restrict__ const * const ix = &data.ix[ offset ];
    float2 __restrict__ const * const x  = &data.x[ offset ];
    float3 __restrict__ const * const u  = &data.u[ offset ];
    
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
            _gather_quant<part::x> <<<grid,block>>>( data, tiles, nx, d_out_offset, d_data );
            break;
        case part::y:
            _gather_quant<part::y> <<<grid,block>>>( data, tiles, nx, d_out_offset, d_data );
            break;
        case part::ux:
            _gather_quant<part::ux> <<<grid,block>>>( data, tiles, nx, d_out_offset, d_data );
            break;
        case part::uy:
            _gather_quant<part::uy> <<<grid,block>>>( data, tiles, nx, d_out_offset, d_data );
            break;
        case part::uz:
            _gather_quant<part::uz> <<<grid,block>>>( data, tiles, nx, d_out_offset, d_data );
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


__global__
void _bnd_out_check( int2 const lim, 
    t_part_tiles const tiles, t_part_data const data, int * __restrict__ d_idx )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const np = tiles.np[ tid ];
    unsigned int const offset =  tiles.offset[ tid ];
    int2   * __restrict__ ix  = &data.ix[ offset ];
    int * __restrict__ idx = &d_idx[ offset ];

    __shared__ int _nout;
    _nout = 0;
    block.sync();

    for( int i = block.thread_rank(); i < np; i+= block.num_threads() ) {
        int out = (ix[i].x < 0) || (ix[i].x >= lim.x) || 
                  (ix[i].y < 0) || (ix[i].y >= lim.y);
        
        if (out) {
            int k = atomicAdd( &_nout, 1 );
            idx[k] = i;
        }
    }
    block.sync();

    if ( block.thread_rank() == 0 ) tiles.nidx[ tid ] = _nout;
}

template < coord::cart dir >
__global__
void _bnd_out( int2 const lim, 
    t_part_tiles const tiles, t_part_data const data, int * __restrict__ d_idx,
    t_part_tiles const tmp_tiles, t_part_data tmp_data )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const np = tiles.np[ tid ];
    unsigned int const offset =  tiles.offset[ tid ];
    int2   * __restrict__ ix  = &data.ix[ offset ];
    float2 * __restrict__ x   = &data.x[ offset ];
    float3 * __restrict__ u   = &data.u[ offset ];

    int * __restrict__ idx = &d_idx[ offset ];
    unsigned int const nidx = tiles.nidx[ tid ];

    unsigned int const tmp_offset =  tmp_tiles.offset[ tid ];
    int2   * __restrict__ tmp_ix = &tmp_data.ix[ tmp_offset ];
    float2 * __restrict__ tmp_x  = &tmp_data.x[ tmp_offset ];
    float3 * __restrict__ tmp_u  = &tmp_data.u[ tmp_offset ];


    // Number of particles leaving to the lower neighbour
    __shared__ int _nl;

    // Number of particles leaving to the upper neighbour
    __shared__ int _nu;

    _nl = _nu = 0;

    block.sync();

    int nl = 0;
    int nu = 0;

    for( int i = block.thread_rank(); i < nidx; i+= block.num_threads() ) {
        
        int k = idx[i];
        
        if ( k < np ) {
            if ( dir == coord::x ) {
                nl += ( ix[k].x < 0 );
                nu += ( ix[k].x >= lim.x );
            }
            if ( dir == coord::y ) {
                nl += ( ix[k].y < 0 );
                nu += ( ix[k].y >= lim.y );
            }
        }
    }

    nl = cg::reduce( warp, nl, cg::plus<int>());
    nu = cg::reduce( warp, nu, cg::plus<int>());
    if ( warp.thread_rank() == 0 ) {
        atomicAdd( &_nl, nl );
        atomicAdd( &_nu, nu );
    }

    block.sync();

    // Number of particles staying in node
    int const n0 = np - (_nl + _nu);

    // Index for copying back particles to fill holes
    __shared__ int _c;
    _c = n0;

    // Indices for tmp partilce buffer ( 0 - lower, 1 - upper )
    __shared__ int _l[2];
    _l[0] =   0;
    _l[1] = _nl;

    block.sync();

    for( int i = block.thread_rank(); i < nidx; i+= block.num_threads() ) {
        int k = idx[i];

        if ( k < np ) {
            int bnd;
            if ( dir == coord::x ) 
                bnd = ( ix[k].x < 0 ) + 2 * ( ix[k].x >= lim.x );
            if ( dir == coord::y )
                bnd = ( ix[k].y < 0 ) + 2 * ( ix[k].y >= lim.y );

            if ( bnd > 0 ) {
                // Copy data to temp buffer
                int l = atomicAdd( & _l[bnd-1], 1 );

                tmp_ix[l] = ix[k];
                tmp_x[l]  = x[k];
                tmp_u[l]  = u[k];

                // Fill hole if needed
                if ( k < n0 ) {
                    int c, invalid;

                    do {
                        c = atomicAdd( &_c, 1 );
                        if ( dir == coord::x ) 
                            invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x);
                        if ( dir == coord::y ) 
                            invalid = ( ix[c].y < 0 ) || ( ix[c].y >= lim.y);
                    } while (invalid);

                    ix[ k ] = ix[ c ];
                    x [ k ] = x [ c ];
                    u [ k ] = u [ c ];
                }
            }
        }
    }

    // Store new values on tile information
    if ( block.thread_rank() == 0 ) {
        tiles.np[ tid ]      = n0;
        tmp_tiles.np[ tid ]  = _nl;
        tmp_tiles.np2[ tid ] = _nu;
    }
}

/**
 * @brief CUDA kernel for copying in particles that moved out of neighboring
 * tiles into the local tile.
 * 
 * @tparam dir          Direction to check `coord::x` or `coord::y`
 * @param lim           Tile size along both directions
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
void _bnd_in( int2 const lim,
    t_part_tiles const tiles, t_part_data const data, int * __restrict__ d_idx,
    t_part_tiles const tmp_tiles, t_part_data tmp_data,
    int2 const periodic )
{

    auto grid  = cg::this_grid(); 
    auto block = cg::this_thread_block();

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const unsigned int n0  = tiles.np[ tid ];
    unsigned int np = n0;
    const int offset =  tiles.offset[ tid ];
    int2   __restrict__ *ix = &data.ix[ offset ];
    float2 __restrict__ *x  = &data.x[ offset ];
    float3 __restrict__ *u  = &data.u[ offset ];

    int * __restrict__ idx = &d_idx[ offset ];
    unsigned int nidx = tiles.nidx[ tid ];

    // Number of particles entering the node
    int nin = 0;

    block.sync();

    // Copy from upper neighbour
    int x_ucoord = blockIdx.x;
    int x_lcoord = blockIdx.x;

    if ( dir == coord::x ) {
        x_lcoord -= 1;
        x_ucoord += 1;
        if ( periodic.x ) {
            if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
            if ( x_ucoord >= gridDim.x ) x_ucoord -= gridDim.x;
        }
    }

    int y_ucoord = blockIdx.y;
    int y_lcoord = blockIdx.y;

    if ( dir == coord::y ) {
        y_lcoord -= 1;
        y_ucoord += 1;
        if ( periodic.y ) {
            if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
            if ( y_ucoord >= gridDim.y ) y_ucoord -= gridDim.y;
        }
    }

    if (( x_ucoord < gridDim.x ) && 
        ( y_ucoord < gridDim.y )) {

        int uid = y_ucoord * gridDim.x + x_ucoord;

        unsigned int nu = tmp_tiles.np[ uid ];
        const int upper_offset =  tmp_tiles.offset[ uid ];
        int2   __restrict__ *upper_ix = &tmp_data.ix[ upper_offset ];
        float2 __restrict__ *upper_x  = &tmp_data.x[ upper_offset ];
        float3 __restrict__ *upper_u  = &tmp_data.u[ upper_offset ];

        for( int i = block.thread_rank(); i < nu; i+= block.num_threads() ) {
            int2 t = upper_ix[i];

            if ( dir == coord::x ) t.x += lim.x;
            if ( dir == coord::y ) t.y += lim.y;

            ix[ np + i ] = t;
            x[ np + i ]  = upper_x[i];
            u[ np + i ]  = upper_u[i];

        }
        np += nu;
        nin += nu;
    }

    // Copy from lower neighbour
    if (( x_lcoord >= 0 ) && 
        ( y_lcoord >= 0 )) {

        int lid = y_lcoord * gridDim.x + x_lcoord;;
        
        unsigned int k  = tmp_tiles.np[ lid ];
        unsigned int nl = tmp_tiles.np2[ lid ];
        const int lower_offset =  tmp_tiles.offset[ lid ];
        int2   __restrict__ *lower_ix = &tmp_data.ix[ lower_offset ];
        float2 __restrict__ *lower_x  = &tmp_data.x[ lower_offset ];
        float3 __restrict__ *lower_u  = &tmp_data.u[ lower_offset ];

        for( int i = block.thread_rank(); i < nl; i+= block.num_threads() ) {
            int2 t = lower_ix[k+i];
            
            if ( dir == coord::x ) t.x -= lim.x;
            if ( dir == coord::y ) t.y -= lim.y;

            ix[ np + i ] = t;
            x[ np + i ]  = lower_x[k+i];
            u[ np + i ]  = lower_u[k+i];
        }
        np += nl;
        nin += nl;
    }

    if ( dir < coord::y ) {
        // Add index of incoming particles to list of particles to check
        for( int i = block.thread_rank(); i < nin; i+= block.num_threads() ) {
            idx[ nidx + i ] = n0 + i;
        }
        nidx += nin;
    }

    if ( block.thread_rank() == 0 ) {
        tiles.np[ tid ] = np;
        tiles.nidx[ tid ] = nidx;
    }
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

    int2 lim;
    lim.x = nx.x;
    lim.y = nx.y;


    // Check which particles have left tiles
    _bnd_out_check<<< grid, block >>> ( 
        lim, tiles, data, idx
    );

    _bnd_out< coord::x > <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data
    );

    _bnd_in< coord::x >  <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data,
        periodic
    );

    _bnd_out< coord::y > <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data
     );

    _bnd_in< coord::y >  <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data,
        periodic
    );

   // validate( "After tile_sort_idx" );
}



/**
 * @brief Moves particles to the correct tiles
 * 
 * The idx buffer must be pre-filled with the indices of particles that 
 * have left the tiles
 * 
 * Note that particles are only expected to have moved no more than 1 tile
 * in each direction
 * 
 * @param tmp   Temporary buffer to hold particles moving out of tiles. This
 *              buffer *MUST* be big enough to hold all the particles moving
 *              out of the tiles. It's size is not checked.
 */
__host__
void Particles::tile_sort_idx( Particles &tmp ) {
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    int2 lim;
    lim.x = nx.x;
    lim.y = nx.y;

    _bnd_out< coord::x > <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data
    );

    _bnd_in< coord::x >  <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data,
        periodic
    );

    _bnd_out< coord::y > <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data
     );

    _bnd_in< coord::y >  <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data,
        periodic
    );

   // validate( "After tile_sort_idx" );
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

/**
 * Low memory sorter - mk 1 
 * 
 */


__global__
/**
 * @brief CUDA Kernel to calculate the new number of particles on each tile
 * 
 * The new_tiles.np array must be zeroed before calling this kernel
 * 
 * @param lim           Tile grid size
 * @param tiles         Original tile info
 * @param data          Particle data
 * @param new_tiles     New tile info
 */
void _new_np( int2 const lim, t_part_tiles const tiles, t_part_data const data, t_part_tiles const new_tiles )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const np     = tiles.np[ tid ];
    unsigned int const offset =  tiles.offset[ tid ];
    int2   * __restrict__ ix  = &data.ix[ offset ];

    __shared__ int _npt[9];
    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) 
        _npt[i] = 0;
    block.sync();

    // Count particles 
    int npt[9];
    for( int i = 0; i < 9; i++ ) npt[i] = 0;

    for( int i = block.thread_rank(); i < np; i+= block.num_threads() ) {
        int2 ipos = ix[i];
        int xcross = ( ipos.x >= lim.x ) - ( ipos.x < 0 ) + 1;
        int ycross = ( ipos.y >= lim.y ) - ( ipos.y < 0 ) + 1;
        
        int idx = ycross * 3 + xcross; 
        npt[ idx ] += 1;
    }

    for( int i = 0; i < 9; i++ ) {
        npt[ i ]  = cg::reduce( warp, npt[ i ], cg::plus<int>());
        if ( warp.thread_rank() == 0 ) 
            atomicAdd( &_npt[i], npt[i] );
    }

    block.sync();

    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) {
        
        // Find target node
        int tx = blockIdx.x + i % 3 - 1;
        int ty = blockIdx.y + i / 3 - 1;

        // Correct for periodic boundaries
        if ( tx < 0 ) tx += gridDim.x; 
        if ( tx >= gridDim.x ) tx -= gridDim.x;
        
        if ( ty < 0 ) ty += gridDim.y;
        if ( ty >= gridDim.y ) ty -= gridDim.y;

        atomicAdd( & new_tiles.np[ ty * gridDim.x + tx ], _npt[i] );
    }

    if ( block.thread_rank() == 0 ) {
        for( int i = 0; i < 9; i++ ) tiles.npt[ 9*tid + i ] = _npt[i];
    }
}

__global__
/**
 * @brief CUDA kernel for recalculating particle tile offset
 * 
 * This is just an (exclusive) prefix scan of tiles.np
 * 
 * @param tiles 
 */
void _update_offset( t_part_tiles const tiles, const unsigned int ntiles ) {
    // 32 is the current maximum number of warps
    __shared__ int tmp[ 32 ];
    __shared__ int prev;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Contribution from previous warp
    prev = 0;

    for( unsigned int i = block.thread_rank(); i < ntiles; i += block.num_threads() ) {
        auto s = tiles.np[i];

        auto v = cg::exclusive_scan( warp, s, cg::plus<int>());
        if ( warp.thread_rank() == warp.num_threads() - 1 ) tmp[ warp.meta_group_rank() ] = v + s;
        block.sync();

        // Only 1 warp does this
        if ( warp.meta_group_rank() == 0 ) {
            auto t = tmp[ warp.thread_rank() ];
            t = cg::exclusive_scan( warp, t, cg::plus<int>());
            tmp[ warp.thread_rank() ] = t + prev;
        }
        block.sync();

        // Add in contribution from previous threads
        v += tmp[ warp.meta_group_rank() ];
        tiles.offset[i] = v;

        // Also store offset on np2
        tiles.np2[i] = v;

        if ((block.thread_rank() == block.num_threads() - 1) || ( i + 1 == ntiles ) )
            prev = v + s;

        block.sync();
    }

}

__global__
/**
 * @brief CUDA kernel for generating the sort index
 * 
 * i.e. the target position on the new buffer
 * 
 * @param tiles 
 * @param data 
 * @param new_tiles 
 */
void _copy_sort( int2 const lim, 
    t_part_tiles const tiles, t_part_data const data, 
    t_part_tiles const new_tiles, t_part_data const new_data ) {

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const np     = tiles.np[ tid ];
    unsigned int const offset = tiles.offset[ tid ];
    int* __restrict__ npt     = &tiles.npt[ 9*tid ];

    int2* __restrict__ ix   = &data.ix[ offset ];
    float2* __restrict__ x  = &data.x[ offset ];
    float3* __restrict__ u  = &data.u[ offset ];

    int2* __restrict__  new_ix  = new_data.ix;
    float2* __restrict__ new_x  = new_data.x;
    float3* __restrict__ new_u  = new_data.u;


    __shared__ int _dir_offset[9];

    // Find offsets on new buffer
    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) {
        
        // Find target node
        int tx = blockIdx.x + i % 3 - 1;
        int ty = blockIdx.y + i / 3 - 1;

        // Correct for periodic boundaries
        if ( tx < 0 ) tx += gridDim.x; 
        if ( tx >= gridDim.x ) tx -= gridDim.x;
        
        if ( ty < 0 ) ty += gridDim.y;
        if ( ty >= gridDim.y ) ty -= gridDim.y;

        int tid2 = ty * gridDim.x + tx;

        _dir_offset[i] = atomicAdd( & new_tiles.np2[ tid2 ], npt[ i ] );
    }

    block.sync();

    // Find indices on new buffer
    for( int i = block.thread_rank(); i < np; i+= block.num_threads() ) {
        int2 nix  = ix[i];
        float2 nx = x[i];
        float3 nu = u[i];
        
        int xcross = ( ix[i].x >= lim.x ) - ( ix[i].x < 0 );
        int ycross = ( ix[i].y >= lim.y ) - ( ix[i].y < 0 );
        
        nix.x -= xcross * lim.x;
        nix.y -= ycross * lim.y;

        int k = (ycross + 1) * 3 + (xcross+1); 

        int idx = atomicAdd( & _dir_offset[k], 1 );

        new_ix[ idx ] = nix;
        new_x[ idx ] = nx;
        new_u[ idx ] = nu;
    }

}

__host__
void Particles::tile_sort_mk1( Particles &tmp ) {
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    int2 lim;
    lim.x = nx.x;
    lim.y = nx.y;

    // Get new number of particles per tile
    device::zero( tmp.tiles.np, ntiles.x * ntiles.y );
    _new_np<<< grid, block >>> ( 
        lim, tiles, data, tmp.tiles
    );

    // Get new offsets (prefix scan of np)
    _update_offset<<< 1, 1024 >>> (
        tmp.tiles, ntiles.x * ntiles.y
    );

    _copy_sort <<< grid, block >>> ( 
        lim, tiles, data, tmp.tiles, tmp.data
    );

    // Swap pointers with tmp. class
    swap( data.ix, tmp.data.ix );
    swap( data.x, tmp.data.x );
    swap( data.u, tmp.data.u );

    swap( tiles.np, tmp.tiles.np );
    swap( tiles.offset, tmp.tiles.offset );

    // validate( "After tile_sort_lowmem ");
}

/**
 * Low memory sorter - mk 2 
 * 
 */

__global__
/**
 * @brief CUDA Kernel for checking which particles have left the tile
 * and determine new number of particles per tile.
 * 
 * This kernel expects that new_tiles.np has been zeroed before being
 * called.
 * 
 * Outputs: 
 * 1. new_tiles.np : new number of particles per tile after sort
 * 2. tiles.nidx : total number of particles exiting the tile
 * 3. tiles.npt : number of particles going into each direction
 * 3. d_idx : indices of particles leaving each tile
 * 
 * @param lim           Tile size
 * @param tiles         Tile structure data
 * @param data          Particle data
 * @param new_tiles     (out) New tile structure data
 * @param d_idx         (out) Indexes of particles leaving tile
 */
void _bnd_check( int2 const lim, 
    t_part_tiles const tiles, t_part_data const data, 
    t_part_tiles const new_tiles, int * __restrict__ d_idx )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const np     = tiles.np[ tid ];
    unsigned int const offset =  tiles.offset[ tid ];
    int2   * __restrict__ ix  = &data.ix[ offset ];
    int * __restrict__ idx = &d_idx[ offset ];

    __shared__ int _npt[9];
    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) 
        _npt[i] = 0;
    
    __shared__ int _nout;
    _nout = 0;

    block.sync();

    // Count particles according to their motion
    // Store indices of particles leaving tile

    for( int i = block.thread_rank(); i < np; i+= block.num_threads() ) {
        int2 ipos = ix[i];
        int xcross = ( ipos.x >= lim.x ) - ( ipos.x < 0 );
        int ycross = ( ipos.y >= lim.y ) - ( ipos.y < 0 );
        
        if ( xcross || ycross ) {
            atomicAdd( &_npt[ (ycross+1) * 3 + (xcross+1) ], 1 );
            idx[ atomicAdd( &_nout, 1 ) ] = i;
        }
    }

    block.sync();

    if ( block.thread_rank() == 0 ) {
        _npt[4] = np - _nout;
    }

    block.sync();

    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) {
        
        // Find target node
        int tx = blockIdx.x + i % 3 - 1;
        int ty = blockIdx.y + i / 3 - 1;

        // Correct for periodic boundaries
        if ( tx < 0 ) tx += gridDim.x; 
        if ( tx >= gridDim.x ) tx -= gridDim.x;
        
        if ( ty < 0 ) ty += gridDim.y;
        if ( ty >= gridDim.y ) ty -= gridDim.y;

        atomicAdd( & new_tiles.np[ ty * gridDim.x + tx ], _npt[i] );
    }

    if ( block.thread_rank() == 0 ) {
        for( int i = 0; i < 9; i++ ) tiles.npt[ 9*tid + i ] = _npt[i];
        tiles.nidx[ tid ] = _nout;
    }
}

__global__
void _copy_sort_idx( int2 const lim, 
    t_part_tiles const tiles, t_part_data const data, int * __restrict__ d_idx,
    t_part_tiles const new_tiles, t_part_data const new_data )
{

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const offset =  tiles.offset[ tid ];
    int* __restrict__ npt     = &tiles.npt[ 9*tid ];

    int2   * __restrict__ ix  = &data.ix[ offset ];
    float2 * __restrict__ x   = &data.x[ offset ];
    float3 * __restrict__ u   = &data.u[ offset ];

    int * __restrict__ idx = &d_idx[ offset ];
    unsigned int const nidx = tiles.nidx[ tid ];

    int2* __restrict__  new_ix  = new_data.ix;
    float2* __restrict__ new_x  = new_data.x;
    float3* __restrict__ new_u  = new_data.u;

    __shared__ int _dir_offset[9];
    __shared__ int _c;

    // Find offsets on new buffer
    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) {
        
        // Find target node
        int tx = blockIdx.x + i % 3 - 1;
        int ty = blockIdx.y + i / 3 - 1;

        // Correct for periodic boundaries
        if ( tx < 0 ) tx += gridDim.x; 
        if ( tx >= gridDim.x ) tx -= gridDim.x;
        
        if ( ty < 0 ) ty += gridDim.y;
        if ( ty >= gridDim.y ) ty -= gridDim.y;

        int tid2 = ty * gridDim.x + tx;

        _dir_offset[i] = atomicAdd( & new_tiles.np2[ tid2 ], npt[ i ] );

    }
    
    const int n0 = npt[4];
    _c = n0;

    block.sync();

    for( int i = block.thread_rank(); i < nidx; i+= block.num_threads() ) {
        
        int k = idx[i];

        int2 nix  = ix[k];
        float2 nx = x[k];
        float3 nu = u[k];
        
        int xcross = ( nix.x >= lim.x ) - ( nix.x < 0 );
        int ycross = ( nix.y >= lim.y ) - ( nix.y < 0 );
        
        nix.x -= xcross * lim.x;
        nix.y -= ycross * lim.y;

        int l = atomicAdd( & _dir_offset[(ycross + 1) * 3 + (xcross + 1)], 1 );

        new_ix[ l ] = nix;
        new_x[ l ] = nx;
        new_u[ l ] = nu;

        // Fill hole if needed
        if ( k < n0 ) {
            int c, invalid;

            do {
                c = atomicAdd( &_c, 1 );
                invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x) || 
                          ( ix[c].y < 0 ) || ( ix[c].y >= lim.y);
            } while (invalid);

            ix[ k ] = ix[ c ];
            x [ k ] = x [ c ];
            u [ k ] = u [ c ];
        }
    }

    const int start = _dir_offset[4];

    block.sync();
    for( int i = block.thread_rank(); i < n0; i+= block.num_threads() ) {
        new_ix[ start + i ] = ix[i];
    }

    for( int i = block.thread_rank(); i < n0; i+= block.num_threads() ) {
        new_x[ start + i ] = x[i];
    }

    // Since float3 does not give coallesced access we copy the u data as float
    float * __restrict__ u0 = (float *) u;
    float * __restrict__ u1 = (float *) &new_u[ start ];

    for( int i = block.thread_rank(); i < 3 * n0; i+= block.num_threads() ) {
        u1[i] = u0[i];
    }
}


__host__
void Particles::tile_sort_mk2( Particles &tmp ) {
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    int2 lim;
    lim.x = nx.x;
    lim.y = nx.y;

    // Get new number of particles per tile
    device::zero( tmp.tiles.np, ntiles.x * ntiles.y );
    _bnd_check<<< grid, block >>> ( 
        lim, tiles, data, tmp.tiles, idx
    );

    // Get new offsets (prefix scan of np)
    _update_offset<<< 1, 1024 >>> (
        tmp.tiles, ntiles.x * ntiles.y
    );

    _copy_sort_idx <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data
    );

    // Swap pointers with tmp. class
    swap( data.ix, tmp.data.ix );
    swap( data.x, tmp.data.x );
    swap( data.u, tmp.data.u );

    swap( tiles.np, tmp.tiles.np );
    swap( tiles.offset, tmp.tiles.offset );

    // validate( "After tile_sort_lowmem ");
}

/**
 * Low memory sorter - mk 3
 * 
 */

__global__
/**
 * @brief CUDA Kernel for checking which particles have left the tile
 * and determine new number of particles per tile.
 * 
 * This kernel expects that new_tiles.np has been zeroed before being
 * called.
 * 
 * Outputs: 
 * 1. new_tiles.np : new number of particles per tile after sort
 * 2. tiles.nidx : total number of particles exiting the tile
 * 3. tiles.npt : number of particles going into each direction
 * 3. d_idx : indices of particles leaving each tile
 * 
 * @param lim           Tile size
 * @param tiles         Tile structure data
 * @param data          Particle data
 * @param new_tiles     (out) New tile structure data
 * @param d_idx         (out) Indexes of particles leaving tile
 */
void _mk3_bnd_check( int2 const lim, 
    t_part_tiles const tiles, t_part_data const data, 
    t_part_tiles const new_tiles, int * __restrict__ d_idx )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int const np     = tiles.np[ tid ];
    unsigned int const offset =  tiles.offset[ tid ];

    int  * __restrict__ npt   = &tiles.npt[ 9*tid ];
    int2 * __restrict__ ix    = &data.ix[ offset ];
    int  * __restrict__ idx   = &d_idx[ offset ];

    // Number of particles moving in each direction
    __shared__ int _npt[9];
    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) 
        _npt[i] = 0;
    
    __shared__ int _nout;
    _nout = 0;

    block.sync();

    // Count particles according to their motion
    // Store indices of particles leaving tile

    for( int i = block.thread_rank(); i < np; i+= block.num_threads() ) {
        int2 ipos = ix[i];
        int xcross = ( ipos.x >= lim.x ) - ( ipos.x < 0 );
        int ycross = ( ipos.y >= lim.y ) - ( ipos.y < 0 );
        
        if ( xcross || ycross ) {
            atomicAdd( &_npt[ (ycross+1) * 3 + (xcross+1) ], 1 );
            idx[ atomicAdd( &_nout, 1 ) ] = i;
        }
    }

    block.sync();

    if ( block.thread_rank() == 0 ) {
        _npt[4] = np - _nout;
    }

    block.sync();

    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) {
        
        // Find target node
        int tx = blockIdx.x + i % 3 - 1;
        int ty = blockIdx.y + i / 3 - 1;

        // Correct for periodic boundaries
        if ( tx < 0 ) tx += gridDim.x; 
        if ( tx >= gridDim.x ) tx -= gridDim.x;
        
        if ( ty < 0 ) ty += gridDim.y;
        if ( ty >= gridDim.y ) ty -= gridDim.y;

        int tid2 = ty * gridDim.x + tx;
        atomicAdd( & new_tiles.np[ tid2 ], _npt[i] );
    }

    if ( block.thread_rank() == 0 ) {
        for( int i = 0; i < 9; i++ ) npt[ i ] = _npt[i];
        tiles.nidx[ tid ] = _nout;
    }
}

__global__
/**
 * @brief CUDA kernel for recalculating particle tile offset
 * 
 * Inputs:
 * 1. tiles.np values
 * 
 * Outputs:
 * 1. tiles.offset new values (prefix scan of tiles.np)
 * 2. tiles.np2 is also set to offset
 * 
 * @param tiles     Tile structure data
 * @param ntiles    Total number of tiles
 */
void _mk3_update_offset( t_part_tiles const tiles, const unsigned int ntiles ) {
    // 32 is the current maximum number of warps
    __shared__ int tmp[ 32 ];
    __shared__ int prev;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Contribution from previous warp
    prev = 0;

    for( unsigned int i = block.thread_rank(); i < ntiles; i += block.num_threads() ) {
        auto s = tiles.np[i];

        auto v = cg::exclusive_scan( warp, s, cg::plus<int>());
        if ( warp.thread_rank() == warp.num_threads() - 1 ) tmp[ warp.meta_group_rank() ] = v + s;
        block.sync();

        // Only 1 warp does this
        if ( warp.meta_group_rank() == 0 ) {
            auto t = tmp[ warp.thread_rank() ];
            t = cg::exclusive_scan( warp, t, cg::plus<int>());
            tmp[ warp.thread_rank() ] = t + prev;
        }
        block.sync();

        // Add in contribution from previous threads
        v += tmp[ warp.meta_group_rank() ];
        tiles.offset[i] = v;

        // Also store offset on np2
        tiles.np2[i] = v;

        if ((block.thread_rank() == block.num_threads() - 1) || ( i + 1 == ntiles ) )
            prev = v + s;

        block.sync();
    }

}

__global__
/**
 * @brief CUDA kernel for copying particles to temp. buffer
 * 
 * @param lim           Tile size
 * @param tiles         Tile structure data
 * @param data          Particle data
 * @param d_idx         Indices of particles leaving the node
 * @param tmp_tiles     Temporary tile structure data
 * @param tmp_data      Temporary particle data
 */
void _mk3_copy_out( int2 const lim, 
    t_part_tiles const tiles, t_part_data const data, int * __restrict__ d_idx,
    t_part_tiles const tmp_tiles, t_part_data const tmp_data )
{

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const old_offset      = tiles.offset[ tid ];
    int * __restrict__ npt    = &tiles.npt[ 9*tid ];

    int2   * __restrict__ ix  = &data.ix[ old_offset ];
    float2 * __restrict__ x   = &data.x[ old_offset ];
    float3 * __restrict__ u   = &data.u[ old_offset ];

    int * __restrict__ idx = &d_idx[ old_offset ];
    unsigned int const nidx = tiles.nidx[ tid ];

    int const new_offset = tmp_tiles.offset[ tid ];
    int const new_np     = tmp_tiles.np[ tid ];
    __shared__ int _dir_offset[9];

    // The _dir_offset variables hold the offset for each of the 9 target
    // tiles so the tmp_* variables just point to the beggining of the buffers
    int2* __restrict__  tmp_ix  = tmp_data.ix;
    float2* __restrict__ tmp_x  = tmp_data.x;
    float3* __restrict__ tmp_u  = tmp_data.u;

    // Number of particles staying in tile
    const int n0 = npt[4];

    // Number of particles staying in the tile that need to be copied to temp memory
    // because tile position in memory has shifted
    int nshift;
    if ( new_offset >= old_offset ) {
        nshift = new_offset - old_offset;
    } else {
        nshift = (old_offset + n0) - (new_offset + new_np);
        if ( nshift < 0 ) nshift = 0;
    }
    
    // At most n0 particles will be shifted
    if ( nshift > n0 ) nshift = n0;

    if( block.thread_rank() == 0 ) {
        _dir_offset[4] = atomicAdd( & tmp_tiles.np2[ tid ], nshift );
    }
    block.sync();

    // Find offsets on new buffer
    for( int i = block.thread_rank(); i < 9; i+= block.num_threads() ) {
        
        if ( i != 4 ) {
            // Find target node
            int tx = blockIdx.x + i % 3 - 1;
            int ty = blockIdx.y + i / 3 - 1;

            // Correct for periodic boundaries
            if ( tx < 0 ) tx += gridDim.x; 
            if ( tx >= gridDim.x ) tx -= gridDim.x;
            
            if ( ty < 0 ) ty += gridDim.y;
            if ( ty >= gridDim.y ) ty -= gridDim.y;

            int tid2 = ty * gridDim.x + tx;

            _dir_offset[i] = atomicAdd( & tmp_tiles.np2[ tid2 ], npt[ i ] );
        } 
    }

    __shared__ int _c;
    _c = n0;

    block.sync();

    // Copy particles moving away from tile and fill holes
    for( int i = block.thread_rank(); i < nidx; i+= block.num_threads() ) {
        
        int k = idx[i];

        int2 nix  = ix[k];
        float2 nx = x[k];
        float3 nu = u[k];
        
        int xcross = ( nix.x >= lim.x ) - ( nix.x < 0 );
        int ycross = ( nix.y >= lim.y ) - ( nix.y < 0 );
        
        nix.x -= xcross * lim.x;
        nix.y -= ycross * lim.y;

        // _dir_offset[] includes the offset in the global tmp particle buffer
        int l = atomicAdd( & _dir_offset[(ycross+1) * 3 + (xcross+1)], 1 );

        tmp_ix[ l ] = nix;
        tmp_x[ l ] = nx;
        tmp_u[ l ] = nu;

        // Fill hole if needed
        if ( k < n0 ) {
            int c, invalid;

            do {
                c = atomicAdd( &_c, 1 );
                invalid = ( ix[c].x < 0 ) || ( ix[c].x >= lim.x) || 
                          ( ix[c].y < 0 ) || ( ix[c].y >= lim.y);
            } while (invalid);

            ix[ k ] = ix[ c ];
            x [ k ] = x [ c ];
            u [ k ] = u [ c ];
        }
    }

    block.sync();
    // At this point all particles up to n0 are correct


    // Copy particles that need to be shifted
    const int new_idx = _dir_offset[4];

    if ( new_offset >= old_offset ) {
        // Copy from beggining of buffer
        for( int i = block.thread_rank(); i < nshift; i+= block.num_threads() ) {
            tmp_ix[ new_idx + i ] = ix[ i ];
            tmp_x[ new_idx + i ]  = x [ i ];
            tmp_u[ new_idx + i ]  = u [ i ];
        }

    } else {

        // Copy from end of buffer
        const int old_idx = n0 - nshift;
        for( int i = block.thread_rank(); i < nshift; i+= block.num_threads() ) {
            tmp_ix[ new_idx + i ] = ix[ old_idx + i ];
            tmp_x[ new_idx + i ]  = x [ old_idx + i ];
            tmp_u[ new_idx + i ]  = u [ old_idx + i ];
        }
    }

    // Store current number of local particles
    // These are already in the correct position in global buffer
    if ( block.thread_rank() == 0 ) {
        tiles.np[ tid ] = n0 - nshift;
    }
}


__global__
void _mk3_copy_in( 
    t_part_tiles const tiles, t_part_data const data,
    t_part_tiles const tmp_tiles, t_part_data const tmp_data )
{

    auto block = cg::this_thread_block();

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int old_offset       =  tiles.offset[ tid ];
    const int old_np           =  tiles.np[ tid ];
    const int new_offset       =  tmp_tiles.offset[ tid ];
    const int tmp_np           =  tmp_tiles.np2[ tid ] - new_offset;

    // Notice that we are already working with the new offset
    int2   * __restrict__ ix  = &data.ix[ new_offset ];
    float2 * __restrict__ x   = &data.x [ new_offset ];
    float3 * __restrict__ u   = &data.u [ new_offset ];

    int2   * __restrict__ tmp_ix = &tmp_data.ix[ new_offset ];
    float2 * __restrict__ tmp_x  = &tmp_data.x [ new_offset ];
    float3 * __restrict__ tmp_u  = &tmp_data.u [ new_offset ];

    if ( new_offset >= old_offset ) {

        // Add particles to the end of the buffer
        for( int i = block.thread_rank(); i < tmp_np; i += block.num_threads() ) {
            ix[ old_np + i ] = tmp_ix[ i ];
            x[ old_np + i ]  = tmp_x[ i ];
            u[ old_np + i ]  = tmp_u[ i ];
        }

    } else {

        // Add particles to the beggining of buffer
        int np0 = old_offset - new_offset;
        if ( np0 > tmp_np ) np0 = tmp_np;
        
        for( int i = block.thread_rank(); i < np0; i += block.num_threads() ) {
            ix[ i ] = tmp_ix[ i ];
            x[ i ]  = tmp_x[ i ];
            u[ i ]  = tmp_u[ i ];
        }

        // If any particles left, add particles to the end of the buffer
        for( int i = np0 + block.thread_rank(); i < tmp_np; i += block.num_threads() ) {
            ix[ old_np + i ] = tmp_ix[ i ];
            x[ old_np + i ]  = tmp_x[ i ];
            u[ old_np + i ]  = tmp_u[ i ];
        }

    }

    block.sync();

    // Store the new offset and number of particles
    if ( block.thread_rank() == 0 ) {
        tiles.np[ tid ] =  old_np + tmp_np;
        tiles.offset[ tid ] = new_offset;
    }
}

__host__
void Particles::tile_sort_mk3( Particles &tmp ) {
    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 1024 );

    int2 lim;
    lim.x = nx.x;
    lim.y = nx.y;

    // Get new number of particles per tile
    device::zero( tmp.tiles.np, ntiles.x * ntiles.y );
    _mk3_bnd_check<<< grid, block >>> ( 
        lim, tiles, data, tmp.tiles, idx
    );

    // Get new offsets (prefix scan of np)
    _mk3_update_offset<<< 1, 1024 >>> (
        tmp.tiles, ntiles.x * ntiles.y
    );

    _mk3_copy_out <<< grid, block >>> ( 
        lim, tiles, data, idx,
        tmp.tiles, tmp.data
    );

    _mk3_copy_in <<< grid, block >>> ( 
        tiles, data,
        tmp.tiles, tmp.data
    );

    //validate( "After tile_sort_mk3() ");
}



__global__
/**
 * @brief CUDA Kernel for shifting particle cells
 * 
 * @param tiles     Particle tile information
 * @param d_ix      Particle cells
 * @param shift     Shift amount (x,y)
 */
void _cell_shift( t_part_tiles const tiles, 
    int2 * const __restrict__ d_ix,
    int2 const shift )
{
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const offset = tiles.offset[ tid ];
    int const np     = tiles.np[ tid ];
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

    _cell_shift <<< grid, block >>> ( tiles, data.ix, shift );
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
    t_part_tiles const tiles,  t_part_data const data,
    uint2 const nx, int const over, unsigned int * out ) {

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const offset = tiles.offset[ tid ];
    int const np     = tiles.np[ tid ];
    int2   const * const __restrict__ ix = &data.ix[ offset ];
    float2 const * const __restrict__ x  = &data.x[ offset ];
    float3 const * const __restrict__ u  = &data.u[ offset ];

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
    t_part_tiles const tiles, 
    t_part_data const data,
    uint2 const nx, int const over, unsigned int * out ) {

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    int const offset = tiles.offset[ tid ];
    int const np     = tiles.np[ tid ];
    int2   const * const __restrict__ ix = &data.ix[ offset ];
    float2 const * const __restrict__ x  = &data.x[ offset ];
    float3 const * const __restrict__ u  = &data.u[ offset ];

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

    _dev_tmp_uint.set(0);

    _validate <<< grid, block >>> ( tiles, data, nx, over, _dev_tmp_uint.ptr() );

    unsigned int nerr = _dev_tmp_uint.get();

    if ( nerr > 0 ) {
        std::cerr << "(*error*) " << msg << "\n";
        std::cerr << "(*error*) invalid particle, aborting...\n";
        cudaDeviceReset();
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

    devhost_memcpy( h_tile_np, tiles.np, ntiles.x * ntiles.y );

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
        cudaDeviceReset();
        exit(1);
    }

    free_host( h_tile_np );
}