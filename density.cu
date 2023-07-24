
#include "density.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__
/**
 * @brief Performs an exclusive scan at the block level
 * 
 *  
 * @param block
 * @param s 
 * @return int 
 */
inline int exclusive_scan_1( cg::thread_block & block, int s ) {

    auto warp  = cg::tiled_partition<32>(block);

    // 32 is the current maximum number of warps
    __shared__ int tmp[ 32 ];
    
    int v = cg::exclusive_scan( warp, s, cg::plus<int>());

    if ( warp.thread_rank() == warp.num_threads() - 1 )
        tmp[ warp.meta_group_rank() ] = v + s;
    block.sync();

    // Only 1 warp does this
    if ( warp.meta_group_rank() == 0 ) {
        auto t = tmp[ warp.thread_rank() ];
        t = cg::exclusive_scan( warp, t, cg::plus<int>());
        tmp[ warp.thread_rank() ] = t;
    }
    block.sync();

    // Add in contribution from previous warps
    v += tmp[ warp.meta_group_rank() ];

    return v;
}

__global__ 
/**
 * @brief CUDA kernel for injecting a uniform plasma density
 * 
 * Particles in the same cell are injected contiguously
 * 
 * @param range     Cell range (global) to inject particles in
 * @param ppc       Number of particles per cell
 * @param nx        Number of cells in tile
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (position)
 * @param d_u       Particle data (generalized velocity)
 */
void _inject_uniform_kernel_mk1( bnd<unsigned int> range,
    uint2 const ppc, uint2 const nx, 
    t_part_tiles const tiles,
    t_part_data const data )
{

    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    const int np = tiles.np[ tid ];
    if ( block.thread_rank() == 0 ) {
        tiles.np2[ tid ] = np;
    }
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   * __restrict__ const ix = &data.ix[ offset ];
        float2 * __restrict__ const x  = &data.x[ offset ];
        float3 * __restrict__ const u  = &data.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        // Each thread takes 1 cell
        for( int idx = threadIdx.x; idx < vol; idx += blockDim.x ) {
            int2 const cell = make_int2( 
                idx % row + ri0,
                idx / row + rj0
            );

            int part_idx = np + idx * np_cell;

            for( int i1 = 0; i1 < ppc.y; i1++ ) {
                for( int i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = {0};
                    part_idx++;
                }
            }
        }

        // Update global number of particles in tile
        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = np + vol * np_cell ;
    }
}

__global__
/**
 * @brief CUDA kernel for injecting a uniform plasma density (mk2)
 * 
 * Places contiguous particles in different cells. This minimizes memory collisions
 * when depositing current, especially for very low temperatures.
 * 
 * @param range     Cell range (global) to inject particles in 
 * @param ppc       Number of particles per cell 
 * @param nx        Number of cells in tile 
 * @param tiles     Particle tile information 
 * @param data      Particle data 
 */
void _inject_uniform_kernel( bnd<unsigned int> range,
    uint2 const ppc, uint2 const nx, 
    t_part_tiles const tiles,
    t_part_data const data )
{

    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    const int np = tiles.np[ tid ];
    if ( block.thread_rank() == 0 ) {
        tiles.np2[ tid ] = np;
    }
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work here
    int const nxx = nx.x;
    int const nxy = nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   * __restrict__ const ix = &data.ix[ offset ];
        float2 * __restrict__ const x  = &data.x[ offset ];
        float3 * __restrict__ const u  = &data.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );

                int ppc_idx = i1 * ppc.x + i0;

                // Each thread takes 1 cell
                for( int idx = threadIdx.x; idx < vol; idx += blockDim.x ) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    int part_idx = np + vol * ppc_idx + idx;

                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = {0};
                    part_idx++;

                }
                ppc_idx ++;
            }
        }

        // Update global number of particles in tile
        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = np + vol * np_cell ;
    }
}

void Density::Uniform::inject( Particles * part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

#if 0

    // Use only for benchmarking
    std::cout << "(*info*) Injecting uniform density using algorithm mk I\n";
    _inject_uniform_kernel_mk1 <<< grid, block >>> ( 
            range, ppc, part -> nx, 
            part -> tiles, part -> data );

#endif

    _inject_uniform_kernel <<< grid, block >>> ( 
            range, ppc, part -> nx, 
            part -> tiles, part -> data );
    
    deviceCheck();

}

__global__
/**
 * @brief CUDA kernel for counting how many particles will be injected
 * 
 * @param range     Cell range (global) to inject particles in
 * @param ppc       Number of particles per cell
 * @param nx        Number of cells in tile
 * @param d_tiles   Particle tile information
 * @param np        Number of particles to inject (out)
 */
void _np_inject_uniform_kernel( bnd<unsigned int> range,
    uint2 const ppc, uint2 const nx, 
    t_part_tiles const tiles,
    int * np )
{

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

        // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work here
    int const nxx = nx.x;
    int const nxy = nx.y;

    int _np;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {
        
        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        _np = vol * ppc.x * ppc.y;

    } else {
        _np = 0;
    }

    np[ tid ] = _np;
}

void Density::Uniform::np_inject( Particles * part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    
    // Run serial inside block
    dim3 block( 1 );

    _np_inject_uniform_kernel <<< grid, block >>> (
        range, ppc, part -> nx, part -> tiles, np
    );
}

/**
 * @brief CUDA kernel for injecting step profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param range     Cell range (global) to inject particles in
 * @param step      Step position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tiles    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */
template < coord::cart dir >
__global__
void _inject_step_kernel_mk0( bnd<unsigned int> range,
    const float step, const uint2 ppc, const uint2 nx,
    t_part_tiles const tiles, t_part_data const data )
{

    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int np;
    if ( block.thread_rank() == 0 ) {
        np = tiles.np[ tid ];
        tiles.np2[ tid ] = np;
    }
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   __restrict__ *ix = &data.ix[ offset ];
        float2 __restrict__ *x  = &data.x[ offset ];
        float3 __restrict__ *u  = &data.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nx.x;
        const int shifty = blockIdx.y * nx.y;

        for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( int i1 = 0; i1 < ppc.y; i1++ ) {
                for( int i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    if ( t > step ) {
                        const int k = atomicAdd( &np, 1 );
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = {0};
                    }
                }
            }
        }

        block.sync();

        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = np;
    }
}

template < coord::cart dir >
__global__
void _inject_step_kernel( bnd<unsigned int> range,
    const float step, const uint2 ppc, const uint2 nx,
    t_part_tiles const tiles, t_part_data const data )
{

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int _np;
    if ( block.thread_rank() == 0 ) {
        _np = tiles.np[ tid ];
        tiles.np2[ tid ] = _np;
    }
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   __restrict__ *ix = &data.ix[ offset ];
        float2 __restrict__ *x  = &data.x[ offset ];
        float3 __restrict__ *u  = &data.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nx.x;
        const int shifty = blockIdx.y * nx.y;

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);

                    int inj = t > step;
                    int off = exclusive_scan( block, inj );

                    if ( inj ) {
                        const int k = _np + off;
                        ix[ k ] = cell;
                        x[ k ]  = pos;
                        u[ k ]  = make_float3(0,0,0);;
                    }

                    inj = cg::reduce( warp, inj, cg::plus<int>());
                    if ( warp.thread_rank() == 0 ) atomicAdd( &_np, inj );
                }
            }
        }

        block.sync();

        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = _np;
    }
}

void Density::Step::inject( Particles * part,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

    float step_pos = (pos - ref.x) / dx.x;

    switch( dir ) {
    case( coord::x ):
        _inject_step_kernel <coord::x> <<< grid, block >>> (
            range, step_pos, ppc, part -> nx, 
            part -> tiles, part -> data );
        break;
    case( coord::y ):
        _inject_step_kernel <coord::y> <<< grid, block >>> (
            range, step_pos, ppc, part -> nx, 
            part -> tiles, part -> data );
        break;
    break;
    }
}

template < coord::cart dir >
__global__
void _np_inject_step_kernel( bnd<unsigned int> range,
    const float step, const uint2 ppc, const uint2 nx,
    t_part_tiles const tiles, int * np )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ int _np;
    _np = 0;
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;
    
    unsigned int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nx.x;
        const int shifty = blockIdx.y * nx.y;

        for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( int i1 = 0; i1 < ppc.y; i1++ ) {
                for( int i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = t > step;
                    inj_np += inj;
                }
            }
        }
    }    

    inj_np = cg::reduce( warp, inj_np, cg::plus<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( &_np, inj_np );

    block.sync();

    if ( block.thread_rank() == 0 )
        np[ tid ] = _np;

}

void Density::Step::np_inject( Particles * part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

    float step_pos = (pos - ref.x) / dx.x;

    switch( dir ) {
    case( coord::x ):
        _np_inject_step_kernel <coord::x> <<< grid, block >>> (
            range, step_pos, ppc, part -> nx, 
            part -> tiles, np );
        break;
    case( coord::y ):
        _np_inject_step_kernel <coord::y> <<< grid, block >>> (
            range, step_pos, ppc, part -> nx, 
            part -> tiles, np );
        break;
    break;
    }
}

/**
 * @brief CUDA kernel for injecting slab profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param range     Cell range (global) to inject particles in
 * @param slab      slab start/end position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tiles   Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */

template < coord::cart dir >
__global__
void _inject_slab_kernel_mk0( bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc, uint2 nx,
    t_part_tiles const tiles, t_part_data const data )
{
    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int np;
    if ( block.thread_rank() == 0 ) {
        np = tiles.np[ tid ];
        tiles.np2[ tid ] = np;
    }
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   __restrict__ *ix = &data.ix[ offset ];
        float2 __restrict__ *x  = &data.x[ offset ];
        float3 __restrict__ *u  = &data.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nx.x;
        const int shifty = blockIdx.y * nx.y;

        for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( int i1 = 0; i1 < ppc.y; i1++ ) {
                for( int i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    if ((t >= start) && (t<finish )) {
                        const int k = atomicAdd( &np, 1 );
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = {0};
                    }
                }
            }
        }

        block.sync();

        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = np;
    }
}


template < coord::cart dir >
__global__
void _inject_slab_kernel( bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc, uint2 nx,
    t_part_tiles const tiles, t_part_data const data )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int _np;
    if ( block.thread_rank() == 0 ) {
        _np = tiles.np[ tid ];
        tiles.np2[ tid ] = _np;
    }
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   __restrict__ *ix = &data.ix[ offset ];
        float2 __restrict__ *x  = &data.x[ offset ];
        float3 __restrict__ *u  = &data.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nx.x;
        const int shifty = blockIdx.y * nx.y;

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );


                for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );
                    int off = exclusive_scan( block, inj );
                    
                    if (inj) {
                        const int k = _np + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = {0};
                    }

                    inj = cg::reduce( warp, inj, cg::plus<int>());
                    if ( warp.thread_rank() == 0 ) atomicAdd( &_np, inj );

                    // Not needed because exclusive_scan() causes sync.
                    // block.sync();

                }
            }
        }

        block.sync();

        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = _np;
    }
}

void Density::Slab::inject( Particles * part,
    uint2 const ppc,float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

    float slab_begin = (begin - ref.x)/ dx.x;
    float slab_end = (end - ref.x)/ dx.x;

    switch( dir ) {
    case( coord::x ):
        _inject_slab_kernel < coord::x > <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, part -> nx, 
            part -> tiles, part -> data );
        break;
    case( coord::y ):
        _inject_slab_kernel < coord::y > <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, part -> nx, 
            part -> tiles, part -> data );
        break;
    }

}

template < coord::cart dir >
__global__
void _np_inject_slab_kernel( bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc, uint2 nx,
    t_part_tiles const tiles, int * np )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ int _np;
    _np = 0;
    block.sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nx.x;
    int ri1 = range.x.upper - blockIdx.x * nx.x;

    int rj0 = range.y.lower - blockIdx.y * nx.y;
    int rj1 = range.y.upper - blockIdx.y * nx.y;

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    unsigned int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nx.x;
        const int shifty = blockIdx.y * nx.y;

        for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( int i1 = 0; i1 < ppc.y; i1++ ) {
                for( int i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = cg::reduce( warp, inj_np, cg::plus<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( &_np, inj_np );

    block.sync();

    if ( block.thread_rank() == 0 )
        np[ tid ] = _np;

}

void Density::Slab::np_inject( Particles * part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

    float slab_begin = (begin - ref.x)/ dx.x;
    float slab_end = (end - ref.x)/ dx.x;

    switch( dir ) {
    case( coord::x ):
        _np_inject_slab_kernel < coord::x > <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, part -> nx, 
            part -> tiles, np );
        break;
    case( coord::y ):
        _np_inject_slab_kernel < coord::y > <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, part -> nx, 
            part -> tiles, np );
        break;
    }
}


/**
 * @brief CUDA kernel for injecting sphere profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param range     Cell range (global) to inject particles in
 * @param center    sphere center in simulation units
 * @param radius    sphere radius in simulation units
 * @param dx        cell size in simulation units
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tiles   Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */
__global__
void _inject_sphere_kernel_mk0( bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc, uint2 nx,
    t_part_tiles const tiles, t_part_data const data )
{

    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int np;
    if ( block.thread_rank() == 0 ) {
        np = tiles.np[ tid ];
        tiles.np2[ tid ] = np;
    }
    block.sync();

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nxx;
    int ri1 = range.x.upper - blockIdx.x * nxx;

    int rj0 = range.y.lower - blockIdx.y * nxy;
    int rj1 = range.y.upper - blockIdx.y * nxy;

    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   __restrict__ *ix = &data.ix[ offset ];
        float2 __restrict__ *x  = &data.x[ offset ];
        float3 __restrict__ *u  = &data.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nxx;
        const int shifty = blockIdx.y * nxy;
        const float r2 = radius*radius;

        for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
            int2 const cell = make_int2( 
                idx % row + ri0,
                idx / row + rj0
            );
            for( int i1 = 0; i1 < ppc.y; i1++ ) {
                for( int i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    if ( (gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y) < r2 ) {
                        const int k = atomicAdd( &np, 1 );
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }
                }
            }
        }

        block.sync();
        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = np;
    }
}

__global__
void _inject_sphere_kernel( bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc, uint2 nx,
    t_part_tiles const tiles, t_part_data const data )
{

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int _np;
    if ( block.thread_rank() == 0 ) {
        _np = tiles.np[ tid ];
        tiles.np2[ tid ] = _np;
    }
    block.sync();

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nxx;
    int ri1 = range.x.upper - blockIdx.x * nxx;

    int rj0 = range.y.lower - blockIdx.y * nxy;
    int rj1 = range.y.upper - blockIdx.y * nxy;

    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  tiles.offset[ tid ];
        int2   __restrict__ *ix = &data.ix[ offset ];
        float2 __restrict__ *x  = &data.x[ offset ];
        float3 __restrict__ *u  = &data.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nxx;
        const int shifty = blockIdx.y * nxy;
        const float r2 = radius*radius;

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = (gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y) < r2;
                    int off = exclusive_scan( block, inj );

                    if ( inj ) {
                        const int k = _np + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3(0,0,0);
                    }
                    inj = cg::reduce( warp, inj, cg::plus<int>());
                    if ( warp.thread_rank() == 0 ) atomicAdd( &_np, inj );
                }
            }
        }

        block.sync();
        if ( block.thread_rank() == 0 )
            tiles.np[ tid ] = _np;
    }
}

void Density::Sphere::inject( Particles * part,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    _inject_sphere_kernel <<< grid, block >>> (
        range, sphere_center, radius, dx, ppc, part -> nx, 
        part -> tiles, part -> data );
}

__global__
void _np_inject_sphere_kernel( bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc, uint2 nx,
    t_part_tiles const tiles, int * np )
{

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ int _np;
    _np = 0;
    block.sync();

    // Comparing signed and unsigned integers does not work
    int const nxx = nx.x;
    int const nxy = nx.y;

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - blockIdx.x * nxx;
    int ri1 = range.x.upper - blockIdx.x * nxx;

    int rj0 = range.y.lower - blockIdx.y * nxy;
    int rj1 = range.y.upper - blockIdx.y * nxy;

    unsigned int inj_np = 0;
    
    // If range overlaps with tile
    if (( ri0 < nxx ) && ( ri1 >= 0 ) &&
        ( rj0 < nxy ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nxx ) ri1 = nxx-1;
        if (rj1 >= nxy ) rj1 = nxy-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = blockIdx.x * nxx;
        const int shifty = blockIdx.y * nxy;
        const float r2 = radius*radius;

        for( int idx = threadIdx.x; idx < vol; idx+= blockDim.x) {
            int2 const cell = make_int2( 
                idx % row + ri0,
                idx / row + rj0
            );
            for( int i1 = 0; i1 < ppc.y; i1++ ) {
                for( int i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = (gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y) < r2;
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = cg::reduce( warp, inj_np, cg::plus<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( &_np, inj_np );

    block.sync();

    if ( block.thread_rank() == 0 )
        np[ tid ] = _np;

}

void Density::Sphere::np_inject( Particles * part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    _np_inject_sphere_kernel <<< grid, block >>> (
        range, sphere_center, radius, dx, ppc, part -> nx, 
        part -> tiles, np );
}
