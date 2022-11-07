#include "density.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ 
/**
 * @brief CUDA kernel for injecting a uniform plasma density
 * 
 * @param range     Cell range (global) to inject particles in
 * @param ppc       Number of particles per cell
 * @param nx        Number of cells in tile
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (position)
 * @param d_u       Particle data (generalized velocity)
 */
void _inject_uniform_kernel( bnd<unsigned int> range,
    uint2 const ppc, uint2 const nx, 
    t_part_tile * const __restrict__ d_tiles,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u )
{

    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    const int np = d_tiles[ tid ].n;
    if ( block.thread_rank() == 0 ) {
        d_tiles[ tid ].nb = np;
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

        const int offset =  d_tiles[ tid ].pos;
        int2   * __restrict__ const ix = &d_ix[ offset ];
        float2 * __restrict__ const x  = &d_x[ offset ];
        float3 * __restrict__ const u  = &d_u[ offset ];

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
            d_tiles[ tid ].n = np + vol * np_cell ;
    }
}

void Density::Uniform::inject( Particles * part, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( part -> ntiles.x, part -> ntiles.y );
    dim3 block( 1024 );

    _inject_uniform_kernel <<< grid, block >>> ( 
            range, ppc, part -> nx, 
            part -> tiles, part -> ix, part -> x, part -> u );
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
void _inject_step_kernel( bnd<unsigned int> range,
    const float step, const uint2 ppc, const uint2 nx,
    t_part_tile * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u )
{

    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int np;
    if ( block.thread_rank() == 0 ) {
        np = d_tiles[ tid ].n;
        d_tiles[ tid ].nb = d_tiles[ tid ].n;
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

        const int offset =  d_tiles[ tid ].pos;
        int2   __restrict__ *ix = &d_ix[ offset ];
        float2 __restrict__ *x  = &d_x[ offset ];
        float3 __restrict__ *u  = &d_u[ offset ];

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
            d_tiles[ tid ].n = np;
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
            part -> tiles, part -> ix, part -> x, part -> u );
        break;
    case( coord::y ):
        _inject_step_kernel <coord::y> <<< grid, block >>> (
            range, step_pos, ppc, part -> nx, 
            part -> tiles, part -> ix, part -> x, part -> u );
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
void _inject_slab_kernel( bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc, uint2 nx,
    t_part_tile * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u )
{
    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int np;
    if ( block.thread_rank() == 0 ) {
        np = d_tiles[ tid ].n;
        d_tiles[ tid ].nb = d_tiles[ tid ].n;
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

        const int offset =  d_tiles[ tid ].pos;
        int2   __restrict__ *ix = &d_ix[ offset ];
        float2 __restrict__ *x  = &d_x[ offset ];
        float3 __restrict__ *u  = &d_u[ offset ];

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
                    if ((t > start) && (t<finish )) {
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
            d_tiles[ tid ].n = np;
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
            part -> tiles, part -> ix, part -> x, part -> u  );
        break;
    case( coord::y ):
        _inject_slab_kernel < coord::y > <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, part -> nx, 
            part -> tiles, part -> ix, part -> x, part -> u  );
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
void _inject_sphere_kernel( bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc, uint2 nx,
    t_part_tile * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u )
{

    auto block = cg::this_thread_block();

    // Tile ID
    int const tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Store number of particles before injection
    __shared__ int np;
    if ( block.thread_rank() == 0 ) {
        np = d_tiles[ tid ].n;
        d_tiles[ tid ].nb = d_tiles[ tid ].n;
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

        const int offset =  d_tiles[ tid ].pos;
        int2   __restrict__ *ix = &d_ix[ offset ];
        float2 __restrict__ *x  = &d_x[ offset ];
        float3 __restrict__ *u  = &d_u[ offset ];

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
            d_tiles[ tid ].n = np;
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
        part -> tiles, part -> ix, part -> x, part -> u  );
}
