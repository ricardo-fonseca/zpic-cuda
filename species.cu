/**
 * @file species.cu
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "species.cuh"
#include <iostream>
#include "tile_zdf.cuh"
#include "timer.cuh"
#include "random.cuh"

#include "util.cuh"

#include <cmath>
#include <math.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

/**
 * @brief Returns reciprocal Lorentz gamma factor
 * 
 * $ \frac{1}{\sqrt{u_x^2 + u_y^2 + u_z^2 + 1 }} $
 * 
 * @param u         Generalized momentum in units of c
 * @return float    Reciprocal Lorentz gamma factor
 */
inline __device__
float rgamma( const float3 u ) {
    // Standard implementation
    // return 1.0f/sqrtf( u.z*u.z + u.y*u.y + u.x*u.x + 1.0f );

    // Using CUDA rsqrt and fma intrinsics
    return rsqrtf( fmaf( u.z, u.z, fmaf( u.y, u.y, fmaf( u.x, u.x, 1.0f ) ) ) );
}

/**
 * @brief Construct a new Species:: Species object
 * 
 * @param name 
 * @param m_q 
 * @param ppc 
 * @param n0
 * @param ufl 
 * @param uth 
 * @param gnx 
 * @param tnx 
 * @param dt 
 */
Species::Species( std::string const name, float const m_q, uint2 const ppc,
        float const n0, float3 const uth, float3 const ufl,
        float2 const box, uint2 const ntiles, uint2 const nx, const float dt ) :
        name{name}, m_q{m_q}, ppc{ppc}, box{box}, dt{dt}
{
    std::cout << "(*info*) Initializing species " << name << " ..." << std::endl;

    q = copysign( fabs(n0), m_q ) / (ppc.x * ppc.y);
    dx.x = box.x / (nx.x * ntiles.x);
    dx.y = box.y / (nx.y * ntiles.y);

    // Maximum number of particles per tile
    unsigned int np_max = nx.x * nx.y * ppc.x * ppc.y * 2;
    
    particles = new Particles( ntiles, nx, np_max );
    tmp = new Particles( ntiles, nx, np_max );

    // Inject particles
    inject_particles();

    // Sets momentum of all particles
    set_u( ufl, uth );

    // Reset iteration numbers
    iter = 0;
}

/**
 * @brief Destroy the Species:: Species object
 * 
 */
Species::~Species() {

    delete( particles );
    delete( tmp );

}

__global__
/**
 * @brief Adds fluid momentum to particles
 * 
 * @param d_tile    Tile information
 * @param d_u       Particle buffer (momenta)
 * @param ufl       Fluid momentum to add
 */
void _set_u_kernel( 
    t_part_tile const * const __restrict__ d_tiles,
    float3 * const __restrict__ d_u, 
    const uint2 seed, const float3 uth, const float3 ufl ) {

    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    rand_init( seed, state, norm );

    // Set particle momenta
    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;
    float3 __restrict__ * const u  = &d_u[ offset ];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        u[i] = make_float3(
            ufl.x + uth.x * rand_norm( state, norm ),
            ufl.y + uth.y * rand_norm( state, norm ),
            ufl.z + uth.z * rand_norm( state, norm )
        );
    }
}

/**
 * @brief Sets momentum of all particles in object using uth / ufl
 * 
 */
void Species::set_u( float3 const uth, float3 const ufl ) {

    Timer t;

    // Set thermal momentum
    dim3 grid( particles->ntiles.x, particles->ntiles.y );
    dim3 block( 64 );
    
    t.start();

    uint2 seed = {12345, 67890};
    _set_u_kernel <<< grid, block >>> ( 
        particles -> tiles, particles -> u, seed, uth, ufl
    );

    t.stop();
    t.report("(*info*) set_u()");
}


/**
 * @brief CUDA kernel for injecting uniform profile
 * 
 * This version does not require atomics
 * 
 * @param nx
 * @param ppc 
 * @param d_tile 
 * @param d_ix 
 * @param d_x 
 * @param d_u
 */
__global__ 
void _inject_uniform_kernel(
    uint2 const nx, uint2 const ppc, 
    t_part_tile * const __restrict__ d_tiles,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u
) {

    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Pointers to tile particle buffers
    const int offset =  d_tiles[ tid ].n;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];
    
    const int np = d_tiles[ tid ].pos;

    const int np_cell = ppc.x * ppc.y;

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    // Each thread takes 1 cell
    for( int idx = threadIdx.x; idx < nx.x*nx.y; idx += blockDim.x ) {
        int2 const cell = make_int2( 
            idx % nx.x,
            idx / nx.x
        );

        int part_idx = np + idx * np_cell;

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ),
                    dpcy * ( i1 + 0.5 )
                );
                ix[ part_idx ] = cell;
                x[ part_idx ] = pos;
                u[ part_idx ] = {0};
                part_idx++;
            }
        }
    }

    // Update global number of particles in tile
    if ( threadIdx.x == 0 )
        d_tiles[ tid ].pos = np + nx.x * nx.y * np_cell ;

}


/**
 * @brief CUDA kernel for injecting step profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param step      Step position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tiles    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */
__global__
void _inject_step_kernel(
    const float step, const uint2 ppc, const uint2 nx,
    t_part_tile * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u ) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tiles[ tid ].pos;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    __shared__ int np;
    np = d_tiles[ tid ].n;

    const int shiftx = blockIdx.x * nx.x;

    for( int idx = threadIdx.x; idx < nx.y * nx.x; idx+= blockDim.x) {
        int2 const cell = make_int2(
            idx % nx.x,
            idx / nx.x
        );
        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ),
                    dpcy * ( i1 + 0.5 )
                );
                if ( shiftx + cell.x + pos.x > step ) {
                    const int k = atomicAdd( &np, 1 );
                    ix[ k ] = cell;
                    x[ k ] = pos;
                    u[ k ] = {0};
                }
            }
        }
    }

    if ( threadIdx.x == 0 )
        d_tiles[ tid ].n = np;
}

/**
 * @brief CUDA kernel for injecting slab profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param slab      slab start/end position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tiles    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */
__global__
void _inject_slab_kernel(
    float2 slab, uint2 ppc, uint2 nx,
    t_part_tile * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u
) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tiles[ tid ].pos;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    __shared__ int np;
    np = d_tiles[ tid ].n;

    const int shiftx = blockIdx.x * nx.x;

    for( int idx = threadIdx.x; idx < nx.y * nx.x; idx+= blockDim.x) {
        int2 const cell = make_int2(
            idx % nx.x,
            idx / nx.x
        );
        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ),
                    dpcy * ( i1 + 0.5 )
                );
                if ( shiftx + cell.x + pos.x > slab.x &&
                        shiftx + cell.x + pos.x < slab.y ) {
                    const int k = atomicAdd( &np, 1 );
                    ix[ k ] = cell;
                    x[ k ] = pos;
                    u[ k ] = {0};
                }
            }
        }
    }

    if ( threadIdx.x == 0 )
        d_tiles[ tid ].n = np;
}

/**
 * @brief CUDA kernel for injecting sphere profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
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
void _inject_sphere_kernel(
    float2 center, float radius, float2 dx, uint2 ppc, uint2 nx,
    t_part_tile * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u
) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tiles[ tid ].pos;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    __shared__ int np;
    np = d_tiles[ tid ].n;

    const int shiftx = blockIdx.x * nx.x;
    const int shifty = blockIdx.y * nx.y;
    const float r2 = radius*radius;

    for( int idx = threadIdx.x; idx < nx.y * nx.x; idx+= blockDim.x) {
        int2 const cell = make_int2( 
            idx % nx.x,
            idx / nx.x
        );
        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ),
                    dpcy * ( i1 + 0.5 )
                );
                float gx = (shiftx + cell.x + pos.x) * dx.x;
                float gy = (shifty + cell.y + pos.y) * dx.y;
                
                if ( (gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y) < r2 ) {
                    const int k = atomicAdd( &np, 1 );
                    ix[ k ] = cell;
                    x[ k ] = pos;
                    u[ k ] = {0};
                }
            }
        }
    }

    if ( threadIdx.x == 0 )
        d_tiles[ tid ].n = np;
}

/**
 * @brief Inject particles in the simulation box
 * 
 * Currently only injecting particles in the whole simulation box is supported.
 * 
 */
void Species::inject_particles( ) {

    // Create particles
/*
    // Uniform density
    
    dim3 grid( data -> ntiles.x, data -> ntiles.y );
    dim3 block( 64 );
    _inject_uniform_kernel <<< grid, block >>> ( 
        particles -> nx, ppc,
        particles -> tiles, particles -> ix, particles -> x, particles -> u 
    );
*/


/*
    // Step density
    float step = 12.8 / dx.x;

    dim3 grid( data -> ntiles.x, data -> ntiles.y );
    dim3 block( 32 );
    _inject_step_kernel <<< grid, block >>> (
        step, ppc, particles -> nx, 
        particles -> tiles, particles -> ix, particles -> x, particles -> u 
    );
*/

/*
    // Slab density
    float2 slab = { 10.f / dx.x, 20.f / dx.y };

    dim3 grid( data -> ntiles.x, data -> ntiles.y );
    dim3 block( 32 );
    _inject_slab_kernel <<< grid, block >>> (
        slab, ppc, particles -> nx, 
        particles -> tiles,
        particles -> ix, particles -> x, particles -> u 
    );
*/

    // Sphere density

    float2 center = { 12.8f, 6.4f};
    float radius = 3.2f;

    dim3 grid( particles-> ntiles.x, particles -> ntiles.y );
    dim3 block( 32 );

    _inject_sphere_kernel <<< grid, block >>> (
        center, radius, dx, ppc, particles -> nx, 
        particles -> tiles, particles -> ix, particles -> x, particles -> u 
    );

}

/**
 * @brief 
 * 
 * @param emf 
 * @param current 
 */
void Species::advance( EMF &emf, Current &current ) {

    std::cerr << __func__ << " not implemented yet." << std::endl;

}

__device__
/**
 * @brief Deposit (charge conserving) current for 1 segment inside a cell
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position
 * @param x1        Final particle position
 * @param deltax    Particle motion
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz       Out of plane current
 * @param J         current(J) grid (should be in shared memory)
 * @param stride    current(J) grid stride
 */
inline void _dep_current_seg(
    const int2 ix, const float2 x0, const float2 x1, const float2 delta,
    const float2 qnx, const float qvz,
    float3 * __restrict__ J, const int stride )
{

    const float S0x0 = 1.0f - x0.x;
    const float S0x1 = x0.x;

    const float S1x0 = 1.0f - x1.x;
    const float S1x1 = x1.x;

    const float S0y0 = 1.0f - x0.y;
    const float S0y1 = x0.y;

    const float S1y0 = 1.0f - x1.y;
    const float S1y1 = x1.y;

    const float wl1 = qnx.x * delta.x;
    const float wl2 = qnx.y * delta.y;
    
    const float wp10 = 0.5f*(S0y0 + S1y0);
    const float wp11 = 0.5f*(S0y1 + S1y1);
    
    const float wp20 = 0.5f*(S0x0 + S1x0);
    const float wp21 = 0.5f*(S0x1 + S1x1);
    
    atomicAdd( &J[ ix.x   + stride* ix.y   ].x, wl1 * wp10 );
    atomicAdd( &J[ ix.x   + stride*(ix.y+1)].x, wl1 * wp11 );

    atomicAdd( &J[ ix.x   + stride* ix.y   ].y, wl2 * wp20 );
    atomicAdd( &J[ ix.x+1 + stride* ix.y   ].y, wl2 * wp21 );

    atomicAdd( &J[ ix.x   + stride* ix.y   ].z, qvz * ( S0x0 * S0y0 + S1x0 * S1y0 + (S0x0 * S1y0 - S1x0 * S0y0)/2.0f ));
    atomicAdd( &J[ ix.x+1 + stride* ix.y   ].z, qvz * ( S0x1 * S0y0 + S1x1 * S1y0 + (S0x1 * S1y0 - S1x1 * S0y0)/2.0f ));

    atomicAdd( &J[ ix.x   + stride*(ix.y+1)].z, qvz * ( S0x0 * S0y1 + S1x0 * S1y1 + (S0x0 * S1y1 - S1x0 * S0y1)/2.0f ));
    atomicAdd( &J[ ix.x+1 + stride*(ix.y+1)].z, qvz * ( S0x1 * S0y1 + S1x1 * S1y1 + (S0x1 * S1y1 - S1x1 * S0y1)/2.0f ));
}

__device__
/**
 * @brief Splits trajectory for y cell crossings
 * 
 * @param ix        Initial cell
 * @param x0        Initial position
 * @param x1        Final position
 * @param dx        Particle motion
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz_2     Out of plane current
 * @param J         Current grid
 * @param stride    Current grid y stride
 */
void _dep_current_split_y( const int2 ix,
    const float2 x0, const float2 x1, const float2 dx,
    const float2 qnx, const float qvz_2,
    float3 * __restrict__ J, const int stride )
{

    const int diy = ( x1.y >= 1.0f ) - ( x1.y < 0.0f );
    if ( diy == 0 ) {
        // No more splits
        _dep_current_seg( ix, x0, x1, dx, qnx, qvz_2, J, stride );
    } else {
        int iyb = ( diy == 1 );
        float delta = (x1.y-iyb)/dx.y;
        float xcross = x0.x + dx.x*(1-delta);

        // First segment
        _dep_current_seg( ix, x0, make_float2( xcross, iyb ), 
            make_float2( dx.x * (1-delta), dx.y * (1-delta) ), 
            qnx, qvz_2 * (1-delta), J, stride );
        // Second segment
        _dep_current_seg( make_int2( ix.x, ix.y + diy), make_float2( xcross, 1.0f - iyb ),
            make_float2( x1.x, x1.y - diy),
            make_float2( dx.x * delta, dx.y * delta ),
            qnx, qvz_2 * delta, J, stride );
    }
}

__global__
/**
 * @brief CUDA kernel for moving particles and depositing current
 * 
 * @param d_tile            Particle tiles information
 * @param d_ix              Particle buffer (cells)
 * @param d_x               Particle buffer (positions)
 * @param d_u               Particle buffer (momenta)
 * @param d_current         Electric current grid
 * @param current_offset    Offset to position 0,0 on tile
 * @param ext_nx            Tile size (external)
 * @param dt_dx             Time step over cell size
 * @param q                 Species charge per particle
 * @param qnx               Normalization values for in plane current deposition
 */
void _move_deposit_kernel(
    t_part_tile const * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u,
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const ext_nx,
    float2 const dt_dx, float const q, float2 const qnx ) 
{
    
    extern __shared__ float3 _move_deposit_buffer[];
    auto block = cg::this_thread_block();

    // Zero current buffer
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        _move_deposit_buffer[i].x = 0;
        _move_deposit_buffer[i].y = 0;
        _move_deposit_buffer[i].z = 0;
    }

    block.sync();

    // Move particles and deposit current
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    float3 * J = _move_deposit_buffer + current_offset;
    const int stride = ext_nx.x;

    const int part_offset = d_tiles[ tid ].pos;
    const int np     = d_tiles[ tid ].n;
    int2   __restrict__ *ix  = &d_ix[ part_offset ];
    float2 __restrict__ *x   = &d_x[ part_offset ];
    float3 __restrict__ *u   = &d_u[ part_offset ];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        float3 pu = u[i];
        float2 x0 = x[i];
        int2 ix0 =ix[i];

        // Get 1 / Lorentz gamma
        float rg = rgamma( pu );

        // Get particle motion
        float2 deltax = make_float2(
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        );

        // Advance position
        float2 x1 = make_float2(
            x0.x + deltax.x,
            x0.y + deltax.y
        );

        // Check for cell crossings
        int2 deltaix = make_int2(
            ((x1.x >= 1.0f) - (x1.x < 0.0f)),
            ((x1.y >= 1.0f) - (x1.y < 0.0f))
        );

        // Deposit current
        {
            float qvz_2 = q * pu.z * rg * 0.5f;
            if ( deltaix.x == 0 ) {
                // No x splits
                _dep_current_split_y( ix0, x0, x1, deltax, qnx, qvz_2, J, stride );
            } else {
                // Split x current into 2 segments
                const int ixb = (deltaix.x == 1);
                const float eps = ( x1.x - ixb ) / deltax.x;
                const float ycross = x0.y + deltax.y * (1 - eps);

                // First segment
                _dep_current_split_y( ix0, x0, make_float2( 1.0f * ixb, ycross ),
                    make_float2( deltax.x * (1-eps), deltax.y * (1-eps)), 
                    qnx, qvz_2 * (1-eps), J, stride );

                // Second segment

                // Correct for y cross on first segment
                int diycross = ( ycross >= 1.0f ) - ( ycross < 0.0f );
                
                _dep_current_split_y( make_int2( ix0.x + deltaix.x, ix0.y + diycross) , 
                    make_float2( 1.0f - ixb, ycross - diycross),
                    make_float2( x1.x - deltaix.x, x1.y - diycross ), 
                    make_float2( deltax.x * eps, deltax.y * eps),
                    qnx, qvz_2 * eps, J, stride );
            }
        }

        // Correct position and store
        x1.x -= deltaix.x;
        x1.y -= deltaix.y;
        x[i] = x1;

        // Modify cell and store
        int2 ix1 = make_int2(
            ix0.x + deltaix.x,
            ix0.y + deltaix.y
        );
        ix[i] = ix1;
    }

    block.sync();

    // Add current to global buffer
    const int tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * ext_nx.x * ext_nx.y;
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_current[tile_off + i].x += _move_deposit_buffer[i].x;
        d_current[tile_off + i].y += _move_deposit_buffer[i].y;
        d_current[tile_off + i].z += _move_deposit_buffer[i].z;
    }

}

__host__
/**
 * @brief Moves particles using current value of u and deposits current
 * 
 * Note that current grid will be zeroed before deposition
 * 
 * @param current   Current grid
 */
void Species::move_deposit( VFLD * current )
{

    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    const float2 qnx = make_float2(
        q * dx.x / dt,
        q * dx.y / dt
    );

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    dim3 block( 1024 );
    uint2 ext_nx = current -> ext_nx();
    size_t shm_size = ext_nx.x * ext_nx.y * sizeof(float3);

    _move_deposit_kernel <<< grid, block, shm_size >>> ( 
        particles -> tiles, 
        particles -> ix, particles -> x, particles -> u,
        current -> d_buffer, current -> offset(), ext_nx, dt_dx, q, qnx
    );
}


__device__
/**
 * @brief Advance memntum using a relativistic Boris pusher.
 * 
 * The momemtum advance in this method is split into 3 parts:
 * 1. Perform half of E-field acceleration
 * 2. Perform full B-field rotation
 * 3. Perform half of E-field acceleration
 * 
 * Note that this implementation (as it is usual in textbooks) uses a
 * linearization of a tangent calculation in the rotation, which may lead
 * to issues for high magnetic fields.
 * 
 * For the future, other, more accurate, rotation algorithms should be used
 * instead, such as employing the full Euler-Rodrigues formula.
 * 
 * Note: uses CUDA intrinsic fmaf() and rsqrtf() functions
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
inline float3 dudt_boris( const float tem, float3 e, float3 b, float3 u)
{

    // First half of acceleration
    e.x *= tem;
    e.y *= tem;
    e.z *= tem;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    // Time centered tem / \gamma
    const float tem_gamma = tem * rsqrtf( fmaf( ut.z, ut.z,
                                          fmaf( ut.y, ut.y, 
                                          fmaf( ut.x, ut.x, 1.0f ) ) ) );

    // Rotation
    b.x *= tem_gamma;
    b.y *= tem_gamma;
    b.z *= tem_gamma;

    u.x = fmaf( b.z, ut.y, ut.x );
    u.y = fmaf( b.x, ut.z, ut.y );
    u.z = fmaf( b.y, ut.x, ut.z );

    u.x = fmaf( -b.y, ut.z, u.x );
    u.y = fmaf( -b.z, ut.x, u.y );
    u.z = fmaf( -b.x, ut.y, u.z );

    const float otsq = 2.0f / fmaf( b.z, b.z,
                              fmaf( b.y, b.y, 
                              fmaf( b.x, b.x, 1.0f ) ) );
    
    b.x *= otsq;
    b.y *= otsq;
    b.z *= otsq;

    ut.x = fmaf( b.z, u.y, ut.x );
    ut.y = fmaf( b.x, u.z, ut.y );
    ut.z = fmaf( b.y, u.x, ut.z );

    ut.x = fmaf( -b.y, u.z, ut.x );
    ut.y = fmaf( -b.z, u.x, ut.y );
    ut.z = fmaf( -b.x, u.y, ut.z );

    // Second half of acceleration
    ut.x += e.x;
    ut.y += e.y;
    ut.z += e.z;

    return ut;
}

__device__
/**
 * @brief Interpolate EM field values at particle position using linear 
 * (1st order) interpolation.
 * 
 * The EM fields are assumed to be organized according to the Yee scheme with
 * the charge defined at lower left corner of the cell
 * 
 * @param E         Pointer to position (0,0) of E field grid
 * @param B         Pointer to position (0,0) of B field grid
 * @param stride    E and B grids y stride
 * @param ix        Particle cell index
 * @param x         Particle postion inside cell
 * @param e[out]    E field at particle position
 * @param b[out]    B field at particleposition
 */
void interpolate_fld( float3 * __restrict__ E, float3 * __restrict__ B, const int stride,
    const int2 ix, const float2 x, float3 & e, float3 & b)
{
    const int i = ix.x;
    const int j = ix.y;

    const float w1 = x.x;
    const float w2 = x.y;

    const int ih = i + (w1 <0.5f)? -1 : 0;
    const int jh = j + (w2 <0.5f)? -1 : 0;

    const float w1h = w1 + ((w1 <0.5f)?0.5f:-0.5f);
    const float w2h = w2 + ((w2 <0.5f)?0.5f:-0.5f);

    // Interpolate E field
    e.x = ( E[ih +     j *stride].x * (1.0f - w1h) + E[ih+1 +     j*stride].x * w1h ) * (1.0f -  w2 ) +
          ( E[ih + (j +1)*stride].x * (1.0f - w1h) + E[ih+1 + (j+1)*stride].x * w1h ) * w2;

    e.y = ( E[i  +     jh*stride].y * (1.0f -  w1) + E[i+1  +     jh*stride].y * w1 ) * (1.0f - w2h ) +
          ( E[i  + (jh+1)*stride].y * (1.0f -  w1) + E[i+1  + (jh+1)*stride].y * w1 ) * w2h;

    e.z = ( E[i  +     j *stride].z * (1.0f - w1) + E[i+1 +     j*stride].z * w1 ) * (1.0f - w2 ) +
          ( E[i  + (j +1)*stride].z * (1.0f - w1) + E[i+1 + (j+1)*stride].z * w1 ) * w2;

    // Interpolate B field
    b.x = ( B[i  +     jh*stride].x * (1.0f -  w1) + B[i+1  +     jh*stride].x * w1 ) * (1.0f - w2h ) +
          ( B[i  + (jh+1)*stride].x * (1.0f -  w1) + B[i+1  + (jh+1)*stride].x * w1 ) * w2h;

    b.y = ( B[ih +      j*stride].y * (1.0f - w1h) + B[ih+1 +      j*stride].y * w1h ) * (1.0f - w2 ) +
          ( B[ih + (j +1)*stride].y * (1.0f - w1h) + B[ih+1 +  (j+1)*stride].y * w1h ) * w2;

    b.z = ( B[ih +     jh*stride].z * (1.0f - w1h) + B[ih+1 +     jh*stride].z * w1h ) * (1.0f - w2h ) +
          ( B[ih + (jh+1)*stride].z * (1.0f - w1h) + B[ih+1 + (jh+1)*stride].z * w1h ) * w2h;
}

__global__
/**
 * @brief CUDA kernel for pushing particles
 * 
 * This kernel will interpolate fields and advance particle momentum using a 
 * relativistic Boris pusher
 * 
 * @param d_tile 
 * @param d_ix 
 * @param d_x 
 * @param d_u 
 * @param d_E 
 * @param d_B 
 * @param field_offset 
 * @param ext_nx 
 * @param dt_dx 
 * @param q 
 * @param qnx 
 */
void _push_kernel ( 
    t_part_tile const * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u,
    float3 * __restrict__ d_E, float3 * __restrict__ d_B, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const tem )
{
    auto block = cg::this_thread_block();
    
    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Copy E and B into shared memory
    extern __shared__ float3 buffer[];

    const int field_vol = ext_nx.x * ext_nx.y;   
    const int tile_off = tid * field_vol;

    for( int i = threadIdx.x; i < field_vol; i += blockDim.x ) {
        buffer[i            ] = d_E[tile_off + i];
        buffer[field_vol + i] = d_B[tile_off + i];
    }
    
    float3* E = buffer + field_offset;
    float3* B = E + field_vol; 

    block.sync();

    // Push particles
    const int part_offset = d_tiles[ tid ].pos;
    const int np          = d_tiles[ tid ].n;
    int2   __restrict__ *ix  = &d_ix[ part_offset ];
    float2 __restrict__ *x   = &d_x[ part_offset ];
    float3 __restrict__ *u   = &d_u[ part_offset ];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {

        // Interpolate field
        float3 e, b;
        interpolate_fld( E, B, ext_nx.x,
            ix[i], x[i], e, b );
        
        // Advance momentum
        float3 pu = u[i];
        dudt_boris( tem, e, b, pu );
        u[i] = pu;
    }
}

__host__
/**
 * @brief Moves particles using current value of u and deposits current
 * 
 * Note that current grid will be zeroed before deposition
 * 
 * @param current   Current grid
 */
void Species::push( VFLD &E, VFLD &B )
{
    const float2 dt_dx = {
        dt / dx.x,
        dt / dx.y
    };

    const float2 qnx = {
        q * dx.x / dt,
        q * dx.y / dt
    };

    const float tem = 0.5 * dt / m_q;

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    dim3 block( 64 );
    uint2 ext_nx = E.ext_nx();
    size_t shm_size = 2*ext_nx.x * ext_nx.y * sizeof(float3);

    _push_kernel <<< grid, block, shm_size >>> ( 
        particles -> tiles, 
        particles -> ix, particles -> x, particles -> u,
        E.d_buffer, B.d_buffer, E.offset(), ext_nx, tem
    );

}

/**
 * @brief Move particles to appropriate tiles after push
 * 
 * Using the tmp particle buffer for temporary storage greatly improves performance.
 */
void Species::tile_sort() {
    
    particles -> tile_sort( *tmp );
}


/**
 * @brief Deposit phasespace density
 * 
 * @param rep_type 
 * @param pha_nx 
 * @param pha_range 
 * @param buf 
 */
void Species::deposit_phasespace( const int rep_type, const int2 pha_nx, const float2 pha_range[2],
        float buf[]) {

    std::cerr << __func__ << " not implemented yet." << std::endl;

}

__global__
/**
 * @brief CUDA kernel for depositing charge
 * 
 * @param d_charge  Charge density grid (will be zeroed by this kernel)
 * @param offset    Offset to position (0,0) of grid
 * @param ext_nx    External tile size (i.e. including guard cells)
 * @param d_tile    Particle tiles information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (position)
 * @param q         Species charge per particle
 */
void _dep_charge_kernel(
    float * const __restrict__ d_charge,
    int offset, uint2 ext_nx,
    t_part_tile const * const __restrict__ d_tiles,
    int2 const * const __restrict__ d_ix, float2 const * const __restrict__ d_x, const float q )
{
    auto block = cg::this_thread_block();

    extern __shared__ float _dep_charge_buffer[];

    // Zero shared memory and sync.
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        _dep_charge_buffer[i] = 0;
    }
    float *charge = &_dep_charge_buffer[ offset ];

    block.sync();

    const int tid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int part_off =  d_tiles[ tid ].pos;
    const int np = d_tiles[ tid ].n;
    int2   __restrict__ const * const ix = &d_ix[ part_off ];
    float2 __restrict__ const * const x  = &d_x[ part_off ];
    const int ystride = ext_nx.x;

    for( int i = threadIdx.x; i < np; i += blockDim.x ) {
        const int idx = ix[i].y * ystride + ix[i].x;
        const float w1 = x[i].x;
        const float w2 = x[i].y;

        atomicAdd( &charge[ idx               ], ( 1.0f - w1 ) * ( 1.0f - w2 ) * q );
        atomicAdd( &charge[ idx + 1           ], (        w1 ) * ( 1.0f - w2 ) * q );
        atomicAdd( &charge[ idx     + ystride ], ( 1.0f - w1 ) * (        w2 ) * q );
        atomicAdd( &charge[ idx + 1 + ystride ], (        w1 ) * (        w2 ) * q );
    }

    block.sync();

    // Copy data to global memory
    const int tile_off = tid * ext_nx.x * ext_nx.y;
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_charge[tile_off + i] += _dep_charge_buffer[i];
    } 
}

__host__
/**
 * @brief Deposit charge density
 * 
 * @param charge    Charge density grid
 */
void Species::deposit_charge( Field &charge ) {

    uint2 ext_nx = charge.ext_nx();
    dim3 grid( charge.nxtiles.x, charge.nxtiles.y );
    dim3 block( 64 );

    size_t shm_size = ext_nx.x * ext_nx.y * sizeof(float);

    _dep_charge_kernel <<< grid, block, shm_size >>> (
        charge.d_buffer, charge.offset(), ext_nx,
        particles -> tiles, particles -> ix, particles -> x, q
    );

}

/**
 * @brief Save particle data to file
 * 
 */
void Species::save_particles( ) {

    const char * quants[] = {
        "x","y",
        "ux","uy","uz"
    };

    const char * qlabels[] = {
        "x","y",
        "u_x","u_y","u_z"
    };

    const char * qunits[] = {
        "c/\\omega_p", "c/\\omega_p",
        "c","c","c"
    };

    t_zdf_iteration iter_info = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_p"
    };

    // Omit number of particles, this will be filled in later
    t_zdf_part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .nquants = 5,
        .quants = (char **) quants,
        .qlabels = (char **) qlabels,
        .qunits = (char **) qunits,
    };

    particles -> save( info, iter_info, "PARTICLES" );
}

/**
 * @brief Saves charge density to file
 * 
 */
void Species::save_charge() {

    const uint2 gnx = particles -> g_nx();
    const uint2 nx = particles -> nx;

    // For linear interpolation we only require 1 guard cell at the upper boundary
    const uint2 gc[2] = {
        {0,0},
        {1,1}
    };

    // Deposit charge on device
    Field charge( gnx, nx, gc );

    charge.zero();
    deposit_charge( charge );

    charge.add_from_gc();

    // Prepare file info
    t_zdf_grid_axis axis[2];
    axis[0] = (t_zdf_grid_axis) {
        .name = (char *) "x",
        .min = 0.,
        .max = box.x,
        .label = (char *) "x",
        .units = (char *) "c/\\omega_p"
    };

    axis[1] = (t_zdf_grid_axis) {
        .name = (char *) "y",
        .min = 0.,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_p"
    };

    std::string grid_name = name + "-charge";
    std::string grid_label = name + " \\rho";

    t_zdf_grid_info info = {
        .name = (char *) grid_name.c_str(),
        .label = (char *) grid_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    info.ndims = 2;
    info.count[0] = gnx.x;
    info.count[1] = gnx.y;

    t_zdf_iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_p"
    };

    std::string path = "CHARGE/";
    path += name;
    
    charge.save( info, iter_info, path.c_str() );
}