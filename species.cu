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

#include "timer.cuh"


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
    // return 1.0f/sqrtf( u.z*u.z + ( u.y*u.y + ( u.x*u.x + 1.0f ) ) );

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
Species::Species( std::string const name, float const m_q, 
        uint2 const ppc, Density::Profile const & density,
        float2 const box, uint2 const ntiles, uint2 const nx,
        const float dt ) :
        name(name), m_q(m_q), ppc(ppc), density(density.clone()), box(box), 
        dt(dt)
{

    // Set charge normalization factor
    q = copysign( density.get_n0() , m_q ) / (ppc.x * ppc.y);
    
    // Set cell size
    dx.x = box.x / (nx.x * ntiles.x);
    dx.y = box.y / (nx.y * ntiles.y);

    // Maximum number of particles per tile
    unsigned int np_max = nx.x * nx.y * ppc.x * ppc.y * 8;
    
    particles = new Particles( ntiles, nx, np_max );
    tmp = new Particles( ntiles, nx, np_max );

    // Reset iteration numbers
    iter = 0;

    // Default pusher type
    push_type = pusher::euler;
}

/**
 * @brief Destroy the Species:: Species object
 * 
 */
Species::~Species() {
    delete( tmp );
    delete( particles );
    delete( density );
}

/**
 * @brief Inject particles in the complete simulation box
 * 
 */
void Species::inject( ) {


    uint2 g_nx = particles -> g_nx();

    bnd<unsigned int> range;
    range.x = { .lower = 0, .upper = g_nx.x - 1 };
    range.y = { .lower = 0, .upper = g_nx.y - 1 };

    float2 ref = make_float2( moving_window.motion(), 0 );

    density -> inject( particles, ppc, dx, ref, range );
}

/**
 * @brief Inject particles in a specific cell range
 * 
 */
void Species::inject( bnd<unsigned int> range ) {

    float2 ref = make_float2( moving_window.motion(), 0 );

    density -> inject( particles, ppc, dx, ref, range );
}

/**
 * @brief Move simulation window - shift particles
 * 
 * When using a moving simulation window checks if a window move is due
 * at the current iteration and if so shifts left particle cells
 * 
 */
void Species::move_window_shift() {

    if ( moving_window.needs_move( iter * dt ) ) {
        // Shift particles left 1 cell
        particles -> cell_shift( make_int2(-1,0) );

        moving_window.advance();
        moving_window.needs_inject = true;
    }
}

void Species::move_window_inject() {

    if ( moving_window.needs_inject ) {
        uint2 g_nx = particles -> g_nx();

        bnd<unsigned int> range;
        range.x = { .lower = g_nx.x - 1, .upper = g_nx.x - 1 };
        range.y = { .lower = 0, .upper = g_nx.y - 1 };

        inject( range );

        moving_window.needs_inject = false;
    }
}

/**
 * @brief 
 * 
 * @param emf 
 * @param current 
 */
void Species::advance( EMF const &emf, Current &current ) {

    // Advance momenta
    push( emf.E, emf.B );

    // Advance positions and deposit current
    move( current.J );

    // Increase internal iteration number
    iter++;

    // Moving window - shift particles
    move_window_shift();

    // Update tiles
    particles -> tile_sort( *tmp );

    // Moving window - inject new particles
    move_window_inject();
}

__device__
/**
 * @brief Deposit (charge conserving) current for 1 segment inside a cell
 * 
 * @param ix        Particle cell
 * @param x0        Initial particle position
 * @param x1        Final particle position
 * @param qnx       Normalization values for in plane current deposition
 * @param qvz       Out of plane current
 * @param J         current(J) grid (should be in shared memory)
 * @param stride    current(J) grid stride
 */
inline void _dep_current_seg(
    const int2 ix, const float2 x0, const float2 x1,
    const float2 qnx, const float qvz,
    float3 * __restrict__ J, const int stride )
{
    const float S0x0 = 0.5f - x0.x;
    const float S0x1 = 0.5f + x0.x;

    const float S1x0 = 0.5f - x1.x;
    const float S1x1 = 0.5f + x1.x;

    const float S0y0 = 0.5f - x0.y;
    const float S0y1 = 0.5f + x0.y;

    const float S1y0 = 0.5f - x1.y;
    const float S1y1 = 0.5f + x1.y;

    const float wl1 = qnx.x * (x1.x - x0.x);
    const float wl2 = qnx.y * (x1.y - x0.y);
    
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
        float2 const x0 = x[i];
        int2   const ix0 =ix[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Get particle motion
        float2 const delta = make_float2(
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        );

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings
        int2 const deltai = make_int2(
            ((x1.x >= 0.5f) - (x1.x < -0.5f)),
            ((x1.y >= 0.5f) - (x1.y < -0.5f))
        );

        // Split trajectories:
        int nvp = 1;
        int2 v0_ix; float2 v0_x0, v0_x1; float v0_qvz;
        int2 v1_ix; float2 v1_x0, v1_x1; float v1_qvz;
        int2 v2_ix; float2 v2_x0, v2_x1; float v2_qvz;

        float eps, xint, yint;
        float qvz = q * pu.z * rg * 0.5f;

        // Initial position is the same on all cases
        v0_ix = ix0; v0_x0 = x0;

        switch( 2*(deltai.x != 0) + (deltai.y != 0) )
        {
        case(0): // no splits
            v0_x1 = x1; v0_qvz = qvz;
            break;

        case(1): // only y crossing
            nvp++;

            yint = 0.5f * deltai.y;
            eps  = ( yint - x0.y ) / delta.y;
            xint = x0.x + delta.x * eps;

            v0_x1  = make_float2(xint,yint);
            v0_qvz = qvz * eps;

            v1_ix = make_int2( ix0.x, ix0.y  + deltai.y );
            v1_x0 = make_float2(xint,-yint);
            v1_x1 = make_float2( x1.x, x1.y  - deltai.y );
            v1_qvz = qvz * (1-eps);

            break;

        case(2): // only x crossing
        case(3): // x-y crossing
            
            // handle x cross
            nvp++;
            xint = 0.5f * deltai.x;
            eps  = ( xint - x0.x ) / delta.x;
            yint = x0.y + delta.y * eps;

            v0_x1 = make_float2(xint,yint);
            v0_qvz = qvz * eps;

            v1_ix = make_int2( ix0.x + deltai.x, ix0.y);
            v1_x0 = make_float2(-xint,yint);
            v1_x1 = make_float2( x1.x - deltai.x, x1.y );
            v1_qvz = qvz * (1-eps);

            // handle additional y-cross, if need be
            if ( deltai.y ) {
                float yint2 = 0.5f * deltai.y;
                nvp++;

                if ( yint >= -0.5f && yint < 0.5f ) {
                    // y crosssing on 2nd vp
                    eps   = (yint2 - yint) / (x1.y - yint );
                    float xint2 = -xint + (x1.x - xint ) * eps;
                    
                    v2_ix = make_int2( v1_ix.x, v1_ix.y + deltai.y );
                    v2_x0 = make_float2(xint2,-yint2);
                    v2_x1 = make_float2( v1_x1.x, v1_x1.y - deltai.y );
                    v2_qvz = v1_qvz * (1-eps);

                    // Correct other particle
                    v1_x1 = make_float2(xint2,yint2);
                    v1_qvz *= eps;
                } else {
                    // y crossing on 1st vp
                    eps   = (yint2 - x0.y) / ( yint - x0.y );
                    float xint2 = x0.x + ( xint - x0.x ) * eps;

                    v2_ix = make_int2( v0_ix.x, v0_ix.y + deltai.y );
                    v2_x0 = make_float2( xint2,-yint2);
                    v2_x1 = make_float2( v0_x1.x, v0_x1.y - deltai.y );
                    v2_qvz = v0_qvz * (1-eps);

                    // Correct other particles
                    v0_x1 = make_float2(xint2,yint2);
                    v0_qvz *= eps;

                    v1_ix.y += deltai.y;
                    v1_x0.y -= deltai.y;
                    v1_x1.y -= deltai.y;
                }
            }
            break;
        }

        // Deposit vp current
                       _dep_current_seg( v0_ix, v0_x0, v0_x1, qnx, v0_qvz, J, stride );
        if ( nvp > 1 ) _dep_current_seg( v1_ix, v1_x0, v1_x1, qnx, v1_qvz, J, stride );
        if ( nvp > 2 ) _dep_current_seg( v2_ix, v2_x0, v2_x1, qnx, v2_qvz, J, stride );

        // Correct position and store
        x1.x -= deltai.x;
        x1.y -= deltai.y;
                
        x[i] = x1;

        // Modify cell and store
        int2 ix1 = make_int2(
            ix0.x + deltai.x,
            ix0.y + deltai.y
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
 * @brief Moves particles and deposit current
 * 
 * Current will be accumulated on existing data
 * 
 * @param current   Current grid
 */
void Species::move( VectorField * J )
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
    uint2 ext_nx = J -> ext_nx();
    size_t shm_size = ext_nx.x * ext_nx.y * sizeof(float3);

    _move_deposit_kernel <<< grid, block, shm_size >>> ( 
        particles -> tiles, 
        particles -> ix, particles -> x, particles -> u,
        J -> d_buffer, J -> offset(), ext_nx, dt_dx, q, qnx
    );
}

__global__
/**
 * @brief CUDA kernel for moving particles
 * 
 * @param d_tile            Particle tiles information
 * @param d_ix              Particle buffer (cells)
 * @param d_x               Particle buffer (positions)
 * @param d_u               Particle buffer (momenta)
 * @param dt_dx             Time step over cell size
 */
void _move_kernel(
    t_part_tile const * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u,
    float2 const dt_dx ) 
{
    
    auto block = cg::this_thread_block();

    // Move particles and deposit current
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

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
        float2 delta = make_float2(
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        );

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings
        int2 deltai = make_int2(
            ((x1.x >= 0.5f) - (x1.x < -0.5f)),
            ((x1.y >= 0.5f) - (x1.y < -0.5f))
        );

        // Correct position and store
        x1.x -= deltai.x;
        x1.y -= deltai.y;
                
        x[i] = x1;

        // Modify cell and store
        int2 ix1 = make_int2(
            ix0.x + deltai.x,
            ix0.y + deltai.y
        );
        ix[i] = ix1;
    }
}

__host__
/**
 * @brief Moves particles (no current deposition)
 * 
 * This is usually used for test species: species that do not self-consistently
 * influence the simulation
 * 
 * @param current   Current grid
 */
void Species::move( )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    dim3 block( 1024 );

    _move_kernel <<< grid, block >>> ( 
        particles -> tiles, particles -> ix, particles -> x, particles -> u,
        dt_dx
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
inline float3 dudt_boris( const float alpha, float3 e, float3 b, float3 u)
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        // Time centered tem / \gamma
        const float alpha_gamma = alpha * rsqrtf( fmaf( ut.z, ut.z,
                                            fmaf( ut.y, ut.y, 
                                            fmaf( ut.x, ut.x, 1.0f ) ) ) );

        // Rotation
        b.x *= alpha_gamma;
        b.y *= alpha_gamma;
        b.z *= alpha_gamma;
    }

    u.x = fmaf( b.z, ut.y, ut.x );
    u.y = fmaf( b.x, ut.z, ut.y );
    u.z = fmaf( b.y, ut.x, ut.z );

    u.x = fmaf( -b.y, ut.z, u.x );
    u.y = fmaf( -b.z, ut.x, u.y );
    u.z = fmaf( -b.x, ut.y, u.z );

    {
        const float otsq = 2.0f / fmaf( b.z, b.z,
                                fmaf( b.y, b.y, 
                                fmaf( b.x, b.x, 1.0f ) ) );
        
        b.x *= otsq;
        b.y *= otsq;
        b.z *= otsq;
    }

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
 * @brief Advance memntum using a relativistic Boris pusher for high magnetic fields
 * 
 * This is similar to the dudt_boris method above, but the rotation is done using
 * using an exact Euler-Rodriguez method.2
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
inline float3 dudt_boris_euler( const float alpha, float3 e, float3 b, float3 u)
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        float const alpha2_gamma = alpha * 2 * 
            rsqrtf( fmaf( ut.z, ut.z, fmaf( ut.y, ut.y, fmaf( ut.x, ut.x, 1.0f ) ) ) );

        b.x *= alpha2_gamma;
        b.y *= alpha2_gamma;
        b.z *= alpha2_gamma;
    }

    {
        float const bnorm = sqrtf(fmaf( b.x, b.x, fmaf( b.y, b.y, b.z * b.z ) ));
        float const s = -(( bnorm > 0 ) ? sinf( bnorm / 2 ) / bnorm : 1 );

        float const ra = cosf( bnorm / 2 );
        float const rb = b.x * s;
        float const rc = b.y * s;
        float const rd = b.z * s;

        float const r11 =   fmaf(ra,ra,rb*rb)-fmaf(rc,rc,rd*rd);
        float const r12 = 2*fmaf(rb,rc,ra*rd);
        float const r13 = 2*fmaf(rb,rd,-ra*rc);

        float const r21 = 2*fmaf(rb,rc,-ra*rd);
        float const r22 =   fmaf(ra,ra,rc*rc)-fmaf(rb,rb,rd*rd);
        float const r23 = 2*fmaf(rc,rd,ra*rb);

        float const r31 = 2*fmaf(rb,rd,ra*rc);
        float const r32 = 2*fmaf(rc,rd,-ra*rb);
        float const r33 =   fmaf(ra,ra,rd*rd)-fmaf(rb,rb,-rc*rc);

        u.x = fmaf( r11, ut.x, fmaf( r21, ut.y , r31 * ut.z ));
        u.y = fmaf( r12, ut.x, fmaf( r22, ut.y , r32 * ut.z ));
        u.z = fmaf( r12, ut.x, fmaf( r23, ut.y , r33 * ut.z ));
    }


    // Second half of acceleration
    u.x += e.x;
    u.y += e.y;
    u.z += e.z;

    return u;
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
void interpolate_fld( 
    volatile float3 const * const __restrict__ E, 
    volatile float3 const * const __restrict__ B, 
    const unsigned int stride,
    const int2 ix, const float2 x, float3 & e, float3 & b)
{
    const int i = ix.x;
    const int j = ix.y;

    const float s0x = 0.5f - x.x;
    const float s1x = 0.5f + x.x;

    const float s0y = 0.5f - x.y;
    const float s1y = 0.5f + x.y;

    const int hx = x.x < 0;
    const int hy = x.y < 0;

    const int ih = i - hx;
    const int jh = j - hy;

    const float s0xh = (1-hx) - x.x;
    const float s1xh = (  hx) + x.x;

    const float s0yh = (1-hy) - x.y;
    const float s1yh = (  hy) + x.y;


    // Interpolate E field
    e.x = ( E[ih +     j *stride].x * s0xh + E[ih+1 +     j*stride].x * s1xh ) * s0y +
          ( E[ih + (j +1)*stride].x * s0xh + E[ih+1 + (j+1)*stride].x * s1xh ) * s1y;

    e.y = ( E[i  +     jh*stride].y * s0x  + E[i+1  +     jh*stride].y * s1x ) * s0yh +
          ( E[i  + (jh+1)*stride].y * s0x  + E[i+1  + (jh+1)*stride].y * s1x ) * s1yh;

    e.z = ( E[i  +     j *stride].z * s0x  + E[i+1  +     j*stride].z * s1x ) * s0y +
          ( E[i  + (j +1)*stride].z * s0x  + E[i+1  + (j+1)*stride].z * s1x ) * s1y;

    // Interpolate B field
    b.x = ( B[i  +     jh*stride].x * s0x + B[i+1  +     jh*stride].x * s1x ) * s0yh +
          ( B[i  + (jh+1)*stride].x * s0x + B[i+1  + (jh+1)*stride].x * s1x ) * s1yh;

    b.y = ( B[ih +      j*stride].y * s0xh + B[ih+1 +      j*stride].y * s1xh ) * s0y +
          ( B[ih + (j +1)*stride].y * s0xh + B[ih+1 +  (j+1)*stride].y * s1xh ) * s1y;

    b.z = ( B[ih +     jh*stride].z * s0xh + B[ih+1 +     jh*stride].z * s1xh ) * s0yh +
          ( B[ih + (jh+1)*stride].z * s0xh + B[ih+1 + (jh+1)*stride].z * s1xh ) * s1yh;
}


/**
 * @brief CUDA kernel for pushing particles
 * 
 * This kernel will interpolate fields and advance particle momentum using a 
 * relativistic Boris pusher
 * 
 * @param d_tiles       Particle tile information
 * @param d_ix          Particle data (cells)
 * @param d_x           Particle data (positions)
 * @param d_u           Particle data (momenta)
 * @param d_E           E field grid
 * @param d_B           B field grid
 * @param field_offset  Tile offset to field position (0,0)
 * @param ext_nx        E,B tile grid external size
 * @param alpha         Force normalization ( 0.5 * q / m * dt )
 */
template < pusher::type type >
__global__
void _push_kernel ( 
    t_part_tile const * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u,
    float3 * __restrict__ d_E, float3 * __restrict__ d_B, 
    unsigned int const field_offset, uint2 const ext_nx,
    float const alpha )
{
    auto block = cg::this_thread_block();
    
    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Copy E and B into shared memory
    extern __shared__ float3 buffer[];

    int const field_vol = ext_nx.x * ext_nx.y;   
    int const tile_off = tid * field_vol;

    for( int i = threadIdx.x; i < field_vol; i += blockDim.x ) {
        buffer[i            ] = d_E[tile_off + i];
        buffer[field_vol + i] = d_B[tile_off + i];
    }
    
    float3 const * const E = buffer + field_offset;
    float3 const * const B = E + field_vol; 

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
        
        if ( type == pusher::boris ) u[i] = dudt_boris( alpha, e, b, pu );
        if ( type == pusher::euler ) u[i] = dudt_boris_euler( alpha, e, b, pu );
    }
}

__host__
/**
 * @brief       Accelerates particles using a Boris pusher
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 */
void Species::push( VectorField * const E, VectorField * const B )
{
    const float2 dt_dx = {
        dt / dx.x,
        dt / dx.y
    };

    const float2 qnx = {
        q * dx.x / dt,
        q * dx.y / dt
    };


    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    dim3 block( 1024 );
    uint2 ext_nx = E -> ext_nx();
    size_t shm_size = 2 * ( ext_nx.x * ext_nx.y * sizeof(float3) );

    const float alpha = 0.5 * dt / m_q;

    switch( push_type ) {
    case( pusher :: euler ):
        _push_kernel <pusher::euler> <<< grid, block, shm_size >>> (
            particles -> tiles, 
            particles -> ix, particles -> x, particles -> u,
            E -> d_buffer, B -> d_buffer, E -> offset(), ext_nx, alpha
        );
        break;
    case( pusher :: boris ):
        _push_kernel <pusher::boris> <<< grid, block, shm_size >>> (
            particles -> tiles, 
            particles -> ix, particles -> x, particles -> u,
            E -> d_buffer, B -> d_buffer, E -> offset(), ext_nx, alpha
        );
        break;
    }

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
        const float s0x = 0.5f - x[i].x;
        const float s1x = 0.5f + x[i].x;
        const float s0y = 0.5f - x[i].y;
        const float s1y = 0.5f + x[i].y;

        atomicAdd( &charge[ idx               ], s0y * s0x * q );
        atomicAdd( &charge[ idx + 1           ], s0y * s1x * q );
        atomicAdd( &charge[ idx     + ystride ], s1y * s0x * q );
        atomicAdd( &charge[ idx + 1 + ystride ], s1y * s1x * q );
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
void Species::deposit_charge( Field &charge ) const {

    uint2 ext_nx = charge.ext_nx();
    dim3 grid( charge.ntiles.x, charge.ntiles.y );
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
void Species::save() const {

    const char * quants[] = {
        "x","y",
        "ux","uy","uz"
    };

    const char * qlabels[] = {
        "x","y",
        "u_x","u_y","u_z"
    };

    const char * qunits[] = {
        "c/\\omega_n", "c/\\omega_n",
        "c","c","c"
    };

    zdf::iteration iter_info = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Omit number of particles, this will be filled in later
    zdf::part_info info = {
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
 * The routine will create a new charge grid, deposit the charge and save the grid
 * 
 */
void Species::save_charge() const {

    // For linear interpolation we only require 1 guard cell at the upper boundary
    const uint2 gc[2] = {
        {0,0},
        {1,1}
    };

    // Deposit charge on device
    Field charge( particles -> ntiles, particles -> nx, gc );

    charge.zero();
    deposit_charge( charge );

    charge.add_from_gc();

    // Prepare file info
    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "x",
        .min = 0. + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "y",
        .min = 0.,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_n"
    };

    std::string grid_name = name + "-charge";
    std::string grid_label = name + " \\rho";

    zdf::grid_info info = {
        .name = (char *) grid_name.c_str(),
        .label = (char *) grid_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    std::string path = "CHARGE/";
    path += name;
    
    charge.save( info, iter_info, path.c_str() );
}

/**
 * @brief CUDA kernel for depositing 1d phasespace
 * 
 * @tparam q        Phasespace quantity
 * @param d_data    Output data
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
 */
template < phasespace::quant q >
__global__ void _dep_pha1_kernel( 
    float * const __restrict__ d_data, float2 const range, unsigned const size,
    uint2 const tile_nx, float const norm, 
    t_part_tile const * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u )
{
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int part_offset = d_tiles[ tid ].pos;
    const int np     = d_tiles[ tid ].n;
    int2   __restrict__ *ix  = &d_ix[ part_offset ];
    float2 __restrict__ *x   = &d_x[ part_offset ];
    float3 __restrict__ *u   = &d_u[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        float d;
        switch( q ) {
        case( phasespace:: x ): d = ( blockIdx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ): d = ( blockIdx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d = u[i].x; break;
        case( phasespace:: uy ): d = u[i].y; break;
        case( phasespace:: uz ): d = u[i].z; break;
        }

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = int( n + 1 ) - 1;
        float w = n - k;

        if ((k   >= 0) && (k   < size-1)) atomicAdd( &d_data[k  ], (1-w) * norm );
        if ((k+1 >= 0) && (k+1 < size-1)) atomicAdd( &d_data[k+1],    w  * norm );
    }
}

__host__
/**
 * @brief Deposit 1D phasespace
 * 
 * Output data will be zeroed before deposition
 * 
 * @param d_data    Output (device) data
 * @param quant     Phasespace quantity
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 */
void Species::dep_phasespace( float * const d_data, phasespace::quant quant, 
    float2 range, unsigned const size ) const
{
    // Zero device memory
    auto err = cudaMemsetAsync( d_data, 0, size * sizeof(float) );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to zero device memory in " <<  __func__ << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    
    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    dim3 block( 1024 );

    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    switch(quant) {
    case( phasespace::x ):
        range.y /= dx.x;
        range.x /= dx.x;
        _dep_pha1_kernel<phasespace::x> <<< grid, block >>> (
            d_data, range, size, particles -> nx, norm, 
            particles -> tiles, particles -> ix, particles -> x, particles -> u
        );
        break;
    case( phasespace:: y ):
        range.y /= dx.y;
        range.x /= dx.y;
        _dep_pha1_kernel<phasespace::y> <<< grid, block >>> (
            d_data, range, size, particles -> nx, norm, 
            particles -> tiles, particles -> ix, particles -> x, particles -> u
        );
        break;
    case( phasespace:: ux ):
        _dep_pha1_kernel<phasespace::ux> <<< grid, block >>> (
            d_data, range, size, particles -> nx, norm, 
            particles -> tiles, particles -> ix, particles -> x, particles -> u
        );
        break;
    case( phasespace:: uy ):
        _dep_pha1_kernel<phasespace::uy> <<< grid, block >>> (
            d_data, range, size, particles -> nx, norm, 
            particles -> tiles, particles -> ix, particles -> x, particles -> u
        );
        break;
    case( phasespace:: uz ):
        _dep_pha1_kernel<phasespace::uz> <<< grid, block >>> (
            d_data, range, size, particles -> nx, norm, 
            particles -> tiles, particles -> ix, particles -> x, particles -> u
        );
        break;
    };
}

__host__
/**
 * @brief Save 1D phasespace
 * 
 * @param q         Phasespace quantity
 * @param range     Phasespace range
 * @param size      Phasespace grid size
 */
void Species::save_phasespace( phasespace::quant quant, float2 const range, 
    unsigned const size ) const
{
    std::string qname, qlabel, qunits;

    phasespace::qinfo( quant, qname, qlabel, qunits );
    
    // Prepare file info
    zdf::grid_axis axis = {
        .name = (char *) qname.c_str(),
        .min = range.x,
        .max = range.y,
        .label = (char *) qlabel.c_str(),
        .units = (char *) qunits.c_str()
    };

    if ( quant == phasespace::x ) {
        axis.min += moving_window.motion();
        axis.max += moving_window.motion();
    }

    std::string pha_name  = name + "-" + qname;
    std::string pha_label = name + "\\,(" + qlabel+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 1,
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = &axis
    };

    info.count[0] = size;

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Deposit 1D phasespace
    float * d_data, * h_data;
    malloc_host( h_data, size  );
    malloc_dev( d_data, size );

    dep_phasespace( d_data, quant, range, size );

    // Copy data to host
    devhost_memcpy( h_data, d_data, size );

    // Save file
    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    free_host( h_data );
    free_dev( d_data );
}

/**
 * @brief CUDA kernel for depositing 2D phasespace
 * 
 * @tparam q0       Quantity 0
 * @tparam q1       Quantity 1
 * @param d_data    Ouput data
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param range1    Range of values of quantity 1
 * @param size1     Range of values of quantity 1
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
 */
template < phasespace::quant quant0, phasespace::quant quant1 >
__global__ void _dep_pha2_kernel( 
    float * const __restrict__ d_data, 
    float2 const range0, unsigned int const size0,
    float2 const range1, unsigned int const size1,
    uint2 const tile_nx, float const norm, 
    t_part_tile const * const __restrict__ d_tiles,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u )
{
    static_assert( quant1 > quant0, "quant1 must be > quant0" );
    
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int part_offset = d_tiles[ tid ].pos;
    const int np     = d_tiles[ tid ].n;
    int2   __restrict__ *ix  = &d_ix[ part_offset ];
    float2 __restrict__ *x   = &d_x[ part_offset ];
    float3 __restrict__ *u   = &d_u[ part_offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        float d0;
        switch( quant0 ) {
        case( phasespace:: x ):  d0 = ( blockIdx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ):  d0 = ( blockIdx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d0 = u[i].x; break;
        case( phasespace:: uy ): d0 = u[i].y; break;
        case( phasespace:: uz ): d0 = u[i].z; break;
        }

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = int( n0 + 1 ) - 1;
        float w0 = n0 - k0;

        float d1;
        switch( quant1 ) {
        //case( phasespace:: x ):  d1 = ( blockIdx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ):  d1 = ( blockIdx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d1 = u[i].x; break;
        case( phasespace:: uy ): d1 = u[i].y; break;
        case( phasespace:: uz ): d1 = u[i].z; break;
        }

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = int( n1 + 1 ) - 1;
        float w1 = n1 - k1;

        if ((k0   >= 0) && (k0   < size0-1) && (k1   >= 0) && (k1   < size1-1))
            atomicAdd( &d_data[(k1  )*size0 + k0  ], (1-w0) * (1-w1) * norm );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1   >= 0) && (k1   < size1-1))
            atomicAdd( &d_data[(k1  )*size0 + k0+1],    w0  * (1-w1) * norm );
        if ((k0   >= 0) && (k0   < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            atomicAdd( &d_data[(k1+1)*size0 + k0  ], (1-w0) *    w1  * norm );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            atomicAdd( &d_data[(k1+1)*size0 + k0+1],    w0  *    w1  * norm );
    }
}

__host__
/**
 * @brief Deposits a 2D phasespace in a device buffer
 * 
 * @param d_data    Pointer to device buffer
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant0    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 1
 */
void Species::dep_phasespace( 
    float * const d_data,
    phasespace::quant quant0, float2 range0, unsigned const size0,
    phasespace::quant quant1, float2 range1, unsigned const size1 ) const
{

    // Zero device memory
    auto err = cudaMemsetAsync( d_data, 0, size0 * size1 * sizeof(float) );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to zero device memory in " <<  __func__ << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    
    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    dim3 block( 1024 );

    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                           ( size0 / (range0.y - range0.x) ) *
                           ( size1 / (range1.y - range1.x) );

    switch(quant0) {
    case( phasespace::x ):
        range0.y /= dx.x;
        range0.x /= dx.x;
        switch(quant1) {
        case( phasespace::y ):
            range1.y /= dx.y;
            range1.x /= dx.y;

            _dep_pha2_kernel<phasespace::x,phasespace::y> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        case( phasespace::ux ):
            _dep_pha2_kernel<phasespace::x,phasespace::ux> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        case( phasespace::uy ):
            _dep_pha2_kernel<phasespace::x,phasespace::uy> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        case( phasespace::uz ):
            _dep_pha2_kernel<phasespace::x,phasespace::uz> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        }
        break;
    case( phasespace:: y ):
        range0.y /= dx.y;
        range0.x /= dx.y;
        switch(quant1) {
        case( phasespace::ux ):
            _dep_pha2_kernel<phasespace::y,phasespace::ux> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        case( phasespace::uy ):
            _dep_pha2_kernel<phasespace::y,phasespace::uy> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        case( phasespace::uz ):
            _dep_pha2_kernel<phasespace::y,phasespace::uz> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        }
        break;
    case( phasespace:: ux ):
        switch(quant1) {
        case( phasespace::uy ):
            _dep_pha2_kernel<phasespace::ux,phasespace::uy> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        case( phasespace::uz ):
            _dep_pha2_kernel<phasespace::ux,phasespace::uz> <<< grid, block >>> (
                d_data, range0, size0, range1, size1, particles -> nx, norm, 
                particles -> tiles, particles -> ix, particles -> x, particles -> u
            );
            break;
        }
        break;
    case( phasespace:: uy ):
        _dep_pha2_kernel<phasespace::uy,phasespace::uz> <<< grid, block >>> (
            d_data, range0, size0, range1, size1, particles -> nx, norm, 
            particles -> tiles, particles -> ix, particles -> x, particles -> u
        );
        break;
    };
}

__host__
/**
 * @brief Save 2D phasespace
 * 
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant1    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 0
 */
void Species::save_phasespace( 
    phasespace::quant quant0, float2 const range0, unsigned const size0,
    phasespace::quant quant1, float2 const range1, unsigned const size1 )
    const
{

    if ( quant0 >= quant1 ) {
        std::cerr << "(*error*) for 2D phasespaces, the 2nd quantity must be indexed higher than the first one\n";
        return;
    }

    std::string qname0, qlabel0, qunits0;
    std::string qname1, qlabel1, qunits1;

    phasespace::qinfo( quant0, qname0, qlabel0, qunits0 );
    phasespace::qinfo( quant1, qname1, qlabel1, qunits1 );
    
    // Prepare file info
    zdf::grid_axis axis[2] = {
        zdf::grid_axis {
            .name = (char *) qname0.c_str(),
            .min = range0.x,
            .max = range0.y,
            .label = (char *) qlabel0.c_str(),
            .units = (char *) qunits0.c_str()
        },
        zdf::grid_axis {
            .name = (char *) qname1.c_str(),
            .min = range1.x,
            .max = range1.y,
            .label = (char *) qlabel1.c_str(),
            .units = (char *) qunits1.c_str()
        }
    };

    if ( quant0 == phasespace::x ) {
        axis[0].min += moving_window.motion();
        axis[0].max += moving_window.motion();
    }


    std::string pha_name  = name + "-" + qname0 + qname1;
    std::string pha_label = name + " \\,(" + qlabel0 + "\\rm{-}" + qlabel1+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 2,
        .count = { size0, size1, 0 },
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    float * d_data, * h_data;
    malloc_host( h_data, size0 * size1  );
    malloc_dev( d_data, size0 * size1 );

    dep_phasespace( d_data, quant0, range0, size0, quant1, range1, size1 );

    devhost_memcpy( h_data, d_data, size0 * size1 );

    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    free_host( h_data );
    free_dev( d_data );
}