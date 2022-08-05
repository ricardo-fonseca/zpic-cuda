#include "particles.cuh"
#include <iostream>
#include "tile_zdf.cuh"
#include "timer.cuh"
#include "random.cuh"

#include "util.cuh"

#include <cmath>
#include <math.h>

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
Species::Species(const std::string name, const float m_q, const int2 ppc,
        const float n0, const float3 ufl, const float3 uth,
        const float2 box, const int2 gnx, const int2 tnx, const float dt ) :
        name{name}, m_q{m_q}, ppc{ppc}, ufl{ufl}, uth{uth}, box{box}, dt{dt}
{
    std::cout << "(*info*) Initializing species " << name << " ..." << std::endl;

    q = copysign( 1.0f, m_q ) / (ppc.x * ppc.y);
    dx.x = box.x / gnx.x;
    dx.y = box.y / gnx.y;

    // Maximum number of particles per tile
    int np_max = tnx.x * tnx.y * ppc.x * ppc.y * 2;
    
    particles = new TilePart( gnx, tnx, np_max );

    // Inject particles
    inject_particles();

    // Sets momentum of all particles
    set_u();

    // Reset iteration numbers
    d_iter = 0;
    h_iter = -1;

}

/**
 * @brief Destroy the Species:: Species object
 * 
 */
Species::~Species() {

    delete( particles );
    //delete( random );

}

__global__
/**
 * @brief Adds fluid momentum to particles
 * 
 * @param d_tile    Tile information
 * @param d_u       Particle buffer (momenta)
 * @param ufl       Fluid momentum to add
 */
void _set_uth_kernel( int2* __restrict__ d_tile, float3* __restrict__ d_u, 
    const uint2 seed, const float3 uth, const float3 ufl ) {

    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    rand_init( seed, state, norm );

    // Set particle momenta
    const int offset = d_tile[ tid ].x;
    const int np     = d_tile[ tid ].y;
    float3 __restrict__ *u  = &d_u[ offset ];

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
void Species::set_u() {

    Timer t;

    // Set thermal momentum
    dim3 grid( particles->nxtiles.x, particles->nxtiles.y );
    dim3 block( 64 );
    
    t.start();

    uint2 seed = {12345, 67890};
    _set_uth_kernel <<< grid, block >>> ( 
        particles -> d_tile, particles -> d_u, seed, uth, ufl
    );

    t.stop();
    t.report("(*info*) set_u()");
}


/**
 * @brief CUDA kernel for injecting uniform profile
 * 
 * This version does not require atomics
 * 
 * @param d_tile 
 * @param buffer_max 
 * @param d_ix 
 * @param d_x 
 * @param ppc 
 */
__global__ 
void _inject_part_kernel(
    int2* __restrict__ d_tile, size_t buffer_max,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u,
    int2 nx, int2 ppc
) {

    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Pointers to tile particle buffers
    const int offset =  d_tile[ tid ].x;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];
    
    const int np = d_tile[ tid ].y;

    const int np_cell = ppc.x * ppc.y;

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    // Each thread takes 1 cell
    for( int idx = threadIdx.x; idx < nx.x*nx.y; idx += blockDim.x ) {
        const int2 cell = { 
            idx % nx.x,
            idx / nx.x
        };

        int part_idx = np + idx * np_cell;

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                const float2 pos = {
                    static_cast<float>(dpcx * ( i0 + 0.5 )),
                    static_cast<float>(dpcy * ( i1 + 0.5 ))
                };
                ix[ part_idx ] = cell;
                x[ part_idx ] = pos;
                u[ part_idx ] = {0};
                part_idx++;
            }
        }
    }

    // Update global number of particles in tile
    if ( threadIdx.x == 0 )
        d_tile[ tid ].y = np + nx.x * nx.y * np_cell ;

}


/**
 * @brief CUDA kernel for injecting step profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param step      Step position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tile    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */
__global__
void _inject_step_kernel(
    const float step, const int2 ppc, const int2 nx, int2* __restrict__ d_tile,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u ) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    __shared__ int np;
    np = d_tile[ tid ].y;

    const int shiftx = blockIdx.x * nx.x;

    for( int idx = threadIdx.x; idx < nx.y * nx.x; idx+= blockDim.x) {
        const int2 cell = {
            idx % nx.x,
            idx / nx.y
        };
        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                const float2 pos = {
                    static_cast<float>(dpcx * ( i0 + 0.5 )),
                    static_cast<float>(dpcy * ( i1 + 0.5 ))
                };
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
        d_tile[ tid ].y = np;
}

/**
 * @brief CUDA kernel for injecting slab profile
 * 
 * This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param slab      slab start/end position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tile    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */
__global__
void _inject_slab_kernel(
    float2 slab, int2 ppc, int2 nx, int2* __restrict__ d_tile,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u
) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    __shared__ int np;
    np = d_tile[ tid ].y;

    const int shiftx = blockIdx.x * nx.x;

    for( int idx = threadIdx.x; idx < nx.y * nx.x; idx+= blockDim.x) {
        const int2 cell = {
            idx % nx.x,
            idx / nx.y
        };
        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                const float2 pos = {
                    static_cast<float>(dpcx * ( i0 + 0.5 )),
                    static_cast<float>(dpcy * ( i1 + 0.5 ))
                };
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
        d_tile[ tid ].y = np;
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
 * @param d_tile    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */
__global__
void _inject_sphere_kernel(
    float2 center, float radius, float2 dx, int2 ppc, int2 nx, int2* __restrict__ d_tile,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u
) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    __shared__ int np;
    np = d_tile[ tid ].y;

    const int shiftx = blockIdx.x * nx.x;
    const int shifty = blockIdx.y * nx.y;
    const float r2 = radius*radius;

    for( int idx = threadIdx.x; idx < nx.y * nx.x; idx+= blockDim.x) {
        const int2 cell = {
            idx % nx.x,
            idx / nx.y
        };
        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                const float2 pos = {
                    static_cast<float>(dpcx * ( i0 + 0.5 )),
                    static_cast<float>(dpcy * ( i1 + 0.5 ))
                };
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
        d_tile[ tid ].y = np;
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
    
    dim3 grid( particles->nxtiles.x, particles->nxtiles.y );
    dim3 block( 64 );

    _inject_part_kernel <<< grid, block >>> ( 
        particles -> d_tile, particles -> buffer_max,
        particles -> d_ix, particles -> d_x, particles -> d_u, 
        particles -> nx, ppc
    );
*/

/*
    // Step density

    float step = 12.8 / dx.x;

    dim3 grid( particles->nxtiles.x, particles->nxtiles.y );
    dim3 block( 32 );
    _inject_step_kernel <<< grid, block >>> (
        step, ppc, particles -> nx, 
        particles -> d_tile, 
        particles -> d_ix, particles -> d_x, particles -> d_u 
    );
*/

/*
    // Slab density
    
    float2 slab = { 10.f / dx.x, 20.f / dx.y };

    dim3 grid( particles->nxtiles.x, particles->nxtiles.y );
    dim3 block( 32 );
    _inject_slab_kernel <<< grid, block >>> (
        slab, ppc, particles -> nx, 
        particles -> d_tile, 
        particles -> d_ix, particles -> d_x, particles -> d_u 
    );
*/

    // Sphere density

    float2 center = { 12.8f, 6.4f};
    float radius = 3.2f;

    dim3 grid( particles->nxtiles.x, particles->nxtiles.y );
    dim3 block( 32 );
    _inject_sphere_kernel <<< grid, block >>> (
        center, radius, dx, ppc, particles -> nx, 
        particles -> d_tile, 
        particles -> d_ix, particles -> d_x, particles -> d_u 
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
    float3 * __restrict__ J, const int stride ) {

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
void _dep_current_split_y( const int2 ix,
    const float2 x0, const float2 x1, const float2 dx,
    const float2 qnx, const float qvz_2,
    float3 * __restrict__ J, const int nrow )
{

    const int diy = ( x1.y >= 1.0f ) - ( x1.y < 0.0f );
    if ( diy == 0 ) {
        // No more splits
        _dep_current_seg( ix, x0, x1, dx, qnx, qvz_2, J, nrow );
    } else {
        int iyb = ( diy == 1 );
        float delta = (x1.y-iyb)/dx.y;
        float xcross = x0.x + dx.x*(1-delta);

        // First segment
        _dep_current_seg( ix, x0, make_float2( xcross, iyb ), 
            make_float2( dx.x * (1-delta), dx.y * (1-delta) ), 
            qnx, qvz_2 * (1-delta), J, nrow );
        // Second segment
        _dep_current_seg( make_int2( ix.x, ix.y + diy), make_float2( xcross, 1.0f - iyb ),
            make_float2( x1.x, x1.y - diy),
            make_float2( dx.x * delta, dx.y * delta ),
            qnx, qvz_2 * delta, J, nrow );
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
void _move_deposit_kernel( int2* __restrict__ d_tile,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u,
    float3 * __restrict__ d_current, const int current_offset, const int2 ext_nx,
    const float2 dt_dx, const float q, const float2 qnx ) {
    
    extern __shared__ float3 _move_deposit_buffer[];
    
    // Zero current buffer
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        _move_deposit_buffer[i].x = 0;
        _move_deposit_buffer[i].y = 0;
        _move_deposit_buffer[i].z = 0;
    }

    __syncthreads();

    // Move particles and deposit current
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    float3 * J = _move_deposit_buffer + current_offset;
    const int stride = ext_nx.x;

    const int part_offset = d_tile[ tid ].x;
    const int np     = d_tile[ tid ].y;
    int2   __restrict__ *ix  = &d_ix[ part_offset ];
    float2 __restrict__ *x   = &d_x[ part_offset ];
    float3 __restrict__ *u   = &d_u[ part_offset ];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        float3 pu = u[i];
        float2 x0 = x[i];
        int2 ix0 =ix[i];

        // Get 1 / Lorentz gamma
        float rg = 1.0f / sqrtf(1.0f + pu.x*pu.x + pu.y*pu.y + pu.z*pu.z);

        // Get particle motion
        float2 deltax = {
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        };

        // Advance position
        float2 x1 = {
            x0.x + deltax.x,
            x0.y + deltax.y
        };

        // Check for cell crossings
        int2 deltaix = {
            ((x1.x >= 1.0f) - (x1.x < 0.0f)),
            ((x1.y >= 1.0f) - (x1.y < 0.0f))
        };

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
        int2 ix1 = {
            ix0.x + deltaix.x,
            ix0.y + deltaix.y
        };
        ix[i] = ix1;
    }

    __syncthreads();

    // Copy current to global buffer
    const int tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * ext_nx.x * ext_nx.y;
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_current[tile_off + i] = _move_deposit_buffer[i];
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
void Species::move_deposit( VFLD &current ) {

    const float2 dt_dx = {
        dt / dx.x,
        dt / dx.y
    };

    const float2 qnx = {
        q * dx.x / dt,
        q * dx.y / dt
    };

    dim3 grid( particles->nxtiles.x, particles->nxtiles.y );
    dim3 block( 64 );
    int2 ext_nx = current.ext_nx();
    size_t shm_size = ext_nx.x * ext_nx.y * sizeof(float3);

    _move_deposit_kernel <<< grid, block, shm_size >>> ( 
        particles -> d_tile, 
        particles -> d_ix, particles -> d_x, particles -> d_u,
        current.d_buffer, current.offset(), ext_nx, dt_dx, q, qnx
    );

    current.add_from_gc();

    d_iter++;
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
void _dep_charge_kernel( float* __restrict__ d_charge, int offset, int2 ext_nx,
    int2* __restrict__ d_tile, int2* __restrict__ d_ix, float2* __restrict__ d_x, const float q ) {

    extern __shared__ float _dep_charge_buffer[];

    // Zero shared memory and sync.
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        _dep_charge_buffer[i] = 0;
    }
    float *charge = &_dep_charge_buffer[ offset ];

    __syncthreads();

    const int tid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int part_off =  d_tile[ tid ].x;
    const int np = d_tile[ tid ].y;
    int2   __restrict__ *ix = &d_ix[ part_off ];
    float2 __restrict__ *x  = &d_x[ part_off ];
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

    __syncthreads();

    // Copy data to global memory
    const int tile_off = tid * ext_nx.x * ext_nx.y;
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_charge[tile_off + i] = _dep_charge_buffer[i];
    } 
}

__host__
/**
 * @brief Deposit charge density
 * 
 * @param charge    Charge density grid
 */
void Species::deposit_charge( Field &charge ) {

    int2 ext_nx = charge.ext_nx();
    dim3 grid( charge.nxtiles.x, charge.nxtiles.y );
    dim3 block( 64 );

    size_t shm_size = ext_nx.x * ext_nx.y * sizeof(float);

    _dep_charge_kernel <<< grid, block, shm_size >>> (
        charge.d_buffer, charge.offset(), ext_nx,
        particles -> d_tile, particles -> d_ix, particles -> d_x, q
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

    t_zdf_iteration iter = {
        .n = d_iter,
        .t = d_iter * dt,
        .time_units = (char *) "1/\\omega_p"
    };

    // Get number of particles on device
    const size_t np = particles->device_np();

    t_zdf_part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .np = np,
        .nquants = 5,
        .quants = (char **) quants,
        .qlabels = (char **) qlabels,
        .qunits = (char **) qunits,
    };

    // Create file and add description
    t_zdf_file part_file;

    std::string path = "PARTICLES/";
    path += name;
    zdf_open_part_file( &part_file, &info, &iter, path.c_str() );

    float * h_data;
    cudaError_t err = cudaMallocHost( &h_data, np * sizeof(float) );
    CHECK_ERR( err, "Failed to allocate host memory for h_data" );
    
    particles -> gather( TilePart::x, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[0], h_data, np );

    particles -> gather( TilePart::y, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[1], h_data, np );

    particles -> gather( TilePart::ux, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[2], h_data, np );

    particles -> gather( TilePart::uy, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[3], h_data, np );

    particles -> gather( TilePart::uz, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[4], h_data, np );

    err = cudaFreeHost( h_data );
    CHECK_ERR( err, "Failed to free host memory for h_data" );

    zdf_close_file( &part_file );
}

/**
 * @brief Saves charge density to file
 * 
 */
void Species::save_charge() {

    const int2 gnx = particles -> g_nx();
    const int2 nx = particles -> nx;

    // For linear interpolation we only require 1 guard cell at the upper boundary
    const int2 gc[2] = {
        {0,0},
        {1,1}
    };

    // Deposit charge on device
    Field charge( gnx, nx, gc );

    // This will also zero the charge density initially
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

    t_zdf_iteration iter = {
        .name = (char *) "ITERATION",
        .n = d_iter,
        .t = d_iter * dt,
        .time_units = (char *) "1/\\omega_p"
    };

    std::string path = "CHARGE/";
    path += name;
    
    charge.save( info, iter, path.c_str() );
}