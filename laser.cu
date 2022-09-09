#include "laser.cuh"
#include <iostream>
#include <cassert>

#include <cooperative_groups.h>
namespace cg=cooperative_groups;

#include "filter.cuh"
#include "util.cuh"

__host__
/**
 * @brief Validates laser parameters
 * 
 * @return      0 on success, -1 on error
 */
int Laser::Pulse::validate() {

    if ( a0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser a0, must be > 0" << std::endl;
        return -1;
    }    

    if ( omega0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser OMEGA0, must be > 0" << std::endl;
        return -1;
    }    

    if ( fwhm > 0 ) {
        // The fwhm parameter overrides the rise/flat/fall parameters
        rise = fwhm;
        fall = fwhm;
        flat = 0.;
    } else {
        if ( rise <= 0 ) {
            std::cerr << "(*error*) Invalid laser RISE, must be > 0" << std::endl;
            return (-1);
        }

        if ( flat < 0 ) {
            std::cerr << "(*error*) Invalid laser FLAT, must be >= 0" << std::endl;
            return (-1);
        }

        if ( fall <= 0 ) {
            std::cerr << "(*error*) Invalid laser FALL, must be > 0" << std::endl;
            return (-1);
        }
    }

    return 0;
}

 
/**
 * @brief Gets longitudinal laser envelope a given position
 * 
 * @param laser     Laser parameters
 * @param z         position
 * @return          laser envelope
 */
__device__
float lon_env( Laser::Pulse & laser, float z ) {

    if ( z > laser.start ) {
        // Ahead of laser
        return 0.0;
    } else if ( z > laser.start - laser.rise ) {
        // Laser rise
        float csi = z - laser.start;
        float e = sin( M_PI_2 * csi / laser.rise );
        return e*e;
    } else if ( z > laser.start - (laser.rise + laser.flat) ) {
        // Flat-top
        return 1.0;
    } else if ( z > laser.start - (laser.rise + laser.flat + laser.fall) ) {
        // Laser fall
        float csi = z - (laser.start - laser.rise - laser.flat - laser.fall);
        float e = sin( M_PI_2 * csi / laser.fall );
        return e*e;
    }

    // Before laser
    return 0.0;
}


/**
 * @brief CUDA kernel for launching a plane wave
 * 
 * Kernel must be launched with a grid [ntiles.x,ntiles.y] and block [nthreads]
 * 
 * @param laser     Laser parameters
 * @param E         Pointer to E field data including offset
 * @param B         Pointer to B field data including offset
 * @param int_nx    Internal tile size
 * @param ext_nx    External tile size
 * @param dx        Cell size
 */
__global__
void _plane_wave_kernel( Laser::PlaneWave laser, 
    float3 * __restrict__ E, float3 * __restrict__ B,
    uint2 int_nx, uint2 ext_nx, float2 const dx )
{
    const int tile_off = (blockIdx.y * gridDim.x + blockIdx.x) * ( ext_nx.x * ext_nx.y );

    const int ix0 = blockIdx.x * int_nx.x;

    const float k = laser.omega0;
    const float amp = laser.omega0 * laser.a0;

    for( int i = threadIdx.x; i < int_nx.x * int_nx.y; i+= blockDim.x ) {
        int const ix = i % int_nx.x;
        int const iy = i / int_nx.x;

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

        float lenv   = amp * lon_env( laser, z   );
        float lenv_2 = amp * lon_env( laser, z_2 );

        const int idx = tile_off + iy * ext_nx.x + ix;
        
        E[ idx ] = make_float3(
            0,
            +lenv * cos( k * z ) * laser.cos_pol,
            +lenv * cos( k * z ) * laser.sin_pol
        );

        B[ idx ] = make_float3(
            0,
            -lenv_2 * cos( k * z_2 ) * laser.sin_pol,
            +lenv_2 * cos( k * z_2 ) * laser.cos_pol
        );
    }
}

/**
 * @brief Launches a plane wave
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param box   Box size
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
__host__
int Laser::PlaneWave::launch( VectorField& E, VectorField& B, float2 box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    uint2 g_nx = E.g_nx();

    float2 dx = make_float2(
        box.x / g_nx.x,
        box.y / g_nx.y
    );

    uint2 ext_nx = E.ext_nx();
    unsigned int offset = E.offset();

    dim3 block( 64 );
    dim3 grid( E.ntiles.x, E.ntiles.y );

    _plane_wave_kernel <<< grid, block >>> ( *this,
        E.d_buffer + offset, B.d_buffer + offset,
        E.nx, ext_nx, dx
    );

    E.copy_to_gc();
    B.copy_to_gc();

    if ( filter > 0 ) {
        Filter::Compensated fcomp( coord::x, filter);
        fcomp.apply(E);
        fcomp.apply(B);
    }

    return 0;
}

/**
 * @brief Validate Gaussian laser parameters
 * 
 * @return      0 on success, -1 on error
 */
__host__
int Laser::Gaussian::validate() {
    
    if ( Laser::Pulse::validate() < 0 ) {
        return -1;
    }

    if ( W0 <= 0 ) {
        std::cerr << "(*error*) Invalid laser W0, must be > 0" << std::endl;
        return (-1);
    }

    return 0;
}


__device__
/**
 * @brief Returns local phase for a gaussian beamn
 * 
 * @param omega0    Beam frequency
 * @param W0        Beam waist
 * @param z         Position along focal line (focal plane at z = 0)
 * @param r         Position transverse to focal line (focal line at r = 0)
 * @return          Local field value
 */
float gauss_phase( const float omega0, const float W0, const float z, const float r ) {
    const float z0   = omega0 * ( W0 * W0 ) / 2;
    const float rho2 = r*r;
    const float curv = rho2 * z / (z0*z0 + z*z);
    const float rWl2 = (z0*z0)/(z0*z0 + z*z);
    const float gouy_shift = atan2( z, z0 );

    return sqrt( sqrt(rWl2) ) * 
        exp( - rho2 * rWl2/( W0 * W0 ) ) * 
        cos( omega0*( z + curv ) - gouy_shift );
}


/**
 * @brief Launch transverse components of Gaussian beam
 *
 * Kernel must be launched with a grid [ntiles.x,ntiles.y] and block [nthreads]
 * 
 * @param beam      Gaussian beam parameters
 * @param E         Electric field
 * @param B         Magnetic field
 * @param int_nx    Internal tile size
 * @param ext_nx    External tile size
 * @param dx        Cell size
 */
__global__
void _gaussian_kernel( Laser::Gaussian beam, 
    float3 * const __restrict__ E, float3 * const __restrict__ B,
    uint2 int_nx, uint2 ext_nx, float2 const dx )
{
    int    tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    size_t tile_off = tile_id * ext_nx.x * ext_nx.y;

    const int ix0 = blockIdx.x * int_nx.x;
    const int iy0 = blockIdx.y * int_nx.y;

    const float amp = beam.omega0 * beam.a0;

    for( int i = threadIdx.x; i < int_nx.x * int_nx.y; i+= blockDim.x ) {
        const int ix = i % int_nx.x;
        const int iy = i / int_nx.x;

        const float z   = ( ix0 + ix ) * dx.x;
        const float z_2 = ( ix0 + ix + 0.5 ) * dx.x;

        const float r   = (iy0 + iy ) * dx.y - beam.axis;
        const float r_2 = (iy0 + iy + 0.5 ) * dx.y - beam.axis;

        const float lenv   = amp * lon_env( beam, z   );
        const float lenv_2 = amp * lon_env( beam, z_2 );

        size_t const idx = tile_off + iy * ext_nx.x + ix;

        E[ idx ] = make_float3(
            0,
            +lenv * gauss_phase( beam.omega0, beam.W0, z - beam.focus, r_2 ) * beam.cos_pol,
            +lenv * gauss_phase( beam.omega0, beam.W0, z - beam.focus, r   ) * beam.sin_pol
        );
        B[ idx ] = make_float3(
            0,
            -lenv_2 * gauss_phase( beam.omega0, beam.W0, z_2 - beam.focus, r   ) * beam.sin_pol,
            +lenv_2 * gauss_phase( beam.omega0, beam.W0, z_2 - beam.focus, r_2 ) * beam.cos_pol
        );
    }
}

/**
 * @brief CUDA kernel for div_corr_x, step A
 * 
 * Get per-tile E and B divergence at tile left edge starting from 0.0
 * Kernel must be launched with a grid [ntiles.x,ntiles.y] and block [nthreads]
 * It also required dynamic shared memory buffer for 2 tiles
 * 
 * Parallelization:
 * - Use all threads for coherently copying in field values;
 * - Use 1 thread per line for divergence calculation.
 */
__global__
void _div_corr_x_kernel_A( 
    float3 * const __restrict__ d_E, 
    float3 * const __restrict__ d_B,
    uint2 const int_nx, uint2 const ext_nx, unsigned int const offset,
    float2 dx, double2 * const __restrict__ tmp )
{
    auto group = cg::this_thread_block();

    extern __shared__ float3 buffer[];
    
    const int tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * ext_nx.x * ext_nx.y;
    const int B_off = ext_nx.x * ext_nx.y;

    // Copy E and B into shared memory and sync
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        buffer[i        ] = d_E[tile_off + i];
        buffer[B_off + i] = d_B[tile_off + i];
    }
    group.sync();

    float3 * const __restrict__ E = buffer + offset;
    float3 * const __restrict__ B = E + B_off; 

    // Process 
    const double dx_dy = (double) dx.x / (double) dx.y;
    const unsigned int tmp_off = blockIdx.y * int_nx.y * gridDim.x;

    for( int iy = threadIdx.x; iy < int_nx.y; iy += blockDim.x ) {
        
        // Find divergence at left edge
        double divEx = 0;
        double divBx = 0;
        for( int ix = int_nx.x - 1; ix >= 0; ix-- ) {
            divEx += dx_dy * (E[ix+1 + iy*ext_nx.x].y - E[ix+1 + (iy-1)*ext_nx.x ].y);
            divBx += dx_dy * (B[ix + (iy+1)*ext_nx.x].y - B[ix + iy*ext_nx.x ].y);
        }

        // Write result to tmp. array
        tmp[ tmp_off + iy * gridDim.x + blockIdx.x ] = make_double2( divEx, divBx );
    }
}

/**
 * @brief CUDA kernel for div_corr_x, step B
 * 
 * Performs a left-going scan operation on the results from step A.
 * Must be called with a grid [ gnx.y ] and block [ nthreads ]
 * 
 * @param tmp       Temporary array holding results from step B
 * @param ntiles   Global tile configuration
 */
__global__
void _div_corr_x_kernel_B( double2 * const __restrict__ tmp, uint2 ntiles )
{
    auto group = cg::this_thread_block();
    extern __shared__ double2 buffer_B[];

    int giy = blockIdx.x;

    // Copy data into shared memory and sync
    for( int i = threadIdx.x; i < ntiles.x; i += blockDim.x ) {
        buffer_B[i] = tmp[ giy * ntiles.x + i ];
    }
    group.sync();

    // Perform scan operation (serial inside block)
    if ( threadIdx.x == 0 ) {
        double2 a = make_double2(0,0);
        for( int i = ntiles.x-1; i >= 0; i--) {
            double2 b = buffer_B[i];
            buffer_B[i] = a;
            a.x += b.x;
            a.y += b.y;
        }
    }

    // Copy data to global memory
    for( int i = threadIdx.x; i < ntiles.x; i += blockDim.x ) {
        tmp[ giy * ntiles.x + i ] = buffer_B[i];
    }
}

/**
 * @brief CUDA kernel for div_corr_x, step C
 * 
 * 
 * 
 * @param d_E 
 * @param d_B 
 * @param int_nx 
 * @param ext_nx 
 * @param dx 
 * @param tmp 
 */
__global__
void _div_corr_x_kernel_C( 
    float3 * const __restrict__ d_E,
    float3 * const __restrict__ d_B,
    uint2 const int_nx, uint2 const ext_nx, unsigned int const offset,
    float2 const dx, double2 const * const __restrict__ tmp )
{
    auto group = cg::this_thread_block();
    
    extern __shared__ float3 buffer[];
    
    const int tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * ext_nx.x * ext_nx.y;
    const int B_off = ext_nx.x * ext_nx.y;

    // Copy E and B into shared memory and sync
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        buffer[i] = d_E[tile_off + i];
        buffer[B_off + i] = d_B[tile_off + i];
    }
    group.sync();

    float3 * const __restrict__ E = buffer + offset;
    float3 * const __restrict__ B = E + B_off; 

    // Process 
    double const dx_dy = (double) dx.x / (double) dx.y;
    unsigned int const tmp_off = blockIdx.y * int_nx.y * gridDim.x;

    for( int iy = threadIdx.x; iy < int_nx.y; iy += blockDim.x ) {
        
        // Get divergence at right edge
        double2 div = tmp[ tmp_off + iy * gridDim.x + blockIdx.x ];

        double divEx = div.x;
        double divBx = div.y;

        for( int ix = int_nx.x - 1; ix >= 0; ix-- ) {
            divEx += dx_dy * (E[ix+1 + iy*ext_nx.x].y - E[ix+1 + (iy-1)*ext_nx.x ].y);
            E[ ix + iy * ext_nx.x].x = divEx;

            divBx += dx_dy * (B[ix + (iy+1)*ext_nx.x].y - B[ix + iy*ext_nx.x ].y);
            B[ ix + iy * ext_nx.x].x = divBx;
        }
    }
    group.sync();

    // Copy data to device memory
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_E[tile_off + i] = buffer[i];
        d_B[tile_off + i] = buffer[B_off + i];
    }
}


/**
 * @brief Sets the longitudinal field components of E and B to ensure 0 divergence
 * 
 * The algorithm assumes 0 field at the right boundary of the box
 * 
 * @param E 
 * @param B 
 * @param dx 
 */
void div_corr_x(VectorField& E, VectorField& B, float2 const dx )
{

    // A. Get accumulated E and B x divergence at the left edge of each
    //    tile (starting at 0 on right edge)
    double2* tmp;
    size_t bsize = E.ntiles.x * (E.ntiles.y * E.nx.y) * sizeof( double2 );
    auto err = cudaMalloc( &tmp, bsize );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate device memory for div_corr_x." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    uint2 const ext_nx = E.ext_nx();
    unsigned int const offset = E.offset();

    dim3 grid( E.ntiles.x, E.ntiles.y );
    dim3 block( 32 );
    size_t shm_size = E.ext_vol() * 2 * sizeof(float3);
    
    // Shared memory size must be below 48 k
    if ( shm_size >= 49152 ) {
        std::cerr << "(*error*) Unable to correct divergence, too much shared memory required " << std::endl;
        std::cerr << "(*error*) Please retry with a smaller tile size" << std::endl;
        return;
    }

    _div_corr_x_kernel_A <<< grid, block, shm_size >>> ( 
        E.d_buffer, B.d_buffer, 
        E.nx, ext_nx, offset,
        dx, tmp
    );

    // B. Left-scan the divergences
    dim3 grid_B( E.ntiles.y * E.nx.y );
    dim3 block_B( E.ntiles.x > 32 ? 32 : E.ntiles.x );
    size_t shm_size_B = E.ntiles.x * sizeof(double2);

    if ( shm_size_B >= 49152 ) {
        std::cerr << "(*error*) Unable to correct divergence, too much shared memory required (shm_size_B)" << std::endl;
        std::cerr << "(*error*) Please retry with a smaller tile size" << std::endl;
        return;
    }

    _div_corr_x_kernel_B <<< grid_B, block_B, shm_size_B >>> (
        tmp, E.ntiles
    );

    // C. Set longitudinal field values
    _div_corr_x_kernel_C <<< grid, block, shm_size >>> ( 
        E.d_buffer, B.d_buffer,
        E.nx, ext_nx, offset,
        dx, tmp
    );

    err = cudaFree( tmp );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to deallocate device memory in div_corr_x." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Correct longitudinal values on guard cells
    E.copy_to_gc();
    B.copy_to_gc();
}


/**
 * @brief Launches a Gaussian pulse
 * 
 * The E and B tiled grids have the complete laser field.
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 * @param dx    Cell size
 * @return      Returns 0 on success, -1 on error (invalid laser parameters)
 */
__host__
int Laser::Gaussian::launch(VectorField& E, VectorField& B, float2 const box ) {

    if ( validate() < 0 ) return -1;

    if (( cos_pol == 0 ) && ( sin_pol == 0 )) {
        cos_pol = cos( polarization );
        sin_pol = sin( polarization );
    }

    float2 dx = {
        .x = box.x / E.g_nx().x,
        .y = box.y / E.g_nx().y
    };

    uint2 const ext_nx = E.ext_nx();
    unsigned int const offset  = E.offset();

    dim3 grid( E.ntiles.x, E.ntiles.y );
    dim3 block( 64 );

    _gaussian_kernel <<< grid, block >>> ( 
        *this,
        E.d_buffer + offset, B.d_buffer + offset,
        E.nx, ext_nx, dx
    );

    E.copy_to_gc();
    B.copy_to_gc();

    if ( filter > 0 ) {
        Filter::Compensated fcomp( coord::x, filter );
        fcomp.apply(E);
        fcomp.apply(B);
    }

    div_corr_x( E, B, dx );

    return 0;
}