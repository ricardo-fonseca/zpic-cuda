#include "emf.cuh"
#include <iostream>

#include <cooperative_groups.h>
namespace cg=cooperative_groups;

/**
 * @brief Construct a new EMF::EMF object
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 * @param box       Simulation box size
 * @param dt        Time step
 */
__host__
EMF::EMF( uint2 const ntiles, uint2 const nx, float2 const box,
    float const dt ) : box{box}, dt{dt}
{
    // Set box limits, cells sizes and time step
    dx.x = box.x / ( nx.x * ntiles.x );
    dx.y = box.y / ( nx.y * ntiles.y );

    // Verify Courant condition
    float cour = sqrtf( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
    if ( dt >= cour ){
        std::cerr << "(*error*) Invalid timestep, courant condition violation.\n";
        std::cerr << "(*error*) For current resolution (" << dx.x << "," << dx.y <<
            ") the maximum timestep is dt = " << cour <<"\n";
        exit(-1);
    }

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    E = new VectorField( ntiles, nx, gc );
    B = new VectorField( ntiles, nx, gc );

    // Zero fields
    E -> zero();
    B -> zero();

    // Set default boundary conditions to periodic
    bc = emf::bc_type (emf::bc::periodic);

    // Reset iteration number
    iter = 0;

    // Reserve device memory for energy diagnostic
    malloc_dev( d_energy, 6 );

}

/**
 * @brief Destroy the EMF::EMF object
 * 
 */
__host__
EMF::~EMF(){
    delete (E);
    delete (B);

    free_dev( d_energy );
}

/**
 * @brief B advance for Yee algorithm
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param stride    Stride for j coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
__device__
void yee_b( 
    float3 const * const __restrict__ E, 
    float3 * const __restrict__ B, 
    uint2 const nx, unsigned int const stride, 
    float const dt_dx, float const dt_dy )
{
    unsigned int const vol = ( nx.x + 2 ) * ( nx.y + 2 );
    int const step = nx.x + 2;

    // The y and x loops are fused into a single loop to improve parallelism
    for( int idx = threadIdx.x; idx < vol; idx += blockDim.x ) {
        const int i = -1 + idx % step;  // range is -1 to nx
        const int j = -1 + idx / step;
        
        B[ i + j*stride ].x += ( - dt_dy * ( E[i+(j+1)*stride].z - E[i+j*stride].z ) );  
        B[ i + j*stride ].y += (   dt_dx * ( E[(i+1)+j*stride].z - E[i+j*stride].z ) );  
        B[ i + j*stride ].z += ( - dt_dx * ( E[(i+1)+j*stride].y - E[i+j*stride].y ) + 
                                   dt_dy * ( E[i+(j+1)*stride].x - E[i+j*stride].x ) );  
    }
}


/**
 * @brief E advance for Yee algorithm ( no current )
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param stride    Stride for j coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
__device__
void yee_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, unsigned int const stride, 
    float const dt_dx, float const dt_dy )
{
    unsigned int const vol = ( nx.x + 2 ) * ( nx.y + 2 );

    // The y and x loops are fused into a single loop to improve parallelism
    for( int idx = threadIdx.x; idx < vol; idx += blockDim.x ) {
        const int i = idx % ( nx.x + 2 );   // range is 0 to nx+1
        const int j = idx / ( nx.x + 2 );

        E[i+j*stride].x += ( + dt_dy * ( B[i+j*stride].z - B[i+(j-1)*stride].z) );
        
        E[i+j*stride].y += ( - dt_dx * ( B[i+j*stride].z - B[(i-1)+j*stride].z) );

        E[i+j*stride].z += ( + dt_dx * ( B[i+j*stride].y - B[(i-1)+j*stride].y) - 
                               dt_dy * ( B[i+j*stride].x - B[i+(j-1)*stride].x) );
    }
}

__global__ 
/**
 * @brief CUDA kernel for advancing EM field 1 timestep
 * 
 * @param d_E       E field grid
 * @param d_B       B field grid
 * @param int_nx    Tile size (internal)
 * @param ext_nx    Tile size (external) i.e including guard cells
 * @param offset    Offset to position (0,0) on tile
 * @param dt_dx     Time step over cell size
 */
void yee_kernel( 
    float3 * const __restrict__ d_E,
    float3 * const __restrict__ d_B,
    uint2 const int_nx, uint2 const ext_nx, unsigned int const offset, 
    float2 const dt_dx ) {

    auto block = cg::this_thread_block();
    extern __shared__ float3 buffer[];
    
    const float dt_dx2 = dt_dx.x / 2.0f;
    const float dt_dy2 = dt_dx.y / 2.0f;

    const int tid      = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int tile_off = tid * tile_vol;

    const int B_off = tile_vol;

    // Copy E and B into shared memory and sync
    for( int i = block.thread_rank(); i < ext_nx.x * ext_nx.y; i += block.num_threads() ) {
        buffer[i        ] = d_E[tile_off + i];
        buffer[B_off + i] = d_B[tile_off + i];
    }

    float3 * const E = buffer + offset;
    float3 * const B = E + B_off; 

    // Perform half B field advance
    block.sync();
    yee_b( E, B , int_nx, ext_nx.x, dt_dx2, dt_dy2 );

    // Perform full E field advance
    block.sync();
    yee_e( E, B, int_nx, ext_nx.x, dt_dx.x, dt_dx.y );

    // Perform half B field advance and sync
    block.sync();
    yee_b( E, B, int_nx, ext_nx.x, dt_dx2, dt_dy2 );
 
    // Copy data to global memory
    block.sync();
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_E[tile_off + i] = buffer[i];
        d_B[tile_off + i] = buffer[B_off + i];
    }
}

/**
 * @brief Move simulation window
 * 
 * When using a moving simulation window checks if a window move is due
 * at the current iteration and if so shifts left the data, zeroing the
 * rightmost cells.
 * 
 */
void EMF::move_window() {

    if ( moving_window.needs_move( iter * dt ) ) {

        E->x_shift_left(1);
        B->x_shift_left(1);

        moving_window.advance();
    }
}


/**
 * @brief Advance EM fields 1 time step (no current)
 * 
 */
__host__
void EMF::advance() {

    // Tile block size (grid + guard cells)
    uint2 ext_nx = E -> ext_nx();

    float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    dim3 grid( E->ntiles.x, E->ntiles.y );
    dim3 block( 64 );

    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    size_t shm_size = 2 * tile_vol * sizeof(float3);

    // Advance EM field using Yee algorithm modified for having E and B time centered
    yee_kernel <<< grid, block, shm_size >>> ( 
        E->d_buffer, B->d_buffer, 
        E->nx, E->ext_nx(), E->offset(), dt_dx
    );

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Do additional bc calculations if needed
    process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    if ( moving_window.active() ) move_window();

}


__global__
/**
 * @brief CUDA kernel for processing EM physical boundaries along x
 * 
 * This kernel must be launched with 2 * ntiles.y blocks
 * 
 * @param d_E       E field grid
 * @param d_B       B field grid
 * @param int_nx    Tile size (internal)
 * @param ext_nx    Tile size (external)
 * @param gc        Number of guard cells
 * @param ntiles_x  Number of tiles along the x direction
 * @param bc        Type of boundary condition
 */
void _emf_bcx( 
    float3 * const __restrict__ d_E,
    float3 * const __restrict__ d_B,
    uint2 const int_nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    uint2 const ntiles, emf::bc_type bc )
{
    const int tid = blockIdx.y * ntiles.x + blockIdx.x * (ntiles.x - 1);

    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int tile_off = tid * tile_vol;

    const int ystride = ext_nx.x;
    const int offset   = gc.x.lower;

    float3 * const __restrict__ E = d_E + tile_off + offset;
    float3 * const __restrict__ B = d_B + tile_off + offset;


    // Lower boundary
    if ( blockIdx.x == 0 ) {
        switch( bc.x.lower ) {
        case( emf::bc::pmc) :
            for( int idx = threadIdx.x; idx < ext_nx.y; idx += blockDim.x ) {
                // j includes the y-stride
                const int j = idx * ystride;

                E[ -1 + j ].x = -E[ 0 + j ].x;
                E[ -1 + j ].y =  E[ 1 + j ].y;
                E[ -1 + j ].z =  E[ 1 + j ].z;

                B[ -1 + j ].x =  B[ 1 + j ].x;
                B[ -1 + j ].y = -B[ 0 + j ].y;
                B[ -1 + j ].z = -B[ 0 + j ].z;

            }
            break;

        case( emf::bc::pec ) :
            for( int idx = threadIdx.x; idx < ext_nx.y; idx += blockDim.x ) {
                const int j = idx * ystride;

                E[ -1 + j ].x =  E[ 0 + j ].x;
                E[ -1 + j ].y = -E[ 1 + j ].y;
                E[ -1 + j ].z = -E[ 1 + j ].z;

                E[  0 + j ].y = 0;
                E[  0 + j ].z = 0;

                B[ -1 + j ].x = -B[ 1 + j ].x;
                B[ -1 + j ].y =  B[ 0 + j ].y;
                B[ -1 + j ].z =  B[ 0 + j ].z;

                B[  0 + j ].x = 0;
            }
            break;
        }
    // Upper boundary
    } else {
        switch( bc.x.upper ) {
        case( emf::bc::pmc) :
            for( int idx = threadIdx.x; idx < ext_nx.y; idx += blockDim.x ) {
                int j = idx * ystride;

                E[ int_nx.x + j ].x = -E[ int_nx.x-1 + j ].x;
                //E[ int_nx.x + j ].y =  E[ int_nx.x + j ].y;
                //E[ int_nx.x + j ].z =  E[ int_nx.x + j ].z;

                E[ int_nx.x+1 + j ].x = -E[ int_nx.x-2 + j ].x;
                E[ int_nx.x+1 + j ].y =  E[ int_nx.x-1 + j ].y;
                E[ int_nx.x+1 + j ].z =  E[ int_nx.x-1 + j ].z;

                // B[ int_nx.x + j ].x = -B[ int_nx.x + j ].x;
                B[ int_nx.x + j ].y = -B[ int_nx.x-1 + j ].y;
                B[ int_nx.x + j ].z = -B[ int_nx.x-1 + j ].z;

                B[ int_nx.x+1 + j ].x =  B[ int_nx.x-1 + j ].x;
                B[ int_nx.x+1 + j ].y = -B[ int_nx.x-2 + j ].y;
                B[ int_nx.x+1 + j ].z = -B[ int_nx.x-2 + j ].z;
            }
            break;

        case( emf::bc::pec) :
            for( int idx = threadIdx.x; idx < ext_nx.y; idx += blockDim.x ) {
                int j = idx * ystride;

                E[ int_nx.x + j ].x =  E[ int_nx.x-1 + j ].x;
                E[ int_nx.x + j ].y =  0;
                E[ int_nx.x + j ].z =  0;

                E[ int_nx.x+1 + j ].x =  E[ int_nx.x-2 + j ].x;
                E[ int_nx.x+1 + j ].y = -E[ int_nx.x-1 + j ].y;
                E[ int_nx.x+1 + j ].z = -E[ int_nx.x-1 + j ].z;

                B[ int_nx.x + j ].x =  0;
                B[ int_nx.x + j ].y =  B[ int_nx.x-1 + j ].y;
                B[ int_nx.x + j ].z =  B[ int_nx.x-1 + j ].z;

                B[ int_nx.x+1 + j ].x = -B[ int_nx.x-1 + j ].x;
                B[ int_nx.x+1 + j ].y =  B[ int_nx.x-2 + j ].y;
                B[ int_nx.x+1 + j ].z =  B[ int_nx.x-2 + j ].z;
            }
            break;
        }
    }
}

__global__
void _emf_bcy(
    float3 * const __restrict__ d_E,
    float3 * const __restrict__ d_B,
    uint2 const int_nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    uint2 const ntiles, emf::bc_type bc )
{
    const int tid = blockIdx.y * (ntiles.y - 1) * ntiles.x + blockIdx.x;

    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int tile_off = tid * tile_vol;
    const int ystride = ext_nx.x;
    const int offset   = gc.y.lower * ystride;

    float3 * const __restrict__ E = d_E + tile_off + offset;
    float3 * const __restrict__ B = d_B + tile_off + offset;


    // Lower boundary
    if ( blockIdx.y == 0 ) {
        switch( bc.y.lower ) {
        case( emf::bc::pmc) :
            for( int idx = threadIdx.x; idx < ext_nx.x; idx += blockDim.x ) {
                int i = idx;

                E[ i - ystride ].x =  E[ i + ystride ].x;
                E[ i - ystride ].y = -E[ i +       0 ].y;
                E[ i - ystride ].z =  E[ i + ystride ].z;

                B[ i - ystride ].x = -B[ i +       0 ].x;
                B[ i - ystride ].y =  B[ i + ystride ].y;
                B[ i - ystride ].z = -B[ i +       0 ].z;
            }
            break;

        case( emf::bc::pec ) :
            for( int idx = threadIdx.x; idx < ext_nx.x; idx += blockDim.x ) {
                int i = idx;

                E[ i - ystride ].x = -E[ i + ystride ].x;
                E[ i - ystride ].y =  E[ i +       0 ].y;
                E[ i - ystride ].z = -E[ i + ystride ].z;

                E[ i +       0 ].x = 0;
                E[ i +       0 ].z = 0;
                
                B[ i - ystride ].x =  B[ i +       0 ].x;
                B[ i - ystride ].y = -B[ i + ystride ].y;
                B[ i - ystride ].z =  B[ i +       0 ].z;

                B[ i +       0 ].y = 0;
            }
            break;
        }
    // Upper boundary
    } else {
        switch( bc.y.upper ) {
        case( emf::bc::pmc) :
            for( int idx = threadIdx.x; idx < ext_nx.x; idx += blockDim.x ) {
                int i = idx;

                E[ i + int_nx.y * ystride ].y = -E[ i + (int_nx.y-1) * ystride ].y;

                E[ i + (int_nx.y+1) * ystride ].x =  E[ i + (int_nx.y-1) * ystride ].x;
                E[ i + (int_nx.y+1) * ystride ].y = -E[ i + (int_nx.y-2) * ystride ].y;
                E[ i + (int_nx.y+1) * ystride ].z =  E[ i + (int_nx.y-1) * ystride ].z;

                B[ i + (int_nx.y) * ystride ].x = -B[ i + (int_nx.y-1)*ystride ].x;
                B[ i + (int_nx.y) * ystride ].z = -B[ i + (int_nx.y-1)*ystride ].z;

                B[ i + (int_nx.y+1) * ystride ].x = -B[ i + (int_nx.x-2) * ystride ].x;
                B[ i + (int_nx.y+1) * ystride ].y =  B[ i + (int_nx.x-1) * ystride ].y;
                B[ i + (int_nx.y+1) * ystride ].z = -B[ i + (int_nx.x-2) * ystride ].z;
            }
            break;

        case( emf::bc::pec) :
            for( int idx = threadIdx.x; idx < ext_nx.x; idx += blockDim.x ) {
                int i = idx;

                E[ i + (int_nx.y)*ystride ].x =  0;
                E[ i + (int_nx.y)*ystride ].y =  E[ i + (int_nx.y-1)*ystride ].y;
                E[ i + (int_nx.y)*ystride ].z =  0;

                E[ i + (int_nx.y+1)*ystride ].x = -E[ i + (int_nx.x-1) * ystride ].x;
                E[ i + (int_nx.y+1)*ystride ].y =  E[ i + (int_nx.x-2) * ystride ].y;
                E[ i + (int_nx.y+1)*ystride ].z = -E[ i + (int_nx.x-1) * ystride ].z;

                B[ i + (int_nx.y)*ystride ].x =  B[ i + (int_nx.y-1) * ystride ].x;
                B[ i + (int_nx.y)*ystride ].y =  0;
                B[ i + (int_nx.y)*ystride ].z =  B[ i + (int_nx.y-1) * ystride ].z;


                B[ i + (int_nx.y+1) * ystride ].x =  B[ i + (int_nx.y-2) * ystride ].x;
                B[ i + (int_nx.y+1) * ystride ].y = -B[ i + (int_nx.y-1) * ystride ].y;
                B[ i + (int_nx.y+1) * ystride ].z =  B[ i + (int_nx.y-2) * ystride ].z;
            }
            break;
        }
    }
}


__host__
/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void EMF::process_bc() {

    dim3 block( 64 );

    // x boundaries
    if ( bc.x.lower > emf::bc::periodic || bc.x.upper > emf::bc::periodic ) {
        dim3 grid( 2, E->ntiles.y );
        _emf_bcx <<< grid, block >>> ( E -> d_buffer, B -> d_buffer, 
            E -> nx, E -> ext_nx(), E -> gc, E -> ntiles, bc );
    }

    // y boundaries
    if ( bc.y.lower > emf::bc::periodic || bc.y.upper > emf::bc::periodic ) {
        dim3 grid( E->ntiles.x, 2 );
        _emf_bcy <<< grid, block >>> ( E -> d_buffer, B -> d_buffer, 
            E -> nx, E -> ext_nx(), E -> gc, E -> ntiles, bc );;
    }

}

/**
 * @brief E advance for Yee algorithm, including current
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param stride    Stride for j coordinate (E,B)
 * @param J         Pointer to 0,0 coordinate of current density
 * @param J_stride  Stride for j coordinate (J)
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 * @param dt        \Delta t
 */
__device__
void yeeJ_e( 
    float3 * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    uint2 const nx, unsigned int const stride, 
    float3 const * const __restrict__ J, unsigned int const stride_J, 
    float const dt_dx, float const dt_dy, float const dt )
{
    int const vol = ( nx.x + 2 ) * ( nx.y + 2 );

    // The y and x loops are fused into a single loop to improve parallelism
    for( int idx = threadIdx.x; idx < vol; idx += blockDim.x ) {
        const int i = idx % ( nx.x + 2 );   // range is 0 to nx+1
        const int j = idx / ( nx.x + 2 );

        E[i+j*stride].x += ( + dt_dy * ( B[i+j*stride].z - B[i+(j-1)*stride].z) ) 
                             - dt * J[i+j*stride_J].x;
        
        E[i+j*stride].y += ( - dt_dx * ( B[i+j*stride].z - B[(i-1)+j*stride].z) )
                             - dt * J[i+j*stride_J].y;

        E[i+j*stride].z += ( + dt_dx * ( B[i+j*stride].y - B[(i-1)+j*stride].y) - 
                               dt_dy * ( B[i+j*stride].x - B[i+(j-1)*stride].x) )
                             - dt * J[i+j*stride_J].z;
    }
}


__global__ 
/**
 * @brief CUDA kernel for advancing EM field 1 timestep, including current
 * 
 * @param d_E       E field grid
 * @param d_B       B field grid
 * @param int_nx    Tile size (internal)
 * @param offset    Offset to position (0,0) on tile
 * @param ext_nx    Tile size (external) i.e including guard cells
 * @param dt_dx     Time step over cell size
 */
void yeeJ_kernel( 
    float3 * const __restrict__ d_E,
    float3 * const __restrict__ d_B,
    uint2 const int_nx, uint2 const ext_nx, unsigned int const offset, 
    float3 * const __restrict__ d_J,
    uint2 const J_ext_nx, unsigned int const J_offset, 
    float2 const dt_dx, float const dt )
{
    auto block = cg::this_thread_block();
    extern __shared__ float3 buffer[];
    
    const float dt_dx2 = dt_dx.x / 2.0f;
    const float dt_dy2 = dt_dx.y / 2.0f;

    const int tid      = blockIdx.y * gridDim.x + blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int tile_off = tid * tile_vol;

    // Copy E and B into shared memory and sync
/*
    for( int i = block.thread_rank(); i < ext_nx.x * ext_nx.y; i += block.num_threads() ) {
        buffer[i        ] = d_E[tile_off + i];
        buffer[tile_vol + i] = d_B[tile_off + i];
    }
*/

    {
        float4 * __restrict__ dstA = (float4 *) & buffer[0];
        float4 * __restrict__ dstB = (float4 *) & buffer[tile_vol];

        float4 * __restrict__ srcA = (float4 *) & d_E[ tile_off ];
        float4 * __restrict__ srcB = (float4 *) & d_B[ tile_off ];

        // tile_vol is always a multiple of 4
        const int size = ( tile_vol * 3 ) / 4;
        for( int i = block.thread_rank(); i < size; i+= block.num_threads() ) {
            dstA[ i ] = srcA[ i ];
            dstB[ i ] = srcB[ i ];
        }
    }

    float3 * const E = buffer + offset;
    float3 * const B = E + tile_vol; 

    const int J_tile_vol = roundup4( J_ext_nx.x * J_ext_nx.y );
    const int J_tile_off = tid * J_tile_vol;
    float3 const * const J = &d_J[ J_tile_off + J_offset ];

    // Perform half B field advance
    block.sync();
    yee_b( E, B , int_nx, ext_nx.x, dt_dx2, dt_dy2 );

    // Perform full E field advance
    block.sync();
    yeeJ_e( E, B, int_nx, ext_nx.x, J, J_ext_nx.x, dt_dx.x, dt_dx.y, dt );

    // Perform half B field advance and sync
    block.sync();
    yee_b( E, B, int_nx, ext_nx.x, dt_dx2, dt_dy2 );
 
    // Copy data to global memory
    block.sync();

/*
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_E[tile_off + i] = buffer[i];
        d_B[tile_off + i] = buffer[B_off + i];
    }
*/

    {
        float4 * __restrict__ dstA = (float4 *) & d_E[ tile_off ];
        float4 * __restrict__ dstB = (float4 *) & d_B[ tile_off ];

        float4 * __restrict__ srcA = (float4 *) & buffer[0];
        float4 * __restrict__ srcB = (float4 *) & buffer[tile_vol];

        // tile_vol is always a multiple of 4
        const int size = ( tile_vol * 3 ) / 4;
        for( int i = block.thread_rank(); i < size; i+= block.num_threads() ) {
            dstA[ i ] = srcA[ i ];
            dstB[ i ] = srcB[ i ];
        }
    }


}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 */
__host__
void EMF::advance( Current & current ) {

    // Tile block size (grid + guard cells)
    uint2 ext_nx = E -> ext_nx();

    float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    dim3 grid( E->ntiles.x, E->ntiles.y );
    dim3 block( 64 );
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    size_t shm_size = 2 * tile_vol * sizeof(float3);

    // Advance EM field using Yee algorithm modified for having E and B time centered
    yeeJ_kernel <<< grid, block, shm_size >>> ( 
        E->d_buffer, B->d_buffer, 
        E->nx, E->ext_nx(), E -> offset(), 
        current.J->d_buffer, current.J->ext_nx(), current.J -> offset(),
        dt_dx, dt
    );

    // Update guard cells with new values
    E -> copy_to_gc( );
    B -> copy_to_gc( );

    // Do additional bc calculations if needed
    process_bc();

    // Advance internal iteration number
    iter += 1;

    // Move simulation window if needed
    move_window();
}



/**
 * @brief Add laser field to simulation
 * 
 * Field is super-imposed (added) on top of existing fields
 * 
 * @param laser     Laser pulse object
 */
void EMF::add_laser( Laser::Pulse & laser ){

    VectorField tmp_E( E -> ntiles, E-> nx, E -> gc );
    VectorField tmp_B( E -> ntiles, B-> nx, B -> gc );

    // Get laser fields
    laser.launch( tmp_E, tmp_B, box );

    // Add laser to simulation
    E -> add( tmp_E );
    B -> add( tmp_B );
}

__host__
/**
 * @brief Save EMF data to diagnostic file
 * 
 * @param field     Field to save (0:E, 1:B)
 * @param fc        Field component to save (0, 1 or 2)
 */
void EMF::save( const emf::field field, fcomp::cart const fc ) {

    char vfname[16];	// Dataset name
    char vflabel[16];	// Dataset label (for plots)

    char comp[] = {'x','y','z'};

    if ( fc < 0 || fc > 2 ) {
        std::cerr << "(*error*) Invalid field component (fc) selected, returning" << std::endl;
        return;
    }

    // Choose field to save
    VectorField * f;
    switch (field) {
        case emf::e :
            f = E;
            snprintf(vfname,16,"E%c",comp[fc]);
            snprintf(vflabel,16,"E_%c",comp[fc]);
            break;
        case emf::b :
            f = B;
            snprintf(vfname,16,"B%1c",comp[fc]);
            snprintf(vflabel,16,"B_%c",comp[fc]);
            break;
        default:
        std::cerr << "(*error*) Invalid field type selected, returning..." << std::endl;
        return;
    }

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "x",
        .min = 0.0 + moving_window.motion(),
        .max = box.x + moving_window.motion(),
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "y",
        .min = 0.0,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_n"
    };

    zdf::grid_info info = {
        .name = vfname,
        .ndims = 2,
        .label = vflabel,
        .units = (char *) "m_e c \\omega_n e^{-1}",
        .axis = axis
    };

    info.count[0] = E -> ntiles.x * E -> nx.x;
    info.count[1] = E -> ntiles.y * E -> nx.y;


    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    f -> save( fc, info, iteration, "EMF" );
}

__global__
void _get_energy_kernel( 
    float3 * const __restrict__ d_E,
    float3 * const __restrict__ d_B,
    uint2 const int_nx, uint2 const ext_nx, unsigned int const offset, 
    double * const __restrict__ d_energy ) {

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const int tid      = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int tile_off = tid * tile_vol + offset;

    // Copy E and B into shared memory and sync
    double3 ene_E = make_double3(0,0,0);
    double3 ene_B = make_double3(0,0,0);

    for( int idx = block.thread_rank(); idx < int_nx.y * int_nx.x; idx += block.num_threads() ) {
        int const i = idx % int_nx.x;
        int const j = idx / int_nx.x;

        float3 const efld = d_E[ tile_off + j * ext_nx.x + i ];
        float3 const bfld = d_B[ tile_off + j * ext_nx.x + i ];

        ene_E.x += efld.x * efld.x;
        ene_E.y += efld.y * efld.y;
        ene_E.z += efld.z * efld.z;

        ene_B.x += bfld.x * bfld.x;
        ene_B.y += bfld.y * bfld.y;
        ene_B.z += bfld.z * bfld.z;
    }

    // Add up energy from all warps
    ene_E.x = cg::reduce( warp, ene_E.x, cg::plus<double>());
    ene_E.y = cg::reduce( warp, ene_E.y, cg::plus<double>());
    ene_E.z = cg::reduce( warp, ene_E.z, cg::plus<double>());

    ene_B.x = cg::reduce( warp, ene_B.x, cg::plus<double>());
    ene_B.y = cg::reduce( warp, ene_B.y, cg::plus<double>());
    ene_B.z = cg::reduce( warp, ene_B.z, cg::plus<double>());

    if ( warp.thread_rank() == 0 ) {
        atomicAdd( &(d_energy[0]), ene_E.x );
        atomicAdd( &(d_energy[1]), ene_E.y );
        atomicAdd( &(d_energy[2]), ene_E.z );

        atomicAdd( &(d_energy[3]), ene_B.x );
        atomicAdd( &(d_energy[4]), ene_B.y );
        atomicAdd( &(d_energy[5]), ene_B.z );
    }
}


/**
 * @brief Get total field energy per field component
 * 
 * @param energy    Array that will hold energy values
 */
__host__
void EMF::get_energy( double3 & ene_E, double3 & ene_B ) {

    // Zero energy values
    device::zero( d_energy, 6 );

    dim3 grid( E->ntiles.x, E->ntiles.y );
    dim3 block( 1024 );
    _get_energy_kernel <<< grid, block >>> ( 
        E->d_buffer, B->d_buffer,
        E->nx, E->ext_nx(), E->offset(),
        d_energy );
    
    double energy[6];
    devhost_memcpy( energy, d_energy, 6 );
    for( int i = 0; i < 6; i++ ) {
        energy[i] *= 0.5 * dx.x * dx.y;
    }

    ene_E.x = energy[0];
    ene_E.y = energy[1];
    ene_E.z = energy[2];

    ene_B.x = energy[3];
    ene_B.y = energy[4];
    ene_B.z = energy[5];
}