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

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    uint2 gc[2] = { make_uint2(1,1), make_uint2(2,2) }; 

    E = new VectorField( ntiles, nx, gc );
    B = new VectorField( ntiles, nx, gc );

    // Zero fields
    E -> zero();
    B -> zero();

    // Reset iteration number
    iter = 0;

    std::cout << "(*info*) EMF object initialized" << std::endl;
}

/**
 * @brief Destroy the EMF::EMF object
 * 
 */
__host__
EMF::~EMF(){

    std::cout << "(*info*) Cleanup emf..." << std::endl;

    delete (E);
    delete (B);
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

    const int tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * ext_nx.x * ext_nx.y;
    const int B_off = ext_nx.x * ext_nx.y;

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
    size_t shm_size = 2 * ext_nx.x * ext_nx.y * sizeof(float3);

    // Advance EM field using Yee algorithm modified for having E and B time centered
    yee_kernel <<< grid, block, shm_size >>> ( 
        E->d_buffer, B->d_buffer, 
        E->nx, E->ext_nx(), E->offset(), dt_dx
    );

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Advance internal iteration number
    iter += 1;
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
    unsigned int const vol = ( nx.x + 2 ) * ( nx.y + 2 );

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

    const int tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * ext_nx.x * ext_nx.y;
    const int B_off = ext_nx.x * ext_nx.y;

    // Copy E and B into shared memory and sync
    for( int i = block.thread_rank(); i < ext_nx.x * ext_nx.y; i += block.num_threads() ) {
        buffer[i        ] = d_E[tile_off + i];
        buffer[B_off + i] = d_B[tile_off + i];
    }

    float3 * const E = buffer + offset;
    float3 * const B = E + B_off; 

    const int J_tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * J_ext_nx.x * J_ext_nx.y;
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
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_E[tile_off + i] = buffer[i];
        d_B[tile_off + i] = buffer[B_off + i];
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
    size_t shm_size = 2 * ext_nx.x * ext_nx.y * sizeof(float3);

    // Advance EM field using Yee algorithm modified for having E and B time centered
    yeeJ_kernel <<< grid, block, shm_size >>> ( 
        E->d_buffer, B->d_buffer, 
        E->nx, E->ext_nx(), E -> offset(), 
        current.J->d_buffer, current.J->ext_nx(), current.J -> offset(),
        dt_dx, dt
    );

    // Update guard cells with new values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Add laser field to simulation
 * 
 * Field is super-imposed (added) on top of existing fields
 * 
 * @param laser     Laser pulse object
 */
void EMF::add_laser( Laser& laser ){

    std::cout << "(*info*) Adding laser..." << std::endl;

    VectorField tmp_E( E -> g_nx(), E-> nx, E -> gc );
    VectorField tmp_B( E -> g_nx(), B-> nx, B -> gc );

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
    	.min = 0.0,
    	.max = box.x,
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
    	.units = (char *) "m_e c \\omega_p e^{-1}",
    	.axis = axis
    };

    info.count[0] = E -> ntiles.x * E -> nx.x;
    info.count[1] = E -> ntiles.y * E -> nx.y;


    zdf::iteration iteration = {
    	.n = iter,
    	.t = iter * dt,
    	.time_units = (char *) "1/\\omega_p"
    };

    f -> save( fc, info, iteration, "EMF" );
}

/**
 * @brief Get total field energy per field component
 * 
 * @param energy    Array that will hold energy values
 */
__host__
void get_energy( double energy[6] ) {
    std::cout << "(*warn*) EMF::get_energy() not implemented yet." << std::endl;
    for( int i = 0; i < 6; i++) energy[i] = 0;
}