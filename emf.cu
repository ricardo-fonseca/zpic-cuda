#include "emf.cuh"
#include <iostream>
#include "tile_zdf.cuh"

/**
 * @brief Construct a new EMF::EMF object
 * 
 * @param gnx   Global grid size
 * @param tnx   Tile grid size
 * @param box_  Simulation box dimensions
 * @param dt_   Time step size
 */
__host__
EMF::EMF( const int2 gnx, const int2 tnx, const float2 box_, const float dt_ ) {

    std::cout << "(*info*) Initialize emf..." << std::endl;

    // Set box limits, cells sizes and time step
    box = box_;
    dt = dt_;
    dx.x = box.x / gnx.x;
    dx.y = box.y / gnx.y;

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    int2 gc[2] = {{1,1},{2,2}}; 

    E = new VFLD( gnx, tnx, gc );
    B = new VFLD( gnx, tnx, gc );

    std::cout << "(*info*) Zeroing fields..." << std::endl;

    // Zero fields
    E -> zero();
    B -> zero();

    // Reset iteration number
    d_iter = h_iter = 0;

    std::cout << "(*info*) Initialize emf done!" << std::endl;
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
    
    d_iter = h_iter = -1;
}

/**
 * @brief Updates host/device data from device/host data
 * 
 * It will also set the iteration numbers accordingly
 * 
 */
void EMF::update_data( const VFLD::copy_direction direction ) {

    E -> update_data( direction );
    B -> update_data( direction );

    switch( direction ) {
    case VFLD::host_device:  // Host to device
        d_iter = h_iter;
       break;
    case VFLD::device_host: // Device to host
        h_iter = d_iter;
        break;
    default:
        std::cerr << "(*error*) Invalid direction in EMF::update() call." << std::endl;
    }
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
void yee_b( float3 * E, float3 * B, int2 nx, 
    const int stride, const float dt_dx, const float dt_dy ) {

    const int vol = ( nx.x + 2 ) * ( nx.y + 2 );

    // The y and x loops are fused into a single loop to improve parallelism
    for( int idx = threadIdx.x; idx < vol; idx += blockDim.x ) {
        const int i = -1 + idx % ( nx.x + 2 );  // range is -1 to nx
        const int j = -1 + idx / ( nx.x + 2 );
        
        B[ i + j*stride ].x += ( - dt_dy * ( E[i+(j+1)*stride].z - E[i+j*stride].z ) );  
        B[ i + j*stride ].y += (   dt_dx * ( E[(i+1)+j*stride].z - E[i+j*stride].z ) );  
        B[ i + j*stride ].z += ( - dt_dx * ( E[(i+1)+j*stride].y - E[i+j*stride].y ) + 
                                   dt_dy * ( E[i+(j+1)*stride].x - E[i+j*stride].x ) );  
    }
}


/**
 * @brief E advance for Yee algorithm
 * 
 * @param E         Pointer to 0,0 coordinate of E field
 * @param B         Pointer to 0,0 coordinate of B field
 * @param nx        Internal tile size
 * @param stride    Stride for j coordinate
 * @param dt_dx     \Delta t / \Delta x
 * @param dt_dy     \Delta t / \Delta y
 */
__device__
void yee_e( float3 * E, float3 * B, int2 nx, 
    const int stride, const float dt_dx, const float dt_dy ) {

    const int vol = ( nx.x + 2 ) * ( nx.y + 2 );

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


__global__ void yee_kernel( float3 * d_E, float3 * d_B, int2 int_nx, int offset, 
    int2 ext_nx, float2 dt_dx ) {

    extern __shared__ float3 buffer[];
    
    const float dt_dx2 = dt_dx.x / 2.0f;
    const float dt_dy2 = dt_dx.y / 2.0f;

    const int tile_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * ext_nx.x * ext_nx.y;
    const int B_off = ext_nx.x * ext_nx.y;

    // Copy E and B into shared memory and sync
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        buffer[i        ] = d_E[tile_off + i];
        buffer[B_off + i] = d_B[tile_off + i];
    }

    float3* E = buffer + offset;
    float3* B = E + B_off; 

    __syncthreads();

    // Perform half B field advance and sync
    yee_b( E, B , int_nx, ext_nx.x, dt_dx2, dt_dy2 );
    __syncthreads();

    // Perform full E field advance and sync
    yee_e( E, B, int_nx, ext_nx.x, dt_dx.x, dt_dx.y );
    __syncthreads();

    // Perform half B field advance and sync
    yee_b( E, B, int_nx, ext_nx.x, dt_dx2, dt_dy2 );
    __syncthreads();

    // Copy data to global memory
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_E[tile_off + i] = buffer[i];
        d_B[tile_off + i] = buffer[B_off + i];
    }
}


/**
 * @brief Advance EM fields 1 time step
 * 
 */
__host__
void EMF::advance() {

    std::cout << "(*info*) Advance emf (" << d_iter << ")" << std::endl;

    // Tile block size (grid + guard cells)
    int2 ext_nx = E -> ext_nx();

    float2 dt_dx = {
        .x = dt / dx.x,
        .y = dt / dx.y
    };


    dim3 grid( E->nxtiles.x, E->nxtiles.y );
    dim3 block( 64 );
    size_t shm_size = 2 * ext_nx.x * ext_nx.y * sizeof(float3);

    // Advance EM field using Yee algorithm modified for having E and B time centered
    yee_kernel <<< grid, block, shm_size >>> ( 
        E->d_buffer, B->d_buffer, 
        E->nx, E->offset(), E->ext_nx(), dt_dx
    );

    // Update guard cells with new values
    E -> update_gc();
    B -> update_gc();

    // Advance internal iteration number
    d_iter += 1;
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

    VFLD tmp_E( E -> g_nx(), E-> nx, E -> gc );
    VFLD tmp_B( E -> g_nx(), B-> nx, B -> gc );

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
void EMF::report( const diag_fld field, const int fc ) {

    char vfname[16];	// Dataset name
    char vflabel[16];	// Dataset label (for plots)

    char comp[] = {'x','y','z'};

    if ( fc < 0 || fc > 2 ) {
        std::cerr << "(*error*) Invalid field component (fc) selected, returning" << std::endl;
        return;
    }

    // Choose field to save
    VFLD * f;
    switch (field) {
        case EFLD:
            f = E;
            snprintf(vfname,16,"E%c",comp[fc]);
            snprintf(vflabel,16,"E_%c",comp[fc]);
            break;
        case BFLD:
            f = B;
            snprintf(vfname,16,"B%1c",comp[fc]);
            snprintf(vflabel,16,"B_%c",comp[fc]);
            break;
        default:
        std::cerr << "(*error*) Invalid field type selected, returning..." << std::endl;
        return;
    }

    t_zdf_grid_axis axis[2];
    axis[0] = (t_zdf_grid_axis) {
    	.name = (char *) "x",
    	.min = 0.0,
    	.max = box.x,
    	.label = (char *) "x",
    	.units = (char *) "c/\\omega_n"
    };

    axis[1] = (t_zdf_grid_axis) {
        .name = (char *) "y",
    	.min = 0.0,
    	.max = box.y,
    	.label = (char *) "y",
    	.units = (char *) "c/\\omega_n"
    };

    t_zdf_grid_info info = {
        .name = vfname,
    	.ndims = 2,
    	.label = vflabel,
    	.units = (char *) "m_e c \\omega_p e^{-1}",
    	.axis = axis
    };

    info.count[0] = E -> nxtiles.x * E -> nx.x;
    info.count[1] = E -> nxtiles.y * E -> nx.y;


    t_zdf_iteration iter = {
    	.n = d_iter,
    	.t = d_iter * dt,
    	.time_units = (char *) "1/\\omega_p"
    };

    zdf_save_tile_vfld( *f, fc, &info, &iter, "EMF" );
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