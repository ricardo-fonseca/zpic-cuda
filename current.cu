#include "current.cuh"

#include <iostream>
#include "tile_zdf.cuh"

__host__
/**
 * @brief Construct a new Current:: Current object
 * 
 * @param gnx   Global grid size
 * @param tnx   Tile grid size
 * @param box_  Simulation box dimensions
 * @param dt_   Time step size
 */
Current::Current(const int2 gnx, const int2 tnx, const float2 box, const float dt ) :
    box{box}, dt{dt} {

    std::cout << "(*info*) Initialize current..." << std::endl;

    dx.x = box.x / gnx.x;
    dx.y = box.y / gnx.y;

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    int2 gc[2] = {{1,1},{2,2}}; 

    J = new VFLD( gnx, tnx, gc );

    // Zero initial current
    // This is only relevant for diagnostics, current is always zeroed before deposition
    std::cout << "(*info*) Zeroing current..." << std::endl;
    J -> zero();

    // Reset iteration number
    d_iter = h_iter = 0;

    std::cout << "(*info*) Initialize current done!" << std::endl;
}

__host__
/**
 * @brief Destroy the Current:: Current object
 * 
 */
Current::~Current() {
    
    std::cout << "(*info*) Cleanup current..." << std::endl;

    delete (J);

    d_iter = h_iter = -1;
}

__host__
/**
 * @brief Updates host/device data from device/host data
 * 
 * It will also set the iteration numbers accordingly
 * 
 * @param direction     host_device or device_host
 */
void Current::update_data( const VFLD::copy_direction direction ) {

    J -> update_data( direction );

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

__host__
/**
 * @brief Advance electric current to next iteration
 * 
 * Adds up current deposited on guard cells and (optionally) applies digital filtering
 * 
 */
void Current::advance() {

    // Add up current deposited on guard cells
    //J -> update_add_gc();

    // Apply filtering
    // filter -> apply( *J );

    d_iter++;
}

__host__
/**
 * @brief Zero electric current values
 * 
 */
void Current::zero() {
    J -> zero();
}

__host__
/**
 * @brief Save electric current data to diagnostic file
 * 
 * @param jc        Current component to save (0, 1 or 2)
 */
void Current::report( const int jc ) {

        char vfname[16];	// Dataset name
    char vflabel[16];	// Dataset label (for plots)

    char comp[] = {'x','y','z'};

    if ( jc < 0 || jc > 2 ) {
        std::cerr << "(*error*) Invalid current component (jc) selected, returning" << std::endl;
        return;
    }

    snprintf(vfname,16,"J%c",comp[jc]);
    snprintf(vflabel,16,"J_%c",comp[jc]);

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
    	.units = (char *) "e \\omega_p^2 / c",
    	.axis = axis
    };

    info.count[0] = J -> nxtiles.x * J -> nx.x;
    info.count[1] = J -> nxtiles.y * J -> nx.y;

    t_zdf_iteration iter = {
    	.n = d_iter,
    	.t = d_iter * dt,
    	.time_units = (char *) "1/\\omega_p"
    };

    J -> save( jc, info, iter, "CURRENT" );
}