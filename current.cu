#include "current.cuh"

#include <iostream>


__host__
/**
 * @brief Construct a new Current:: Current object
 * 
 * @param gnx   Global grid size
 * @param tnx   Tile grid size
 * @param box_  Simulation box dimensions
 * @param dt_   Time step size
 */

/**
 * @brief Construct a new Current:: Current object
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 * @param box       Box size
 * @param dt        Time step
 */
Current::Current( uint2 const ntiles, uint2 const nx, float2 const box,
    float const dt ) : box{box}, dt{dt}
{

    dx.x = box.x / ( nx.x * ntiles.x );
    dx.y = box.y / ( nx.y * ntiles.y );

    // Guard cells (1 below, 2 above)
    // These are required for the Yee solver AND for field interpolation
    uint2 gc[2] = { make_uint2(1,1), make_uint2(2,2)}; 

    J = new VectorField( ntiles, nx, gc );

    // Zero initial current
    // This is only relevant for diagnostics, current should always zeroed before deposition
    J -> zero();

    // Reset iteration number
    iter = 0;

    std::cout << "(*info*) Current object initialized." << std::endl;
}

__host__
/**
 * @brief Destroy the Current:: Current object
 * 
 */
Current::~Current()
{   
    delete (J);
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
    J -> add_from_gc( );
    J -> copy_to_gc( );

    // Apply filtering
    // filter -> apply( *J );

    iter++;

    // I'm not sure if this should be before or after `iter++`
    // Note that it only affects the axis range on output data
    if ( moving_window.needs_move( iter * dt ) )
        moving_window.advance();
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
void Current::save( fcomp::cart const jc ) {

    char vfname[16];	// Dataset name
    char vflabel[16];	// Dataset label (for plots)

    char comp[] = {'x','y','z'};

    if ( jc < 0 || jc > 2 ) {
        std::cerr << "(*error*) Invalid current component (jc) selected, returning" << std::endl;
        return;
    }

    snprintf(vfname,16,"J%c",comp[jc]);
    snprintf(vflabel,16,"J_%c",comp[jc]);

    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
    	.name = (char *) "x",
    	.min = 0.0 + moving_window.motion(),
    	.max = box.x,
    	.label = (char *) "x",
    	.units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "y",
    	.min = 0.0 + moving_window.motion(),
    	.max = box.y,
    	.label = (char *) "y",
    	.units = (char *) "c/\\omega_n"
    };

    zdf::grid_info info = {
        .name = vfname,
    	.ndims = 2,
    	.label = vflabel,
    	.units = (char *) "e \\omega_n^2 / c",
    	.axis = axis
    };

    info.count[0] = J -> ntiles.x * J -> nx.x;
    info.count[1] = J -> ntiles.y * J -> nx.y;

    zdf::iteration iteration = {
    	.n = iter,
    	.t = iter * dt,
    	.time_units = (char *) "1/\\omega_n"
    };

    J -> save( jc, info, iteration, "CURRENT" );
}