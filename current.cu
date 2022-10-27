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
    // These are required for the Yee solver AND for current deposition
    bnd<unsigned int> gc;
    gc.x = {1,2};
    gc.y = {1,2};

    J = new VectorField( ntiles, nx, gc );

    // Zero initial current
    // This is only relevant for diagnostics, current should always zeroed before deposition
    J -> zero();

    // Set default boundary conditions to periodic
    bc = current::bc_type (current::bc::periodic);

    // Disable filtering by default
    filter = new Filter::None();

    // Reset iteration number
    iter = 0;

    std::cout << "(*info*) Current object initialized." << std::endl;
}

__global__
void _current_bcx(
    float3 * const __restrict__ d_J,
    uint2 const int_nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    uint2 const ntiles, current::bc_type bc )
{
    const int tid = blockIdx.y * ntiles.x + blockIdx.x * (ntiles.x - 1);

    const int tile_off = tid * ext_nx.x * ext_nx.y;
    const int ystride = ext_nx.x;
    const int offset   = gc.x.lower;

    float3 * const __restrict__ J = d_J + tile_off + offset;

    if ( blockIdx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( current::bc::reflecting ):
            for( int idx = threadIdx.x; idx < ext_nx.y; idx += blockDim.x ) {
                // j includes the y-stride
                const int j = idx * ystride;

                float jx0 = -J[ -1 + j ].x + J[ 0 + j ].x; 
                float jy1 =  J[ -1 + j ].y + J[ 1 + j ].y;
                float jz1 =  J[ -1 + j ].z + J[ 1 + j ].z;

                J[ -1 + j ].x = J[ 0 + j ].x = jx0;
                J[ -1 + j ].y = J[ 1 + j ].y = jy1;
                J[ -1 + j ].z = J[ 1 + j ].z = jz1;
            }
            break;
        }
    } else {
        // Upper boundary
        switch( bc.x.upper ) {
        case( current::bc::reflecting ):
            for( int idx = threadIdx.x; idx < ext_nx.y; idx += blockDim.x ) {
                int j = idx * ystride;

                float jx0 =  J[ int_nx.x-1 + j ].x - J[ int_nx.x + 0 + j ].x; 
                float jy1 =  J[ int_nx.x-1 + j ].y + J[ int_nx.x + 1 + j ].y;
                float jz1 =  J[ int_nx.x-1 + j ].z + J[ int_nx.x + 1 + j ].z;

                J[ int_nx.x-1 + j ].x = J[ int_nx.x + 0 + j ].x = jx0;
                J[ int_nx.x-1 + j ].y = J[ int_nx.x + 1 + j ].y = jy1;
                J[ int_nx.x-1 + j ].z = J[ int_nx.x + 1 + j ].z = jz1;
            }
            break;
        }
    }
}


__global__
void _current_bcy(
    float3 * const __restrict__ d_J,
    uint2 const int_nx, uint2 const ext_nx, bnd<unsigned int> gc, 
    uint2 const ntiles, current::bc_type bc )
{
    const int tid = blockIdx.y * (ntiles.y - 1) * ntiles.x + blockIdx.x;

    const int tile_off = tid * ext_nx.x * ext_nx.y;
    const int ystride = ext_nx.x;
    const int offset   = gc.y.lower * ystride;

    float3 * const __restrict__ J = d_J + tile_off + offset;

    if ( blockIdx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( current::bc::reflecting ):
            for( int idx = threadIdx.x; idx < ext_nx.x; idx += blockDim.x ) {
                int i = idx;

                float jx1 =  J[ i - ystride ].x + J[ i + ystride ].x; 
                float jy0 = -J[ i - ystride ].y + J[ i +       0 ].y;
                float jz1 =  J[ i - ystride ].z + J[ i + ystride ].z;

                J[ i - ystride ].x = J[ i + ystride ].x = jx1;
                J[ i - ystride ].y = J[ i +       0 ].y = jy0;
                J[ i - ystride ].z = J[ i + ystride ].z = jz1;
            }
            break;
        }
    } else {
        // Upper boundary
        switch( bc.y.upper ) {
        case( current::bc::reflecting ):
            for( int idx = threadIdx.x; idx < ext_nx.x; idx += blockDim.x ) {
                int i = idx;

                float jx1 =  J[ i + (int_nx.y-1)*ystride ].x + J[ i + (int_nx.y + 1)*ystride ].x; 
                float jy0 =  J[ i + (int_nx.y-1)*ystride ].y - J[ i + (int_nx.y + 0)*ystride ].y;
                float jz1 =  J[ i + (int_nx.y-1)*ystride ].z + J[ i + (int_nx.y + 1)*ystride ].z;

                J[ i + (int_nx.y-1)*ystride ].x = J[ i + (int_nx.y + 1)*ystride ].x = jx1;
                J[ i + (int_nx.y-1)*ystride ].y = J[ i + (int_nx.y + 0)*ystride ].y = jy0;
                J[ i + (int_nx.y-1)*ystride ].z = J[ i + (int_nx.y + 1)*ystride ].z = jz1;
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
void Current::process_bc() {

    dim3 block( 64 );

    // x boundaries
    if ( bc.x.lower > current::bc::periodic || bc.x.upper > current::bc::periodic ) {
        dim3 grid( 2, J->ntiles.y );
        _current_bcx <<< grid, block >>> ( J -> d_buffer, J -> nx, J -> ext_nx(), J -> gc, J -> ntiles, bc );
    }

    // y boundaries
    if ( bc.y.lower > current::bc::periodic || bc.y.upper > current::bc::periodic ) {
        dim3 grid( J->ntiles.x, 2 );
        _current_bcy <<< grid, block >>> ( J -> d_buffer, J -> nx, J -> ext_nx(), J -> gc, J -> ntiles, bc );;
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
    J -> add_from_gc( );
    J -> copy_to_gc( );

    // Do additional bc calculations if needed
    process_bc();

    // Apply filtering
    filter -> apply( *J );

    // Advance iteration count
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