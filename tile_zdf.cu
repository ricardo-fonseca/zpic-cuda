#include "tile_zdf.cuh"
#include <iostream>

/**
 * @brief Saves a tile vfld into a zdf file. Data is gathered into a standard grid first discarding guard cells.
 * @param   vfld        Tile vfld object
 * @param   fc          Field component to save
 * @param   info        Grid metadata (label, units, axis, etc.). Information is used to set file name
 * @param   iteration   Iteration metadata
 * @param   path        Path where to save the file
 **/
__host__
int zdf_save_tile_vfld( VFLD &vfld, const int fc, const t_zdf_grid_info *info, 
	const t_zdf_iteration *iteration, char const path[] ) {

    const size_t size = ( vfld.nxtiles.x * vfld.nx.x ) *
                        ( vfld.nxtiles.y * vfld.nx.y ) *
                        sizeof( float );

    float *data;
    cudaError_t err;
    err = cudaMallocHost( &data, size );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate data on cpu for zdf_save_tile_vfld()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    vfld.gather( fc, data );

    zdf_save_grid( data, zdf_float32, info, iteration, path );

    err = cudaFreeHost( data );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free data on cpu for zdf_save_tile_vfld()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}

__host__
int zdf_save_grid( float *buffer,  t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    return zdf_save_grid( (void*) buffer, zdf_float32, &info, &iter, path.c_str() );
}

__host__
int zdf_save_grid( double *buffer, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    return zdf_save_grid( (void*) buffer, zdf_float64, &info, &iter, path.c_str() );
}
