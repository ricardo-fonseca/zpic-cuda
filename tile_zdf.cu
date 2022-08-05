#include "tile_zdf.cuh"
#include <iostream>

__host__
int zdf_save_grid( float *buffer,  t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    return zdf_save_grid( (void*) buffer, zdf_float32, &info, &iter, path.c_str() );
}

__host__
int zdf_save_grid( double *buffer, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    return zdf_save_grid( (void*) buffer, zdf_float64, &info, &iter, path.c_str() );
}
