#ifndef __TILE_ZDF__
#define __TILE_ZDF__

#include "tile_vfld.cuh"
#include <string>
#include "zdf.h"

__host__
int zdf_save_tile_vfld( VFLD &vfld, const int fc, const t_zdf_grid_info *info, 
	const t_zdf_iteration *iteration, char const path[] );

__host__
int zdf_save_grid( float *buffer,  t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path );

__host__
int zdf_save_grid( double *buffer, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path );

#endif