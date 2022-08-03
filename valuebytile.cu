#include "tile_vfld.cuh"

/**
 * @brief CUDA kernel for valuebytileSets x value in each tile to the tile id
 * 
 * Must be called with one block per tile i.e. with grid equal to nxtiles
 * 
 * @param data 
 * @param size 
 * @param int_nx 
 * @param ext_nx 
 */
__global__
void valuebytile_kernel( float3 * data, int2 int_nx, int2 ext_nx ) {

    int    tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    size_t tile_off = tile_id * ext_nx.x * ext_nx.y;

    for( int i = threadIdx.x; i < int_nx.x * int_nx.y; i+= blockDim.x ) {
        int const ix = i % int_nx.x;
        int const iy = i / int_nx.x;

        size_t const idx = tile_off + iy * ext_nx.x + ix;

        data[ idx ].x = tile_id;
    }
}

/**
 * @brief Sets x value in each tile to the tile id
 * 
 * @param data      VFLD object
 */
void valuebytile( VFLD &data ) {

    int2 int_nx = data.nx;
    int2 ext_nx = data.ext_nx();
    int offset = data.offset();

    dim3 grid( data.nxtiles.x, data.nxtiles.y );
    dim3 block( 64 );

    valuebytile_kernel <<< grid, block >>> ( 
        data.d_buffer + offset,
        int_nx, ext_nx
    );
}