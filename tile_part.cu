#include "tile_part.cuh"
#include <iostream>
#include <string>

#define CHECK_ERR( err_, msg_ ) { \
    if ( err_ != cudaSuccess ) { \
        std::cerr << "(*error*) " << msg_ << std::endl; \
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

/*
__host__
void device_malloc( auto &buf, size_t size, char *name = nullptr ) {
    cudaError_t err = cudaMalloc( &buf, size );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate " << size << " bytes of device memory"
        if ( name ) std::cerr << " for " << name;
        std::cerr << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__host__
void host_malloc( auto &buf, size_t size, char *name = nullptr ) {
    cudaError_t err = cudaMallocHost( &buf, size );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate " << size << " bytes of host memory"
        if ( name ) std::cerr << " for " << name;
        std::cerr << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
*/


__host__
/**
 * @brief Construct a new Tile Part:: Tile Part object
 * 
 * @param gnx 
 * @param tnx 
 * @param tnp_max 
 */
TilePart::TilePart(const int2 global_nx, const int2 tile_nx, const int tile_np_max ){

    // Validate grid and tile sizes
    if (( global_nx.x <= 0 ) || ( global_nx.y <= 0)) {
        std::cerr << "(*error*) Invalid number of cells gnx: ";
        std::cerr << global_nx.x << "," << global_nx.y << std::endl;
        exit(1);
    }

    if (( tile_nx.x <= 0 ) || ( tile_nx.y <= 0)) {
        std::cerr << "(*error*) Invalid tile size tnx: ";
        std::cerr << tile_nx.x << "," << tile_nx.y << std::endl;
        exit(1);
    }

    if ( global_nx.x % tile_nx.x ) {
        std::cerr << "(*error*) global x grid size, " << global_nx.x;
        std::cerr << "is not a mutliple of x tile size, " << tile_nx.x << "endl";
        exit(1);
    }

    if ( global_nx.y % tile_nx.y ) {
        std::cerr << "(*error*) global y grid size, " << global_nx.y;
        std::cerr << "is not a mutliple of y tile size, " << tile_nx.y << "endl";
        exit(1);
    }

    if ( tile_np_max <= 0 ) {
        std::cerr << "(*error*) Invalid max. number of particles per tile, " << tile_np_max;
        exit(1);
    }

    // Get number of tiles in each direction
    nxtiles.x = global_nx.x / tile_nx.x;
    nxtiles.y = global_nx.y / tile_nx.y;

    // Store tile size
    nx = tile_nx;

    // Get global size of particle buffers
    buffer_max = nxtiles.x * nxtiles.y * tile_np_max;

    // Allocate buffers
    cudaError_t err;

    err = cudaMalloc( &d_ix, buffer_max * sizeof(int2) );
    CHECK_ERR( err, "Failed to allocate device memory for d_ix" );

    err = cudaMalloc( &d_x,  buffer_max * sizeof(float2) );
    CHECK_ERR( err, "Failed to allocate device memory for d_x" );

    err = cudaMalloc( &d_u,  buffer_max * sizeof(float3) );
    CHECK_ERR( err, "Failed to allocate device memory for d_u" );

    err = cudaMallocHost( &h_ix, buffer_max * sizeof(int2) );
    CHECK_ERR( err, "Failed to allocate host memory for h_ix" );

    err = cudaMallocHost( &h_x,  buffer_max * sizeof(float2) );
    CHECK_ERR( err, "Failed to allocate host memory for h_x" );

    err = cudaMallocHost( &h_u,  buffer_max * sizeof(float3) );
    CHECK_ERR( err, "Failed to allocate host memory for h_u" );

    // Allocate array for buffer positions
    err = cudaMalloc( &d_tile, nxtiles.x * nxtiles.y * sizeof(int2) );
    CHECK_ERR( err, "Failed to allocate device memory for d_tile" );

    err = cudaMallocHost( &h_tile, nxtiles.x * nxtiles.y * sizeof(int2) );
    CHECK_ERR( err, "Failed to allocate host memory for h_ix" );

    // Initialize values
    for( int i = 0; i < nxtiles.x * nxtiles.y; i++ ) {
        // Position of first particle in global buffer 
        h_tile[i].x = i * tile_np_max;
        // Number of particles in tile
        h_tile[i].y = 0;
    }

    // Copy to device
    err = cudaMemcpy( d_tile, h_tile, nxtiles.x * nxtiles.y * sizeof(int2), 
            cudaMemcpyHostToDevice );
    CHECK_ERR( err, "Failed to copy h_tile to device" );

}

__host__
/**
 * @brief Destroy the Tile Part:: Tile Part object
 * 
 */
TilePart::~TilePart(){

    cudaError_t err;

    err = cudaFree( d_ix );
    CHECK_ERR( err, "Failed to free device memory for d_ix");

    err = cudaFree( d_x );
    CHECK_ERR( err, "Failed to free device memory for d_x");

    err = cudaFree( d_u );
    CHECK_ERR( err, "Failed to free device memory for d_u");

    err = cudaFree( d_tile );
    CHECK_ERR( err, "Failed to free device memory for d_tile");

    err = cudaFreeHost( h_ix );
    CHECK_ERR( err, "Failed to free host memory for h_ix");

    err = cudaFreeHost( h_x );
    CHECK_ERR( err, "Failed to free host memory for h_x");

    err = cudaFreeHost( h_u );
    CHECK_ERR( err, "Failed to free host memory for h_u");

    err = cudaFreeHost( h_tile );
    CHECK_ERR( err, "Failed to free host memory for h_tile");

}

__host__
/**
 * @brief Returns total number of particles in all tiles
 * 
 * This version adds up the total number of particles in the CPU, a new version
 * doing this on the GPU should be done instead
 * 
 * @return size_t 
 */
size_t TilePart::device_np() {

    std::cerr << "(*warn*) " << __func__ << "() is doing calculations on host." << std::endl;

    // Update host copy of tile info
    size_t size = nxtiles.x * nxtiles.y * sizeof(int2);

    cudaError_t err = cudaMemcpy( h_tile, d_tile, size, cudaMemcpyDeviceToHost );
    CHECK_ERR( err, "Unable to copy tile information to host");

    // Sum up number of particles in all tiles
    size_t np = 0;
    for( size_t i = 0; i < size; i++ ) {
        np += h_tile[i].y;
    }

    return np;
}


/**
 * @brief Gather x quantity data
 * 
 * @param d_ix 
 * @param d_x 
 * @param d_tile 
 * @param d_idx 
 * @param d_data 
 */
__global__
void _gather_x_kernel( int2* __restrict__ d_ix, float2* __restrict__ d_x, int2* __restrict__ d_tile, int tnx,
    int* __restrict__ d_idx, float* __restrict__ d_data ) {
    
    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    
    const int np = d_tile[ tid ].y;
    const int out_offset = d_idx[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = blockIdx.x * tnx + (ix[idx].x + x[idx].x);
    }
};


__global__
void _gather_y_kernel( int2* __restrict__ d_ix, float2* __restrict__ d_x, int2* __restrict__ d_tile, int tny,
    int* __restrict__ d_idx, float* __restrict__ d_data ) {
    
    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    
    const int np = d_tile[ tid ].y;
    const int out_offset = d_idx[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = blockIdx.y * tny + (ix[idx].y + x[idx].y);
    }
};

__global__
void _gather_ux_kernel( float3* __restrict__ d_u, int2* __restrict__ d_tile, 
    int* __restrict__ d_idx, float* __restrict__ d_data ) {
    
    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    float3 __restrict__ *u = &d_u[ offset ];
    
    const int np = d_tile[ tid ].y;
    const int out_offset = d_idx[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = u[idx].x;
    }
};

__global__
void _gather_uy_kernel( float3* __restrict__ d_u, int2* __restrict__ d_tile, 
    int* __restrict__ d_idx, float* __restrict__ d_data ) {
    
    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    float3 __restrict__ *u = &d_u[ offset ];
    
    const int np = d_tile[ tid ].y;
    const int out_offset = d_idx[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = u[idx].y;
    }
};

__global__
void _gather_uz_kernel( float3* __restrict__ d_u, int2* __restrict__ d_tile, 
    int* __restrict__ d_idx, float* __restrict__ d_data ) {
    
    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset =  d_tile[ tid ].x;
    float3 __restrict__ *u = &d_u[ offset ];
    
    const int np = d_tile[ tid ].y;
    const int out_offset = d_idx[ tid ];

    for( int idx = threadIdx.x; idx < np; idx += blockDim.x ) {
        d_data[ out_offset + idx ] = u[idx].z;
    }
};

__host__
/**
 * @brief Gather data from a specific particle quantity in a device buffer
 * 
 * @param quant 
 * @param data 
 * @param np_max 
 */
void TilePart::gather( rep_quant quant, float * __restrict__ h_data, const int np_max ) {

    float *d_data;
    int *d_idx, *h_idx;

    cudaError_t err;

    // Get tile information
    size_t size = nxtiles.x * nxtiles.y * sizeof(int2);
    err = cudaMemcpy( h_tile, d_tile, size, cudaMemcpyDeviceToHost );
    CHECK_ERR( err, "Unable to copy tile information to host");

    // Find positions in output buffer
    err = cudaMallocHost( &h_idx, nxtiles.x * nxtiles.y * sizeof(int) );
    CHECK_ERR( err, "Unable to allocate d_idx for gather");

    size_t np = h_tile[0].y;
    h_idx[0] = 0;
    for( int i = 1; i < nxtiles.x * nxtiles.y; i++ ) {
        h_idx[i] = np;
        np += h_tile[i].y;
    }

    // Copy data to device
    err = cudaMalloc( &d_idx, nxtiles.x * nxtiles.y * sizeof(int) );
    CHECK_ERR( err, "Unable to allocate d_idx for gather");

    err = cudaMemcpy( d_idx, h_idx, nxtiles.x * nxtiles.y * sizeof(int), cudaMemcpyHostToDevice );
    CHECK_ERR( err, "Unable to copy idx to device");

    // Gather data
    err = cudaMalloc( &d_data, np * sizeof(float) );
    CHECK_ERR( err, "Unable to allocate d_data for gather");

    dim3 grid( nxtiles.x, nxtiles.y );
    dim3 block( 64 );

    switch (quant)
    {
    case x:
        _gather_x_kernel <<< grid, block >>> ( d_ix, d_x, d_tile, nx.x, d_idx, d_data );
        break;
    case y:
        _gather_y_kernel <<< grid, block >>> ( d_ix, d_x, d_tile, nx.y, d_idx, d_data );
        break;
    case ux:
        _gather_ux_kernel <<< grid, block >>> ( d_u, d_tile, d_idx, d_data );
        break;
    case uy:
        _gather_uy_kernel <<< grid, block >>> ( d_u, d_tile, d_idx, d_data );
        break;
    case uz:
        _gather_uz_kernel <<< grid, block >>> ( d_u, d_tile, d_idx, d_data );
        break;
    }

    cudaDeviceSynchronize();

    // Copy data to host
    err = cudaMemcpy( h_data, d_data, np * sizeof(float), cudaMemcpyDeviceToHost );
    CHECK_ERR( err, "Unable to copy data to host");

    // Free temporary data
    err = cudaFree( d_data );
    CHECK_ERR( err, "Unable to free device data");

    err = cudaFree( d_idx );
    CHECK_ERR( err, "Unable to free device idx");

    err = cudaFreeHost( h_idx );
    CHECK_ERR( err, "Unable to free host idx");
}