#include "tile_vfld.cuh"
#include <iostream>


/**
 * @brief Class vfld (float3 grid) constructor.
 * 
 * Data is allocated both on the host and the GPU. No data initialization is performed.
 * 
 * @param gnx       Global dimensions of grid
 * @param tnx       Tile dimensions
 * @param gc        Number of guard cells
 */
__host__
VFLD::VFLD( const int2 gnx_, const int2 tnx_, const int2 gc_[2] ) {

    // Validate grid and tile sizes
    if (( gnx_.x <= 0 ) || ( gnx_.y <= 0)) {
        std::cerr << "(*error*) Invalid number of cells gnx: ";
        std::cerr << gnx_.x << "," << gnx_.y << std::endl;
        exit(1);
    }

    if (( tnx_.x <= 0 ) || ( tnx_.y <= 0)) {
        std::cerr << "(*error*) Invalid tile size tnx: ";
        std::cerr << tnx_.x << "," << tnx_.y << std::endl;
        exit(1);
    }

    if ( gnx_.x % tnx_.x ) {
        std::cerr << "(*error*) global x grid size, " << gnx_.x;
        std::cerr << "is not a mutliple of x tile size, " << tnx_.x << "endl";
         exit(1);
    }

    if ( gnx_.y % tnx_.y ) {
        std::cerr << "(*error*) global y grid size, " << gnx_.y;
        std::cerr << "is not a mutliple of y tile size, " << tnx_.y << "endl";
         exit(1);
    }

    // Setup tile size
    nx = tnx_;

    // Setup guard cells
    if ( gc_ ) {
        gc[0] = gc_[0];
        gc[1] = gc_[1];
    } else {
        gc[0] = int2{0};
        gc[1] = int2{0};
    }

    // Get number of tiles in each direction
    nxtiles.x = gnx_.x / tnx_.x;
    nxtiles.y = gnx_.y / tnx_.y;

    // Allocate global buffers
    size_t bsize = buffer_size( ) * sizeof(float3);

    cudaError_t err;

    err = cudaMallocHost( &buffer, bsize );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate host memory for float3 tiled grid." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc( &d_buffer, bsize );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate device memory for float3 tiled grid." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

/**
 * @brief VFLD destructor
 * 
 * Deallocates dynamic host and GPU memory
 */
__host__
VFLD::~VFLD() {

    cudaError_t err;

    // Free host memory
    err = cudaFreeHost( buffer );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free host memory for float3 tiled grid." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
    }
    buffer = NULL;

    // Free device memory
    err = cudaFree( d_buffer );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free device memory for float3 tiled grid." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
    }
    d_buffer = NULL;

    nx = int2{0};
    gc[0] = int2{0};
    gc[1] = int2{0};

    nxtiles = int2{0};
}

/**
 * @brief zero host and device data on a vfld grid
 * 
 * Note that the device data is zeroed using the `cudaMemset()` function that is
 * asynchronous with respect to the host.
 * 
 * @return int       Returns 0 on success, -1 on error
 */
__host__
int VFLD::zero( ) {

    size_t size = buffer_size( ) * sizeof(float3);

    // zero GPU data
    cudaError_t err = cudaMemset( d_buffer, 0, size );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to zero device memory for float3 tiled grid." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // zero CPU data
    memset( buffer, 0, size );

    return 0;
}


/**
 * @brief CUDA kernel for vfld_set
 * 
 * @param d_buffer      float3 Data buffer
 * @param val           float3 value to set
 * @param size          buffer size
 */
__global__
void _set_kernel( float3* d_buffer, const float3 val, size_t size ) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < size ) d_buffer[idx] = val;
}

/**
 * @brief Sets host and device data to a constant value
 * 
 * @param vfld      Pointer to vfld variable
 * @param val       float3 value
 */
__host__
void VFLD :: set( const float3 val ) {

    const size_t size = buffer_size( );

    const int nthreads = 32;
    int nblocks = size / nthreads;
    if ( nthreads * nblocks < size ) nblocks++;

    _set_kernel <<< nblocks, nthreads >>> ( d_buffer, val, size );

    // set CPU data
    for( size_t i = 0; i < size; i++ ) {
        buffer[i] = val;
    }
}

/**
 * @brief   Updates device/host data from host/device
 * 
 * @param direction     Copy direction, 0: host -> device, 1: device -> host
 * @return int          Returns 0 on success, -1 on error
 */
__host__
int VFLD :: update_data( const copy_direction direction ) {
    cudaError_t err;
    size_t size = buffer_size( ) * sizeof(float3);

    switch( direction ) {
        case host_device:  // Host to device
            err = cudaMemcpy( d_buffer, buffer, size, cudaMemcpyHostToDevice );
            break;
        case device_host: // Device to host
            err = cudaMemcpy( buffer, d_buffer, size, cudaMemcpyDeviceToHost );
            break;
    }

    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable copy data in vfld_update()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}

#if 0

/**
 * @brief CUDA kernel for VFLD::gather(x)
 * 
 * @param out       Outuput float array
 * @param in        Input float3 tiled array (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernelx( float * out, float3 * in,
    int2 const gnx, int2 const int_nx, int2 const ext_nx ) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ( idx < gnx.x * gnx.y ) {

        size_t vol_int = int_nx.x * int_nx.y;

        int tile = idx / vol_int;
        int tidx = idx - tile * vol_int;

        // Position inside tile
        int iy = tidx / int_nx.x;
        int ix = tidx - iy * int_nx.x;

        size_t in_idx = ( tile * ext_nx.y + iy ) * ext_nx.x + ix;

        int ntiles_x =  gnx.x / int_nx.x;
        int tile_y = tile / ntiles_x;
        int tile_x = tile - tile_y * ntiles_x;

        size_t out_idx = ( tile_y * int_nx.y + iy ) * gnx.x + ( tile_x * int_nx.x + ix );
        out[ out_idx ] = in[ in_idx ].x;
    }
}


/**
 * @brief Gather field component values from all tiles into a contiguous grid
 * 
 * Used mostly for diagnostic output
 * 
 * @param vfld      Pointer to vfld variable
 * @param fc        Field component choice (1, 2 or 3)
 * @param data      Output buffer, must be pre-allocated
 */
__host__
int VFLD :: gather( const int fc, float * data ) {

    // Output data x, y dimensions
    int2 gsize = { 
        .x = nxtiles.x * nx.x,
        .y = nxtiles.y * nx.y
    };

    float* d_data;
    cudaError_t err;
    ssize_t size = gsize.x * gsize.y;

    err = cudaMalloc( &d_data, size * sizeof(float));
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate device memory for vfld_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Tile block size (grid + guard cells)
    int2 ext_nx = {
        .x = gc[0].x +  nx.x + gc[1].x,
        .y = gc[0].y +  nx.y + gc[1].y
    };

    int offset = gc[0].y * ext_nx.x + gc[0].x;
    
    // Gather data on device
    const int nthreads = 32;
    int nblocks = size / nthreads;
    if ( nthreads * nblocks < size ) nblocks++;

    switch (fc) {
/*
        case 2:
            _gather_kernelz <<< nblocks, nthreads >>> ( 
                d_data, d_buffer + offset, size,
                nx, ext_nx );
            break;
        case 1:
            _gather_kernely <<< nblocks, nthreads >>> ( 
                d_data, d_buffer + offset, size,
                nx, ext_nx );
            break;
*/
        default:
            _gather_kernelx <<< nblocks, nthreads >>> ( 
                d_data, d_buffer + offset, gsize,
                nx, ext_nx );
    }

    // Copy data to local buffer
    err = cudaMemcpy( data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to copy data back to cpu in vfld_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Free temporary device memory
    err = cudaFree( d_data );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free device memory in vfld_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}

#endif


/**
 * @brief CUDA kernel for VFLD::gather(x)
 * 
 * @param out       Outuput float array
 * @param in        Input float3 tiled array (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernelx( float * out, float3 * in,
    int2 const gnx, int2 const int_nx, int2 const ext_nx ) {

    int    tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    size_t tile_off = tile_id * ext_nx.x * ext_nx.y;

    for( int i = threadIdx.x; i < int_nx.x * int_nx.y; i+= blockDim.x ) {
        int const ix = i % int_nx.x;
        int const iy = i / int_nx.x;

        size_t const in_idx = tile_off + iy * ext_nx.x + ix;

        int const gix = blockIdx.x * int_nx.x + ix;
        int const giy = blockIdx.y * int_nx.y + iy;

        size_t const out_idx = giy * gnx.x + gix;

        out[ out_idx ] = in[ in_idx ].x;
    }
}

/**
 * @brief CUDA kernel for VFLD::gather(y)
 * 
 * @param out       Outuput float array
 * @param in        Input float3 tiled array (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernely( float * out, float3 * in,
    int2 const gnx, int2 const int_nx, int2 const ext_nx ) {

    int    tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    size_t tile_off = tile_id * ext_nx.x * ext_nx.y;

    for( int i = threadIdx.x; i < int_nx.x * int_nx.y; i+= blockDim.x ) {
        int const ix = i % int_nx.x;
        int const iy = i / int_nx.x;

        size_t const in_idx = tile_off + iy * ext_nx.x + ix;

        int const gix = blockIdx.x * int_nx.x + ix;
        int const giy = blockIdx.y * int_nx.y + iy;

        size_t const out_idx = giy * gnx.x + gix;

        out[ out_idx ] = in[ in_idx ].y;
    }
}
/**
 * @brief CUDA kernel for VFLD::gather(z)
 * 
 * @param out       Outuput float array
 * @param in        Input float3 tiled array (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernelz( float * out, float3 * in,
    int2 const gnx, int2 const int_nx, int2 const ext_nx ) {

    int    tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    size_t tile_off = tile_id * ext_nx.x * ext_nx.y;

    for( int i = threadIdx.x; i < int_nx.x * int_nx.y; i+= blockDim.x ) {
        int const ix = i % int_nx.x;
        int const iy = i / int_nx.x;

        size_t const in_idx = tile_off + iy * ext_nx.x + ix;

        int const gix = blockIdx.x * int_nx.x + ix;
        int const giy = blockIdx.y * int_nx.y + iy;

        size_t const out_idx = giy * gnx.x + gix;

        out[ out_idx ] = in[ in_idx ].z;
    }
}

/**
 * @brief Gather field component values from all tiles into a contiguous grid
 * 
 * Used mostly for diagnostic output
 * 
 * @param vfld      Pointer to vfld variable
 * @param fc        Field component choice (1, 2 or 3)
 * @param data      Output buffer, must be pre-allocated
 */
__host__
int VFLD :: gather( const int fc, float * data ) {

    // Output data x, y dimensions
    int2 gsize = { 
        .x = nxtiles.x * nx.x,
        .y = nxtiles.y * nx.y
    };

    float* d_data;
    cudaError_t err;
    ssize_t size = gsize.x * gsize.y;

    err = cudaMalloc( &d_data, size * sizeof(float));
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate device memory for vfld_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Tile block size (grid + guard cells)
    int2 ext_nx = {
        .x = gc[0].x +  nx.x + gc[1].x,
        .y = gc[0].y +  nx.y + gc[1].y
    };

    int offset = gc[0].y * ext_nx.x + gc[0].x;
    
    // Gather data on device
    dim3 block( 64 );
    dim3 grid( nxtiles.x, nxtiles.y );

    switch (fc) {
        case 2:
            _gather_kernelz <<< grid, block >>> ( 
                d_data, d_buffer + offset, gsize,
                nx, ext_nx );
            break;
        case 1:
            _gather_kernely <<< grid, block >>> ( 
                d_data, d_buffer + offset, gsize,
                nx, ext_nx );
            break;
        default:
            _gather_kernelx <<< grid, block >>> ( 
                d_data, d_buffer + offset, gsize,
                nx, ext_nx );
    }

    // Copy data to local buffer
    err = cudaMemcpy( data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to copy data back to cpu in vfld_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Free temporary device memory
    err = cudaFree( d_data );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free device memory in vfld_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}


/**
 * @brief CUDA kernel for VFLD::add
 * 
 * @param a     Pointer to object a data (in/out)
 * @param b     Pointer to object b data (in)
 * @param size  Number of grid elements
 */
__global__
void _add_kernel( float3 * __restrict__ a, float3 * __restrict__ b, size_t size ) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < size ) {
        float3 sa = a[idx];
        float3 const sb = b[idx];

        sa.x += sb.x;
        sa.y += sb.y;
        sa.z += sb.z;

        a[idx] = sa;
    }
}

/**
 * @brief Adds another VFLD object on top of local object
 * 
 * Addition is done on device, data is not copied to CPU
 * 
 * @param rhs         Other object to add
 * @return VFLD&    Reference to local object
 */
void VFLD::add( const VFLD &rhs ) {

    const size_t size = buffer_size( );
    float3 * __restrict__ a = d_buffer;
    float3 * __restrict__ b = rhs.d_buffer;

    const int nthreads = 32;
    int nblocks = size / nthreads;
    if ( nthreads * nblocks < size ) nblocks++;

    _add_kernel <<< nblocks, nthreads >>> ( a, b, size );
}



/**
 * @brief CUDA kernel for updating guard cell values along x direction
 * 
 * @param buffer    Global data buffer (no offset)
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcx0      Number of guard cells at the lower x boundary
 * @param gcx1      Number of guard cells at the upper x boundary
 */
__global__
void _update_gcx_kernel( float3 * buffer, const int2 ext_nx, const int2 int_nx,
    const int gcx0, const int gcx1 ) {

    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;

    const int x_lcoord  = ((x_coord > 0)? x_coord : gridDim.x) - 1;
    const int x_ucoord  = ((x_coord < gridDim.x-1)? x_coord : 0) + 1;

    const int tile_vol = ext_nx.x * ext_nx.y;

    float3 * __restrict__ local   = buffer + (y_coord * gridDim.x + x_coord ) * tile_vol;
    float3 * __restrict__ x_lower = buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol;
    float3 * __restrict__ x_upper = buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol;

    // j = [0 .. ext_nx.y[
    // i = [0 .. gc0[
    for( int idx = threadIdx.x; idx < ext_nx.y * gcx0; idx += blockDim.x ) {
        const int i = idx % gcx0;
        const int j = idx / gcx0;
        local[ i + j * ext_nx.x ] = x_lower[ int_nx.x + i + j * ext_nx.x ];
    }

    // j = [0 .. ext_nx.y[
    // i = [0 .. gc1[
    for( int idx = threadIdx.x; idx < ext_nx.y * gcx1; idx += blockDim.x ) {
        const int i = idx % gcx1;
        const int j = idx / gcx1;
        local[ gcx0 + int_nx.x + i + j * ext_nx.x ] = x_upper[ gcx0 + i + j * ext_nx.x ];
    }
}

/**
 * @brief CUDA kernel for updating guard cell values along y direction
 * 
 * @param buffer    Global data buffer (no offset)
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcy0      Number of guard cells at the lower y boundary
 * @param gcy1      Number of guard cells at the upper y boundary
 */
__global__
void _update_gcy_kernel( float3 * buffer, const int2 ext_nx, const int2 int_nx,
    const int gcy0, const int gcy1 ) {

    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;

    const int y_lcoord  = ((y_coord > 0)? y_coord : gridDim.y) - 1;
    const int y_ucoord  = ((y_coord < gridDim.y-1)? y_coord : 0) + 1;

    const int tile_vol = ext_nx.x * ext_nx.y;

    float3 * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;
    float3 * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;
    float3 * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;

    // j = [0 .. gcy0[
    // i = [0 .. ext_nx.x[
    for( int idx = threadIdx.x; idx < gcy0 * ext_nx.x; idx += blockDim.x ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        local[ i + j * ext_nx.x ] = y_lower[ i + (int_nx.y+j) * ext_nx.x ];
    }

    // j = [0 .. gcy1[
    // i = [0 .. ext_nx.x[
    for( int idx = threadIdx.x; idx < gcy1 * ext_nx.x; idx += blockDim.x ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        local[ i + ( gcy0 + int_nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gcy0 + j ) * ext_nx.x ];
    }
}

/**
 * @brief Updates guard cell values
 * 
 * Guard cell values are copied from neighboring tiles assuming periodic boundaries
 * Values are copied along x first and then along y.
 * 
 */
void VFLD::update_gc() {

    int2 ext = ext_nx();

    dim3 block( 64 );
    dim3 grid( nxtiles.x, nxtiles.y );

    _update_gcx_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].x, gc[1].x
    );

    _update_gcy_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].y, gc[1].y
    );

}
