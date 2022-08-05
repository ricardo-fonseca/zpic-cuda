#include "field.cuh"

#include <string>
#include <iostream>

#include "util.cuh"


namespace {

/**
 * @brief CUDA kernel for Field_set
 * 
 * @param d_buffer      Type float Data buffer
 * @param val           Type float value to set
 * @param size          buffer size
 */
__global__
void _set_kernel( float *  d_buffer, const float val, size_t size ) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < size ) d_buffer[idx] = val;
}

/**
 * @brief CUDA kernel for Field::add
 * 
 * @param a     Pointer to object a data (in/out)
 * @param b     Pointer to object b data (in)
 * @param size  Number of grid elements
 */
__global__
void _add_kernel( float * __restrict__ a, float * __restrict__ b, size_t size ) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < size ) {
        a[idx] += b[idx];
    }
}

/**
 * @brief CUDA kernel for Field::gather
 * 
 * @param out       Outuput type float array
 * @param in        Input tiled Field (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernel( float * __restrict__ out, float * __restrict__ in,
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

        out[ out_idx ] = in[ in_idx ];
    }

}

/**
 * @brief CUDA kernel for adding guard cell values along x direction
 * 
 * Values from neighbouring tile guard cells are added to local tile
 * 
 * @param buffer    Global data buffer (no offset)
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcx0      Number of guard cells at the lower x boundary
 * @param gcx1      Number of guard cells at the upper x boundary
 */
__global__
void _add_gcx_kernel( float * buffer, const int2 ext_nx, const int2 int_nx,
    const int gcx0, const int gcx1 ) {

    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;

    const int x_lcoord  = ((x_coord > 0)? x_coord : gridDim.x) - 1;
    const int x_ucoord  = ((x_coord < gridDim.x-1)? x_coord : 0) + 1;

    const int tile_vol = ext_nx.x * ext_nx.y;

    float * __restrict__ local   = buffer + (y_coord * gridDim.x + x_coord ) * tile_vol;
    float * __restrict__ x_lower = buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol;
    float * __restrict__ x_upper = buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol;

    // j = [0 .. ext_nx.y[
    // i = [0 .. gc1[
    for( int idx = threadIdx.x; idx < ext_nx.y * gcx1; idx += blockDim.x ) {
        const int i = idx % gcx1;
        const int j = idx / gcx1;
        local[ gcx0 + i + j * ext_nx.x ] += x_lower[ gcx0 + int_nx.x + i + j * ext_nx.x ];
    }

    // j = [0 .. ext_nx.y[
    // i = [0 .. gc0[
    for( int idx = threadIdx.x; idx < ext_nx.y * gcx0; idx += blockDim.x ) {
        const int i = idx % gcx0;
        const int j = idx / gcx0;
        local[ int_nx.x - gcx0 + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
    }
}

/**
 * @brief CUDA kernel for adding guard cell values along y direction
 * 
 * Values from neighbouring tile guard cells are added to local tile
 * 
 * @param buffer    Global data buffer (no offset)
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcy0      Number of guard cells at the lower y boundary
 * @param gcy1      Number of guard cells at the upper y boundary
 */
__global__
void _add_gcy_kernel( float * buffer, const int2 ext_nx, const int2 int_nx,
    const int gcy0, const int gcy1 ) {

    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;

    const int y_lcoord  = ((y_coord > 0)? y_coord : gridDim.y) - 1;
    const int y_ucoord  = ((y_coord < gridDim.y-1)? y_coord : 0) + 1;

    const int tile_vol = ext_nx.x * ext_nx.y;

    float * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;
    float * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;
    float * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;

    // j = [0 .. gcy1[
    // i = [0 .. ext_nx.x[
    for( int idx = threadIdx.x; idx < gcy1 * ext_nx.x; idx += blockDim.x ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        local[ i + (gcy0 + j) * ext_nx.x ] += y_lower[ i + (gcy0 + int_nx.y + j) * ext_nx.x ];
    }

    // j = [0 .. gcy0[
    // i = [0 .. ext_nx.x[
    for( int idx = threadIdx.x; idx < gcy0 * ext_nx.x; idx += blockDim.x ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        local[ i + ( int_nx.y - gcy0 + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
    }
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
void _copy_gcx_kernel( float * buffer, const int2 ext_nx, const int2 int_nx,
    const int gcx0, const int gcx1 ) {

    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;

    const int x_lcoord  = ((x_coord > 0)? x_coord : gridDim.x) - 1;
    const int x_ucoord  = ((x_coord < gridDim.x-1)? x_coord : 0) + 1;

    const int tile_vol = ext_nx.x * ext_nx.y;

    float * __restrict__ local   = buffer + (y_coord * gridDim.x + x_coord ) * tile_vol;
    float * __restrict__ x_lower = buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol;
    float * __restrict__ x_upper = buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol;

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
void _copy_gcy_kernel( float * buffer, const int2 ext_nx, const int2 int_nx,
    const int gcy0, const int gcy1 ) {

    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;

    const int y_lcoord  = ((y_coord > 0)? y_coord : gridDim.y) - 1;
    const int y_ucoord  = ((y_coord < gridDim.y-1)? y_coord : 0) + 1;

    const int tile_vol = ext_nx.x * ext_nx.y;

    float * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;
    float * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;
    float * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;

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

}

/**
 * @brief Class Field (float grid) constructor.
 * 
 * Data is allocated both on the host and the GPU. No data initialization is performed.
 * 
 * @param gnx       Global dimensions of grid
 * @param tnx       Tile dimensions
 * @param gc        Number of guard cells
 */
__host__ Field::Field( const int2 gnx_, const int2 tnx_, const int2 gc_[2]) {

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
    size_t bsize = buffer_size( ) * sizeof( float );

    cudaError_t err;

    err = cudaMallocHost( &h_buffer, bsize );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate host memory for tiled Field." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc( &d_buffer, bsize );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate device memory for tiled Field." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }
};

/**
 * @brief Field destructor
 * 
 * Deallocates dynamic host and GPU memory
 */
__host__ Field::~Field() {

    cudaError_t err;

    // Free host memory
    err = cudaFreeHost( h_buffer );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free host memory for tiled Field." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
    }
    h_buffer = NULL;

    // Free device memory
    err = cudaFree( d_buffer );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free device memory for tiled Field." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
    }
    d_buffer = NULL;

    nx = int2{0};
    gc[0] = int2{0};
    gc[1] = int2{0};

    nxtiles = int2{0};
};

/**
 * @brief Sets host and device data to a constant value
 * 
 * @param val       Type float value
 */
__host__ void Field::set( const float val ) {

    const size_t size = buffer_size( );

    const int nthreads = 32;
    int nblocks = size / nthreads;
    if ( nthreads * nblocks < size ) nblocks++;

    _set_kernel <<< nblocks, nthreads >>> ( d_buffer, val, size );

    // set CPU data
    for( size_t i = 0; i < size; i++ ) {
        h_buffer[i] = val;
    }
};


/**
 * @brief Gather field component values from all tiles into a contiguous grid
 * 
 * Used mostly for diagnostic output
 * 
 * @param Field      Pointer to Field variable
 * @param data      Output buffer, must be pre-allocated
 */
__host__ int Field::gather_host( float *  __restrict__ h_data ) {

    // Output data x, y dimensions
    int2 gsize = { 
        .x = nxtiles.x * nx.x,
        .y = nxtiles.y * nx.y
    };

    float *  d_data;
    cudaError_t err;
    size_t size = gsize.x * gsize.y;

    err = cudaMalloc( &d_data, size * sizeof( float ));
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate device memory for Field gather()." << std::endl;
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

    _gather_kernel <<< grid, block >>> ( 
        d_data, d_buffer + offset, gsize,
        nx, ext_nx );

    // Copy data to local buffer
    err = cudaMemcpy( h_data, d_data, size * sizeof( float ), cudaMemcpyDeviceToHost );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to copy data back to cpu in Field_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Free temporary device memory
    err = cudaFree( d_data );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free device memory in Field_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
};

/**
 * @brief Adds another Field object on top of local object
 * 
 * Addition is done on device, data is not copied to CPU
 * 
 * @param rhs         Other object to add
 * @return Field&    Reference to local object
 */
__host__ void Field::add( const Field &rhs ) {

    const size_t size = buffer_size( );
    float * __restrict__ a = d_buffer;
    float * __restrict__ b = rhs.d_buffer;

    const int nthreads = 32;
    int nblocks = size / nthreads;
    if ( nthreads * nblocks < size ) nblocks++;

    _add_kernel <<< nblocks, nthreads >>> ( a, b, size );
};


/**
 * @brief Copies edge valies to neighboring guard cells
 * 
 */
__host__
void Field::copy_to_gc() {

    int2 ext = ext_nx();

    dim3 block( 64 );
    dim3 grid( nxtiles.x, nxtiles.y );

    _copy_gcx_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].x, gc[1].x
    );

    _copy_gcy_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].y, gc[1].y
    );
};

/**
 * @brief Adds values from neighboring guard cells to local data
 * 
 */
__host__
void Field::add_from_gc() {

    int2 ext = ext_nx();

    dim3 block( 64 );
    dim3 grid( nxtiles.x, nxtiles.y );

    _add_gcx_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].x, gc[1].x
    );

    _add_gcy_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].y, gc[1].y
    );

};

/**
 * @brief Save field values to disk
 * 
 * The field type float must be supported by ZDF file format
 * 
 */
__host__
void Field::save( t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {

    // Fill in grid dimensions
    info.ndims = 2;
    info.count[0] = nxtiles.x * nx.x;
    info.count[1] = nxtiles.y * nx.y;

    // Allocate buffer on host to gather data
    float *  h_buffer;
    cudaError_t err;

    err = cudaMallocHost( &h_buffer, info.count[0] * info.count[1] * sizeof( float ) );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to allocate host memory for field save" << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    if ( ! gather_host( h_buffer ) ) 
        zdf_save_grid( h_buffer, info, iter, path );

    err = cudaFreeHost( h_buffer );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to free host memory for field save" << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return;
    }

};
