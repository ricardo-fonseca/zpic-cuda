#include "vector_field.cuh"
#include <iostream>
#include <string>

#include "util.cuh"

__host__
/**
 * @brief Construct a new VectorField::VectorField object
 * 
 * Data is allocated only on the GPU (device). No data initialization is performed.
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 * @param gc        Number of guard cells
 */
VectorField::VectorField( uint2 const ntiles, uint2 const nx, uint2 const gc_[2]) :
    ntiles( ntiles ), nx( nx )
{
    gc[0] = gc_[0];
    gc[1] = gc_[1];

    malloc_dev( d_buffer, buffer_size() );
}

/**
 * @brief Construct a new VectorField::VectorField object without guard cells
 * 
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 */
VectorField::VectorField( uint2 const ntiles, uint2 const nx ) :
    ntiles( ntiles ), nx( nx )
{
    gc[0] = {0};
    gc[1] = {0};

    malloc_dev( d_buffer, buffer_size() );
}

/**
 * @brief VectorField destructor
 * 
 * Deallocates dynamic GPU memory
 */
__host__
VectorField::~VectorField() {
    free_dev( d_buffer );
}

/**
 * @brief CUDA kernel for VectorField_set
 * 
 * @param d_buffer      float3 Data buffer
 * @param val           float3 value to set
 * @param size          buffer size
 */
__global__
void _set_kernel( float3 * const __restrict__ d_buffer, float3 const val, size_t const size ) {
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < size ) d_buffer[idx] = val;
}

/**
 * @brief Sets host and device data to a constant value
 * 
 * @param VectorField      Pointer to VectorField variable
 * @param val       float3 value
 */
__host__
void VectorField :: set( const float3 val ) {

    size_t const size = buffer_size( );
    int const block = 32;
    int const grid = (size -1) / block + 1;

    _set_kernel <<< grid, block >>> ( d_buffer, val, size );
}

/**
 * @brief CUDA kernel for VectorField::gather(x)
 * 
 * @param out       Outuput float array
 * @param in        Input float3 tiled array (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernelx( 
    float * const __restrict__ out, float3 const * const __restrict__ in,
    uint2 const gnx, uint2 const int_nx, uint2 const ext_nx ) {

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
 * @brief CUDA kernel for VectorField::gather(y)
 * 
 * @param out       Outuput float array
 * @param in        Input float3 tiled array (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernely( 
        float * const __restrict__ out, float3 const * const __restrict__ in,
    uint2 const gnx, uint2 const int_nx, uint2 const ext_nx )
{
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
 * @brief CUDA kernel for VectorField::gather(z)
 * 
 * @param out       Outuput float array
 * @param in        Input float3 tiled array (must include offset to position 0,0)
 * @param size      Global grid dimensions
 * @param int_nx    Internal dimensions of tile
 * @param ext_nx    External dimensions of tile
 */
__global__
void _gather_kernelz(
    float * const __restrict__ out, float3 const * const __restrict__ in,
    uint2 const gnx, uint2 const int_nx, uint2 const ext_nx )
{

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
 * @param VectorField      Pointer to VectorField variable
 * @param fc        Field component choice (1, 2 or 3)
 * @param data      Output buffer, must be pre-allocated
 */
__host__
int VectorField :: gather_host( const int fc, float * const __restrict__ h_data )
 {

    // Output data x, y dimensions
    uint2 gsize = make_uint2( 
        ntiles.x * nx.x,
        ntiles.y * nx.y
    );

    float * d_data;
    size_t const size = gsize.x * gsize.y;
    malloc_dev( d_data, size );

    // Gather data on device
    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    switch (fc) {
        case 2:
            _gather_kernelz <<< grid, block >>> ( 
                d_data, d_buffer + offset(), gsize,
                nx, ext_nx() );
            break;
        case 1:
            _gather_kernely <<< grid, block >>> ( 
                d_data, d_buffer + offset(), gsize,
                nx, ext_nx() );
            break;
        default:
            _gather_kernelx <<< grid, block >>> ( 
                d_data, d_buffer + offset(), gsize,
                nx, ext_nx() );
    }

    // Copy data to local buffer
    auto err = cudaMemcpy( h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to copy data back to cpu in VectorField_gather()." << std::endl;
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    free_dev( d_data );

    return 0;
}


/**
 * @brief CUDA kernel for VectorField::add
 * 
 * @param a     Pointer to object a data (in/out)
 * @param b     Pointer to object b data (in)
 * @param size  Number of grid elements
 */
__global__
void _add_kernel( 
    float3 * const __restrict__ a, float3 const * const __restrict__ b,
    size_t size )
{
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
 * @brief Adds another VectorField object on top of local object
 * 
 * Addition is done on device, data is not copied to CPU
 * 
 * @param rhs         Other object to add
 * @return VectorField&    Reference to local object
 */
void VectorField::add( const VectorField &rhs )
{
    size_t const size = buffer_size( );
    int const block = 32;
    int const grid = (size -1) / block + 1;

    _add_kernel <<< grid, block >>> ( d_buffer, rhs.d_buffer, size );
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
void _copy_gcx_kernel( 
    float3 * const __restrict__ buffer, 
    uint2 const ext_nx, uint2 const int_nx,
    const int gcx0, const int gcx1 )
{
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
void _copy_gcy_kernel( 
    float3 * const __restrict__ buffer,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcy0, int const gcy1 )
{
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
 * @brief Copies edge valies to neighboring guard cells
 * 
 */
__host__
void VectorField::copy_to_gc() {

    uint2 ext = ext_nx();

    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    _copy_gcx_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].x, gc[1].x
    );

    _copy_gcy_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].y, gc[1].y
    );
};


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
void _add_gcx_kernel(
    float3 * const __restrict__ buffer, 
    uint2 const ext_nx, uint2 const int_nx,
    int const gcx0, int const gcx1 )
{
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
    // i = [0 .. gc1[
    for( int idx = threadIdx.x; idx < ext_nx.y * gcx1; idx += blockDim.x ) {
        const int i = idx % gcx1;
        const int j = idx / gcx1;
        float3 a = local[ gcx0 + i + j * ext_nx.x ];
        float3 b = x_lower[ gcx0 + int_nx.x + i + j * ext_nx.x ];
        a.x += b.x; a.y += b.y; a.z += b.z;
        local[ gcx0 + i + j * ext_nx.x ] = a;
    }

    // j = [0 .. ext_nx.y[
    // i = [0 .. gc0[
    for( int idx = threadIdx.x; idx < ext_nx.y * gcx0; idx += blockDim.x ) {
        const int i = idx % gcx0;
        const int j = idx / gcx0;
        float3 a = local[ int_nx.x - gcx0 + i + j * ext_nx.x ];
        float3 b = x_upper[ i + j * ext_nx.x ];
        a.x += b.x; a.y += b.y; a.z += b.z;
        local[ int_nx.x - gcx0 + i + j * ext_nx.x ] = a;
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
void _add_gcy_kernel(
    float3 * const __restrict__ buffer,
    uint2 const ext_nx, uint2 const int_nx,
    const int gcy0, const int gcy1 )
{
    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;

    const int y_lcoord  = ((y_coord > 0)? y_coord : gridDim.y) - 1;
    const int y_ucoord  = ((y_coord < gridDim.y-1)? y_coord : 0) + 1;

    const int tile_vol = ext_nx.x * ext_nx.y;

    float3 * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;
    float3 * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;
    float3 * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;

    // j = [0 .. gcy1[
    // i = [0 .. ext_nx.x[
    for( int idx = threadIdx.x; idx < gcy1 * ext_nx.x; idx += blockDim.x ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        float3 a = local[ i + (j + gcy0) * ext_nx.x ];
        float3 b = y_lower[ i + (gcy0 + int_nx.y + j) * ext_nx.x ];
        a.x += b.x; a.y += b.y; a.z += b.z;
        local[ i + (j + gcy0) * ext_nx.x ] = a;
    }

    // j = [0 .. gcy0[
    // i = [0 .. ext_nx.x[
    for( int idx = threadIdx.x; idx < gcy0 * ext_nx.x; idx += blockDim.x ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        float3 a = local[ i + ( int_nx.y - gcy0 + j ) * ext_nx.x ];
        float3 b = y_upper[ i + j * ext_nx.x ];
        a.x += b.x; a.y += b.y; a.z += b.z;
        local[ i + ( int_nx.y - gcy0 + j ) * ext_nx.x ] = a;
    }
}

/**
 * @brief Adds values from neighboring guard cells to local data
 * 
 */
__host__
void VectorField::add_from_gc()
{
    uint2 ext = ext_nx();

    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    _add_gcx_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].x, gc[1].x
    );

    _add_gcy_kernel <<< grid, block >>> (
        d_buffer, ext, nx, gc[0].y, gc[1].y
    );
};


__host__
/**
 * @brief  Save field values to disk
 * 
* @param   fc          Field component to save
* @param   info        Grid metadata (label, units, axis, etc.). Information is used to set file name
* @param   iter        Iteration metadata
* @param   path        Path where to save the file
*/
int VectorField::save( const int fc, zdf::grid_info &info, zdf::iteration &iter, std::string path )
{

    // Fill in grid dimensions
    info.ndims = 2;
    info.count[0] = ntiles.x * nx.x;
    info.count[1] = ntiles.y * nx.y;

    // Allocate buffer on host to gather data
    float *h_data;
    malloc_host( h_data, info.count[0] * info.count[1] );

    if ( ! gather_host( fc, h_data ) )
        zdf::save_grid( h_data, info, iter, path );

    free_host( h_data );

    return 0;
}