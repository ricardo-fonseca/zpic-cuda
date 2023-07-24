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
VectorField::VectorField( uint2 const ntiles, uint2 const nx, bnd<unsigned int> const gc ) :
    ntiles( ntiles ), nx( nx ), periodic( make_int2(1,1) ), gc( gc )
{
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
    ntiles( ntiles ), nx( nx ), periodic( make_int2(1,1) )
{
    gc = {0};

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

    const int tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_size ;

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
    const int tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_size ;

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

    const int tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    const int tile_size = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_size ;

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
    CHECK_ERR( err, "Unable to copy data back to cpu");

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

//#define _copy_gcx_kernel _copy_gcx_kernel_mk1
#define _copy_gcx_kernel _copy_gcx_kernel_mk2
//#define _copy_gcx_kernel _copy_gcx_kernel_mk3

/**
 * @brief CUDA kernel for updating guard cell values along x direction
 * 
 * @param buffer    Global data buffer (no offset)
 * @param periodic  Use periodic boundaries along x direction
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcx0      Number of guard cells at the lower x boundary
 * @param gcx1      Number of guard cells at the upper x boundary
 */
__global__
void _copy_gcx_kernel_mk1( 
    float3 * const __restrict__ buffer, 
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    const int gcx0, const int gcx1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float3 * __restrict__ local   = buffer + (y_coord * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int x_lcoord = x_coord - 1;
    int x_ucoord = x_coord + 1;

    if ( periodic ) {
        if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
        if ( x_ucoord >= gridDim.x ) x_ucoord -= gridDim.x;
    }

    // Copy data to lower guard cells
    if ( x_lcoord >= 0 ) {
        float3 * __restrict__ x_lower = buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol;
        // j = [0 .. ext_nx.y[
        // i = [0 .. gc0[
        for( int idx = threadIdx.x; idx < ext_nx.y * gcx0; idx += blockDim.x ) {
            const int i = idx % gcx0;
            const int j = idx / gcx0;
            local[ i + j * ext_nx.x ] = x_lower[ int_nx.x + i + j * ext_nx.x ];
        }
    }

    // Copy data to upper guard cells
    if ( x_ucoord < gridDim.x ) {
        float3 * __restrict__ x_upper = buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol;
        // j = [0 .. ext_nx.y[
        // i = [0 .. gc1[
        for( int idx = threadIdx.x; idx < ext_nx.y * gcx1; idx += blockDim.x ) {
            const int i = idx % gcx1;
            const int j = idx / gcx1;
            local[ gcx0 + int_nx.x + i + j * ext_nx.x ] = x_upper[ gcx0 + i + j * ext_nx.x ];
        }
    }
}

__global__
/**
 * @brief CUDA kernel for updating guard cell values along x direction
 * 
 * This version does all copies using float (not float3) and attempts to
 * improve coallescence.
 * 
 * Improves performance by ~ 8 % (with 128 threads/block)
 * 
 * @param buffer    Global data buffer (no offset)
 * @param periodic  Use periodic boundaries along x direction
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcx0      Number of guard cells at the lower x boundary
 * @param gcx1      Number of guard cells at the upper x boundary
 */
void _copy_gcx_kernel_mk2( 
    float3 * const __restrict__ buffer, 
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    const int gcx0, const int gcx1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local   = (float *) ( buffer + (y_coord * gridDim.x + x_coord ) * tile_vol );

    // Find neighbours
    int x_lcoord = x_coord - 1;
    int x_ucoord = x_coord + 1;

    if ( periodic ) {
        if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
        if ( x_ucoord >= gridDim.x ) x_ucoord -= gridDim.x;
    }

    // Copy data to lower guard cells
    if ( x_lcoord >= 0 ) {
        float * __restrict__ x_lower = (float *)( buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol );
        // j = [0 .. ext_nx.y[
        // i = [0 .. gc0[
        for( int k = threadIdx.x; k < 3 * ext_nx.y * gcx0; k += blockDim.x ) {
            const int idx = k / 3;
            const int fc = k % 3;

            const int i = idx % gcx0;
            const int j = idx / gcx0;
            
            local[ fc + 3 * (i + j * ext_nx.x) ] = x_lower[ fc + 3 * ( int_nx.x + i + j * ext_nx.x ) ];
        }
    }

    // Copy data to upper guard cells
    if ( x_ucoord < gridDim.x ) {
        float * __restrict__ x_upper = (float *) (buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol );
        // j = [0 .. ext_nx.y[
        // i = [0 .. gc1[
        for( int k = threadIdx.x; k < 3 * ext_nx.y * gcx1; k += blockDim.x ) {
            const int idx = k / 3;
            const int fc = k % 3;

            const int i = idx % gcx1;
            const int j = idx / gcx1;
            local[ fc + 3* (gcx0 + int_nx.x + i + j * ext_nx.x) ] = x_upper[ fc + 3 * (gcx0 + i + j * ext_nx.x) ];
        }
    }
}

__global__
void _copy_gcx_kernel_mk3( 
    float3 * const __restrict__ buffer, 
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    const int gcx0, const int gcx1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local   = (float *) ( buffer + (y_coord * gridDim.x + x_coord ) * tile_vol );

    const int block3 = ( blockDim.x / 3 ) * 3;
    const int fc = threadIdx.x % 3;

    // Find neighbours
    int x_lcoord = x_coord - 1;
    int x_ucoord = x_coord + 1;

    if ( periodic ) {
        if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
        if ( x_ucoord >= gridDim.x ) x_ucoord -= gridDim.x;
    }

    // Copy data to lower guard cells
    if ( x_lcoord >= 0 ) {
        float * __restrict__ x_lower = (float *)( buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol );

        if ( threadIdx.x < block3 ) {

            // j = [0 .. ext_nx.y[
            // i = [0 .. gc0[
            for( int idx = threadIdx.x/3; idx < ext_nx.y * gcx0; idx += block3/3 ) {
                const int i = idx % gcx0;
                const int j = idx / gcx0;
                
                local[ fc + 3 * (i + j * ext_nx.x) ] = x_lower[ fc + 3 * ( int_nx.x + i + j * ext_nx.x ) ];
            }
        }
    }

    // Copy data to upper guard cells
    if ( x_ucoord < gridDim.x ) {
        float * __restrict__ x_upper = (float *) (buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol );

        if ( threadIdx.x < block3 ) {

            // j = [0 .. ext_nx.y[
            // i = [0 .. gc1[
            for( int idx = threadIdx.x/3; idx < ext_nx.y * gcx1; idx += block3/3 ) {
                const int i = idx % gcx1;
                const int j = idx / gcx1;
                local[ fc + 3* (gcx0 + int_nx.x + i + j * ext_nx.x) ] = x_upper[ fc + 3 * (gcx0 + i + j * ext_nx.x) ];
            }
        }
    }
}


//#define _copy_gcy_kernel _copy_gcy_kernel_mk1
#define _copy_gcy_kernel _copy_gcy_kernel_mk2
//#define _copy_gcy_kernel _copy_gcy_kernel_mk3

/**
 * @brief CUDA kernel for updating guard cell values along y direction
 * 
 * @param buffer    Global data buffer (no offset)
 * @param periodic  Use periodic boundaries along y direction
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcy0      Number of guard cells at the lower y boundary
 * @param gcy1      Number of guard cells at the upper y boundary
 */
__global__
void _copy_gcy_kernel_mk1( 
    float3 * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcy0, int const gcy1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float3 * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int y_lcoord = y_coord - 1;
    int y_ucoord = y_coord + 1;

    if ( periodic ) {
        if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
        if ( y_ucoord >= gridDim.y ) y_ucoord -= gridDim.y;
    }

    // Copy data to lower guard cells
    if ( y_lcoord >= 0 ) {
        float3 * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;
        // j = [0 .. gcy0[
        // i = [0 .. ext_nx.x[
        for( int idx = threadIdx.x; idx < gcy0 * ext_nx.x; idx += blockDim.x ) {
            const int i = idx % ext_nx.x;
            const int j = idx / ext_nx.x;
            local[ i + j * ext_nx.x ] = y_lower[ i + (int_nx.y+j) * ext_nx.x ];
        }
    }

    // Copy data to upper guard cells
    if ( y_ucoord < gridDim.y ) {
        float3 * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;
        // j = [0 .. gcy1[
        // i = [0 .. ext_nx.x[
        for( int idx = threadIdx.x; idx < gcy1 * ext_nx.x; idx += blockDim.x ) {
            const int i = idx % ext_nx.x;
            const int j = idx / ext_nx.x;
            local[ i + ( gcy0 + int_nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gcy0 + j ) * ext_nx.x ];
        }
    }
}

__global__
/**
 * @brief CUDA kernel for updating guard cell values along y direction
 *
 * This version does all copies using float (not float3) and attempts to
 * improve coallescence.
 * 
 * Improves performance by ~ 20 % (with 128 threads/block)
 * (there's more opportunity for coallesced access when compared to
 * copy_gcx)
 * 
 * @param buffer    Global data buffer (no offset)
 * @param periodic  Use periodic boundaries along y direction
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcy0      Number of guard cells at the lower y boundary
 * @param gcy1      Number of guard cells at the upper y boundary
 */
void _copy_gcy_kernel_mk2( 
    float3 * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcy0, int const gcy1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local = (float *) (buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol);

    // Find neighbours
    int y_lcoord = y_coord - 1;
    int y_ucoord = y_coord + 1;

    if ( periodic ) {
        if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
        if ( y_ucoord >= gridDim.y ) y_ucoord -= gridDim.y;
    }

    // Copy data to lower guard cells
    if ( y_lcoord >= 0 ) {
        float * __restrict__ y_lower = (float *) (buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol );
        // j = [0 .. gcy0[
        // i = [0 .. ext_nx.x[
        for( int k = threadIdx.x; k < 3 * gcy0 * ext_nx.x; k += blockDim.x ) {
            const int idx = k / 3;
            const int fc = k % 3;

            const int j = idx / ext_nx.x;
            const int i = idx % ext_nx.x;

            local[ fc + 3*(i + j * ext_nx.x) ] = y_lower[ fc + 3*(i + (int_nx.y+j) * ext_nx.x) ];
        }
    }

    // Copy data to upper guard cells
    if ( y_ucoord < gridDim.y ) {
        float * __restrict__ y_upper = (float *)(buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol);
        // j = [0 .. gcy1[
        // i = [0 .. ext_nx.x[
        for( int k = threadIdx.x; k < 3 * gcy1 * ext_nx.x; k += blockDim.x ) {
            const int idx = k / 3;
            const int fc = k % 3;

            const int j = idx / ext_nx.x;
            const int i = idx % ext_nx.x;

            local[ fc + 3*(i + ( gcy0 + int_nx.y + j ) * ext_nx.x) ] = y_upper[ fc + 3*(i + ( gcy0 + j ) * ext_nx.x) ];
        }
    }
}

__global__
void _copy_gcy_kernel_mk3( 
    float3 * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcy0, int const gcy1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local = (float *) (buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol);

    const int block3 = ( blockDim.x / 3 ) * 3;
    const int fc = threadIdx.x % 3;

    // Find neighbours
    int y_lcoord = y_coord - 1;
    int y_ucoord = y_coord + 1;

    if ( periodic ) {
        if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
        if ( y_ucoord >= gridDim.y ) y_ucoord -= gridDim.y;
    }

    // Copy data to lower guard cells
    if ( y_lcoord >= 0 ) {
        float * __restrict__ y_lower = (float *) (buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol );
        
        
        if ( threadIdx.x < block3 ) {
            
            // j = [0 .. gcy0[
            // i = [0 .. ext_nx.x[
            for( int idx = threadIdx.x/3; idx < gcy0 * ext_nx.x; idx += block3/3 ) {

                const int j = idx / ext_nx.x;
                const int i = idx % ext_nx.x;

                local[ fc + 3*(i + j * ext_nx.x) ] = y_lower[ fc + 3*(i + (int_nx.y+j) * ext_nx.x) ];
            }
        }
    }

    // Copy data to upper guard cells
    if ( y_ucoord < gridDim.y ) {
        float * __restrict__ y_upper = (float *)(buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol);
        // j = [0 .. gcy1[
        // i = [0 .. ext_nx.x[
        for( int idx = threadIdx.x/3; idx < gcy1 * ext_nx.x; idx += block3/3 ) {

            const int j = idx / ext_nx.x;
            const int i = idx % ext_nx.x;

            local[ fc + 3*(i + ( gcy0 + int_nx.y + j ) * ext_nx.x) ] = y_upper[ fc + 3*(i + ( gcy0 + j ) * ext_nx.x) ];
        }
    }
}

/**
 * @brief Copies edge valies to neighboring guard cells
 * 
 */
__host__
void VectorField::copy_to_gc( ) {

    uint2 ext = ext_nx();

    dim3 block( 128 );
    dim3 grid( ntiles.x, ntiles.y );

    _copy_gcx_kernel <<< grid, block >>> (
        d_buffer, periodic.x, ext, nx, gc.x.lower, gc.x.upper
    );

    _copy_gcy_kernel <<< grid, block >>> (
        d_buffer, periodic.y, ext, nx, gc.y.lower, gc.y.upper
    );

};


/**
 * @brief CUDA kernel for adding guard cell values along x direction
 * 
 * Values from neighbouring tile guard cells are added to local tile
 * 
 * @param buffer    Global data buffer (no offset)
 * @param periodic  Use periodic boundaries along x
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcx0      Number of guard cells at the lower x boundary
 * @param gcx1      Number of guard cells at the upper x boundary
 */
__global__
void _add_gcx_kernel(
    float3 * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcx0, int const gcx1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float3 * __restrict__ local   = buffer + (y_coord * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int x_lcoord = x_coord - 1;
    int x_ucoord = x_coord + 1;

    if ( periodic ) {
        if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
        if ( x_ucoord >= gridDim.x ) x_ucoord -= gridDim.x;
    }

    if ( x_lcoord >= 0 ) {
        float3 * __restrict__ x_lower = buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol;

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
    }

    if ( x_ucoord < gridDim.x ) {
        float3 * __restrict__ x_upper = buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol;

        // j = [0 .. ext_nx.y[
        // i = [0 .. gc0[
        for( int idx = threadIdx.x; idx < ext_nx.y * gcx0; idx += blockDim.x ) {
            const int i = idx % gcx0;
            const int j = idx / gcx0;
            float3 a = local[ int_nx.x + i + j * ext_nx.x ];
            float3 b = x_upper[ i + j * ext_nx.x ];
            a.x += b.x; a.y += b.y; a.z += b.z;
            local[ int_nx.x + i + j * ext_nx.x ] = a;
        }
    }
}

/**
 * @brief CUDA kernel for adding guard cell values along y direction
 * 
 * Values from neighbouring tile guard cells are added to local tile
 * 
 * @param buffer    Global data buffer (no offset)
 * @param periodic  Use periodic boundaries along y
 * @param ext_nx    External tile size
 * @param int_nx    Internal tile size
 * @param gcy0      Number of guard cells at the lower y boundary
 * @param gcy1      Number of guard cells at the upper y boundary
 */
__global__
void _add_gcy_kernel(
    float3 * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    const int gcy0, const int gcy1 )
{
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float3 * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int y_lcoord = y_coord - 1;
    int y_ucoord = y_coord + 1;

    if ( periodic ) {
        if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
        if ( y_ucoord >= gridDim.y ) y_ucoord -= gridDim.y;
    }

    if ( y_lcoord >= 0 ) {
        float3 * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;

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
    }

    if ( y_ucoord < gridDim.y ) {
        float3 * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;
        // j = [0 .. gcy0[
        // i = [0 .. ext_nx.x[
        for( int idx = threadIdx.x; idx < gcy0 * ext_nx.x; idx += blockDim.x ) {
            const int i = idx % ext_nx.x;
            const int j = idx / ext_nx.x;
            float3 a = local[ i + ( int_nx.y + j ) * ext_nx.x ];
            float3 b = y_upper[ i + j * ext_nx.x ];
            a.x += b.x; a.y += b.y; a.z += b.z;
            local[ i + ( int_nx.y + j ) * ext_nx.x ] = a;
        }
    }
}

/**
 * @brief Adds values from neighboring guard cells to local data
 * 
 */
__host__
void VectorField::add_from_gc( )
{
    uint2 ext = ext_nx();

    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    _add_gcx_kernel <<< grid, block >>> (
        d_buffer, periodic.x, ext, nx, gc.x.lower, gc.x.upper
    );

    _add_gcy_kernel <<< grid, block >>> (
        d_buffer, periodic.y, ext, nx, gc.y.lower, gc.y.upper
    );
}

__global__
/**
 * @brief CUDA kernel for left shifting grid data
 * 
 * @param shift     Number of cells to shift
 * @param buffer    Global data buffer
 * @param ext_nx    External tile size
 */
void _x_shift_left_kernel_mk1( unsigned int const shift, float3 * const __restrict__ buffer, uint2 const ext_nx )
{
    extern __shared__ float3 local[];

    auto block = cg::this_thread_block();

    const int tid      = ( blockIdx.y * gridDim.x + blockIdx.x );
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    unsigned int offset = tid * tile_vol;

    // j = [0 .. ext_nx.y[
    // i = [0 .. ext_nx.x[
    for( int idx = block.thread_rank(); idx < ext_nx.y * ext_nx.x; idx += block.num_threads() ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        if ( i < ext_nx.x - shift ) {
            local[ i + j * ext_nx.x ] = buffer[ offset + (i + shift) + j * ext_nx.x ];
        } else {
            local[ i + j * ext_nx.x ] = make_float3( 0., 0., 0. );
        }
    }

    block.sync();

    for( int idx = block.thread_rank(); idx < ext_nx.y * ext_nx.x; idx += block.num_threads() ) {
        buffer[ offset + idx ] = local[ idx ];
    }

}

__global__
/**
 * @brief CUDA kernel for left shifting grid data - mk2
 * 
 * This version copies device data into shared memory using coallesced float4
 * operations, does the shift in local memory, and then copies back to main
 * memory using the same strategy
 * 
 * @param shift     Number of cells to shift
 * @param buffer    Global data buffer
 * @param ext_nx    External tile size
 */
void _x_shift_left_kernel_mk2( unsigned int const shift, float3 * const __restrict__ buffer, uint2 const ext_nx )
{
    extern __shared__ float3 local[];

    auto block = cg::this_thread_block();

    const int tid      = ( blockIdx.y * gridDim.x + blockIdx.x );
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    unsigned int offset = tid * tile_vol;

    float3 * __restrict__ A = (float3 *) & local[0];
    float3 * __restrict__ B = (float3 *) & local[tile_vol];

    // Copy values from global memory
    {
        float4 * __restrict__ dst = (float4 * ) A;
        float4 * __restrict__ src = (float4 * ) &buffer[ offset ];
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            dst[i] = src[i];
        }
    }

    block.sync();

    // j = [0 .. ext_nx.y[
    // i = [0 .. ext_nx.x[
    for( int idx = block.thread_rank(); idx < ext_nx.y * ext_nx.x; idx += block.num_threads() ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;
        if ( i < ext_nx.x - shift ) {
            B[ i + j * ext_nx.x ] = A[ (i + shift) + j * ext_nx.x ];
        } else {
            B[ i + j * ext_nx.x ] = make_float3( 0., 0., 0. );
        }
    }

    block.sync();

    // Copy values to global memory
    {
        float4 * __restrict__ dst = (float4 * ) &buffer[ offset ];
        float4 * __restrict__ src = (float4 * ) B;
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            dst[i] = src[i];
        }
    }

}

__global__
/**
 * @brief CUDA kernel for left shifting grid data - mk3
 * 
 * Same as mk2 but avoids float3 altogether
 * 
 * @param shift     Number of cells to shift
 * @param buffer    Global data buffer
 * @param ext_nx    External tile size
 */
void _x_shift_left_kernel_mk3( unsigned int const shift, float3 * const __restrict__ buffer, uint2 const ext_nx )
{
    extern __shared__ float3 local[];

    auto block = cg::this_thread_block();

    const int tid      = ( blockIdx.y * gridDim.x + blockIdx.x );
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    unsigned int offset = tid * tile_vol;

    // Copy values from global memory
    {
        float4 * __restrict__ dst = (float4 * ) & local[0];
        float4 * __restrict__ src = (float4 * ) & buffer[ offset ];
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            dst[i] = src[i];
        }
    }

    block.sync();

    float * __restrict__ A = (float *) & local[0];
    float * __restrict__ B = (float *) & local[tile_vol];

    // j = [0 .. ext_nx.y[
    // i = [0 .. ext_nx.x[
    for( int idx = block.thread_rank(); idx < ext_nx.y * ext_nx.x; idx += block.num_threads() ) {
        const int i = idx % ext_nx.x;
        const int j = idx / ext_nx.x;

        const int idxB = 3 * (i + j * ext_nx.x);
        const int idxA = idxB + 3 * shift;
        if ( i < ext_nx.x - shift ) {
            B[ 0 + idxB ] = A[ 0 + idxA ];
            B[ 1 + idxB ] = A[ 1 + idxA ];
            B[ 2 + idxB ] = A[ 2 + idxA ];
        } else {
            B[ 0 + idxB ] = 0;
            B[ 1 + idxB ] = 0;
            B[ 2 + idxB ] = 0;
        }
    }

    block.sync();

    // Copy values to global memory
    {
        float4 * __restrict__ dst = (float4 * ) & buffer[ offset ];
        float4 * __restrict__ src = (float4 * ) & local[tile_vol];
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            dst[i] = src[i];
        }
    }
}

__host__
/**
 * @brief Shift data left by the specified amount injecting zeros
 * 
 * Note: this operation is only allowed if the number of upper x guard cells
 *      is greater or equal to the requested shift
 * 
 * @param shift     Number of cells to shift
 */
void VectorField::x_shift_left( unsigned int const shift ) {

    if ( gc.x.upper >= shift ) {
        uint2 ext = ext_nx();

        dim3 block( 1024 );
        dim3 grid( ntiles.x, ntiles.y );


        const int tile_vol = roundup4( ext.x * ext.y );

/*
        size_t shm_size = tile_vol * sizeof(float3);
        _x_shift_left_kernel_mk1<<< grid, block, shm_size >>> ( shift, d_buffer, ext );
*/


        size_t shm_size = 2 * tile_vol * sizeof(float3);
        _x_shift_left_kernel_mk2<<< grid, block, shm_size >>> ( shift, d_buffer, ext );

/*
        size_t shm_size = 2 * tile_vol * sizeof(float3);
        _x_shift_left_kernel_mk3<<< grid, block, shm_size >>> ( shift, d_buffer, ext );
*/

        // Update x guard cells not changing lower and upper global guard cells
        _copy_gcx_kernel <<<grid, block>>> ( d_buffer, 0, ext, nx, gc.x.lower, gc.x.upper );
    } else {
        std::cerr << "(*error*) VectorField::x_shift_left(), shift value too large, must be <= gc.x.upper\n";
        cudaDeviceReset();
        exit(1);
    }
}

__global__
/**
 * @brief CUDA kernel for VectorField::kernel3_x
 * 
 * Copies data to shared memory, does convolution in shared
 * memory and then copies result to global memory
 * 
 * Requires shm_size = 2 * tile_vol * sizeof(float3);
 * 
 * @param a         Convolution kernel a value
 * @param b         Convolution kernel b value
 * @param c         Convolution kernel c value
 * @param buffer    Data buffer
 * @param int_nx    Internal tile size
 * @param ext_nx    External tile size
 * @param gcx0      Number of lower x guard cells
 */
void _kernel3_x( float const a, float const b, float const c, 
    float3 * const __restrict__ buffer, 
    uint2 const int_nx, uint2 const ext_nx, unsigned int const gcx0 )
{

    auto block = cg::this_thread_block();
    extern __shared__ float3 local[];

    const int tid      = ( blockIdx.y * gridDim.x + blockIdx.x );
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int tile_off = tid * tile_vol;

    // Copy values from global memory
    {
        float4 * __restrict__ dst = (float4 * ) & local[0];
        float4 * __restrict__ src = (float4 * ) & buffer[ tile_off ];
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            float4 v = src[i];
            dst[i] = v;
        }
    }

    block.sync();

    // Do kernel convolution in shared memory
    {
        const int stride = ext_nx.x;

        float3 * __restrict__ A = & local[0];
        float3 * __restrict__ B = & local[tile_vol];

        // The convolution is also applied in the y guard cells
        for( int idx = threadIdx.x; idx < int_nx.x * ext_nx.y; idx += blockDim.x ) {
            const int j = idx / int_nx.x;
            const int i = idx % int_nx.x + gcx0;

            float3 val;

            val.x = a * A[ j*stride + (i-1) ].x + b * A[ j*stride + i ].x + c * A[ j*stride + (i+1) ].x;
            val.y = a * A[ j*stride + (i-1) ].y + b * A[ j*stride + i ].y + c * A[ j*stride + (i+1) ].y;
            val.z = a * A[ j*stride + (i-1) ].z + b * A[ j*stride + i ].z + c * A[ j*stride + (i+1) ].z;

            B[ j*stride + i ] = val;
        }
    }

    block.sync();

    // Copy values back to global memory
    {
        float4 * __restrict__ dst = (float4 * ) & buffer[ tile_off ];
        float4 * __restrict__ src = (float4 * ) & local[ tile_vol ];
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            dst[i] = src[i];
        }
    }

}

/**
 * @brief Perform a convolution with a 3 point kernel [a,b,c] along x
 * 
 * @param a     Kernel a value
 * @param b     Kernel b value
 * @param c     Kernel c value
 */
void VectorField::kernel3_x( float const a, float const b, float const c ) {

    if (( gc.x.lower > 0) && (gc.x.upper > 0)) {

        uint2 ext = ext_nx();

        dim3 block( 1024 );
        dim3 grid( ntiles.x, ntiles.y );
        const int tile_vol = roundup4( ext.x * ext.y );

        size_t shm_size = 2 * tile_vol * sizeof(float3);
        _kernel3_x<<< grid, block, shm_size >>> ( a, b, c, d_buffer, nx, ext, gc.x.lower );       


        // _copy_gcx_kernel is faster with a smaller number of threads per block
        _copy_gcx_kernel <<<grid, 128>>> ( d_buffer, periodic.x, ext, nx, gc.x.lower, gc.x.upper );

    } else {
        std::cerr << "(*error*) VectorField::kernel_x3() requires at least 1 guard cell at both the lower and upper x boundaries.\n";
        cudaDeviceReset();
        exit(1);
    }
}

__global__
/**
 * @brief CUDA kernel for VectorField::kernel3_y
 * 
 * Copies data to shared memory, does convolution in shared
 * memory and then copies result to global memory
 * 
 * Requires shm_size = 2 * tile_vol * sizeof(float3);
 * 
 * @param a         Convolution kernel a value
 * @param b         Convolution kernel b value
 * @param c         Convolution kernel c value
 * @param buffer    Data buffer
 * @param int_nx    Internal tile size
 * @param ext_nx    External tile size
 * @param gcy0      Number of lower y guard cells
 */
void _kernel3_y( float const a, float const b, float const c, 
    float3 * const __restrict__ buffer, 
    uint2 const int_nx, uint2 const ext_nx, unsigned int const gcy0 )
{

    auto block = cg::this_thread_block();
    extern __shared__ float3 local[];

    const int tid      = ( blockIdx.y * gridDim.x + blockIdx.x );
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const int tile_off = tid * tile_vol;

    // Copy values from global memory
    {
        float4 * __restrict__ dst = (float4 * ) & local[0];
        float4 * __restrict__ src = (float4 * ) & buffer[ tile_off ];
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            float4 v = src[i];
            dst[i] = v;
        }
    }

    block.sync();

    // Do kernel convolution in shared memory
    {
        const int stride = ext_nx.x;

        float3 * __restrict__ A = & local[0];
        float3 * __restrict__ B = & local[tile_vol];

        for( int idx = threadIdx.x; idx < ext_nx.x * int_nx.y; idx += blockDim.x ) {
            const int j = idx / ext_nx.x + gcy0;
            const int i = idx % ext_nx.x;

            float3 val;

            val.x = a * A[ (j-1)*stride + i ].x + b * A[ j*stride + i ].x + c * A[ (j+1)*stride + i ].x;
            val.y = a * A[ (j-1)*stride + i ].y + b * A[ j*stride + i ].y + c * A[ (j+1)*stride + i ].y;
            val.z = a * A[ (j-1)*stride + i ].z + b * A[ j*stride + i ].z + c * A[ (j+1)*stride + i ].z;

            B[ j*stride + i ] = val;
        }
    }

    block.sync();

    // Copy values back to global memory
    {
        float4 * __restrict__ dst = (float4 * ) & buffer[ tile_off ];
        float4 * __restrict__ src = (float4 * ) & local[ tile_vol ];
        const int size = (3 * tile_vol) / 4;

        for( int i = threadIdx.x; i < size; i += blockDim.x ) {
            dst[i] = src[i];
        }
    }
}

/**
 * @brief Perform a convolution with a 3 point kernel [a,b,c] along y
 * 
 * @param a     Kernel a value
 * @param b     Kernel b value
 * @param c     Kernel c value
 */
void VectorField::kernel3_y( float const a, float const b, float const c ) {

    if (( gc.y.lower > 0) && (gc.y.upper > 0)) {

        uint2 ext = ext_nx();

        dim3 block( 1024 );
        dim3 grid( ntiles.x, ntiles.y );
        const int tile_vol = roundup4( ext.x * ext.y );

        size_t shm_size = 2 * tile_vol * sizeof(float3);
        _kernel3_y<<< grid, block, shm_size >>> ( a, b, c, d_buffer, nx, ext, gc.y.lower );

        _copy_gcy_kernel <<<grid, 128>>> ( d_buffer, periodic.y, ext, nx, gc.y.lower, gc.y.upper );

    } else {
        std::cerr << "(*error*) VectorField::kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries.\n";
        exit(1);
    }
}

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