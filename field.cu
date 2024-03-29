#include "field.cuh"

#include <string>
#include <iostream>

#include "util.cuh"


/**
 * @brief Construct a new VFLD::VFLD object
 * 
 * Data is allocated on the GPU. No data initialization is performed.
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 * @param gc        Number of guard cells
 */
__host__ Field::Field( uint2 const ntiles, uint2 const nx,  bnd<unsigned int> const gc) :
    ntiles( ntiles ), nx( nx ), periodic( make_int2(1,1) ), gc(gc)
{
    malloc_dev( d_buffer, buffer_size() );
};

/**
 * @brief Construct a new VFLD::VFLD object without guard cells
 * 
 * Data is allocated both on the host and the GPU. No data initialization is performed.
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 */
__host__ Field::Field( uint2 const ntiles, uint2 const nx ) :
    ntiles( ntiles ), nx( nx ), periodic( make_int2(1,1) )
{
    gc = {0};

    malloc_dev( d_buffer, buffer_size() );
};

/**
 * @brief Field destructor
 * 
 * Deallocates dynamic GPU memory
 */
__host__ Field::~Field() {

    free_dev( d_buffer );
};


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
 * @brief Sets host and device data to a constant value
 * 
 * @param val       Type float value
 */
__host__ void Field::set( const float val ) {

    const size_t size = buffer_size( );
    int const block = 32;
    int const grid = (size -1) / block + 1;

    _set_kernel <<< grid, block >>> ( d_buffer, val, size );
};


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
 * @brief Adds another Field object on top of local object
 * 
 * Addition is done on device, data is not copied to CPU
 * 
 * @param rhs         Other object to add
 * @return Field&    Reference to local object
 */
__host__ void Field::add( const Field &rhs ) {
    size_t const size = buffer_size( );
    int const block = 32;
    int const grid = (size -1) / block + 1;

    _add_kernel <<< grid, block >>> ( d_buffer, rhs.d_buffer, size );
};


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
void _field_gather_kernel( 
    float * const __restrict__ out, 
    float const * const __restrict__ in,
    uint2 const gnx, uint2 const int_nx, uint2 const ext_nx ) {

    const int    tile_id  = blockIdx.y * gridDim.x + blockIdx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol ;


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
 * @brief Gather field component values from all tiles into a contiguous grid
 * 
 * Used mostly for diagnostic output
 * 
 * @param Field      Pointer to Field variable
 * @param data      Output buffer, must be pre-allocated
 */
__host__ int Field::gather_host( float *  __restrict__ h_data ) {

    // Output data x, y dimensions
    uint2 gsize = make_uint2( 
        ntiles.x * nx.x,
        ntiles.y * nx.y
    );

    float *  d_data;
    size_t const size = gsize.x * gsize.y;
    malloc_dev( d_data, size );
    
    // Gather data on device
    dim3 block( 64 );
    dim3 grid( ntiles.x, ntiles.y );

    _field_gather_kernel <<< grid, block >>> ( 
        d_data, d_buffer + offset(), gsize,
        nx, ext_nx() );

    // Copy data to local buffer
    auto err = cudaMemcpy( h_data, d_data, size * sizeof( float ), cudaMemcpyDeviceToHost );
    CHECK_ERR( err, "Unable to copy data back to cpu");

    free_dev( d_data );

    return 0;
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
    float * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcx0, int const gcx1 ) {

    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local   = buffer + (y_coord * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int x_lcoord = x_coord - 1;
    int x_ucoord = x_coord + 1;

    if ( periodic ) {
        if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
        if ( x_ucoord > gridDim.x-1 ) x_ucoord -= gridDim.x;
    }

    // Add from lower neighbour
    if ( x_lcoord >= 0 ) {
        float * __restrict__ x_lower = buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol;
        // j = [0 .. ext_nx.y[
        // i = [0 .. gc1[
        for( int idx = threadIdx.x; idx < ext_nx.y * gcx1; idx += blockDim.x ) {
            const int i = idx % gcx1;
            const int j = idx / gcx1;
            local[ gcx0 + i + j * ext_nx.x ] += x_lower[ gcx0 + int_nx.x + i + j * ext_nx.x ];
        }
    }

    // Add from upper neighbour
    if ( x_ucoord < gridDim.x ) {
        float * __restrict__ x_upper = buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol;
        // j = [0 .. ext_nx.y[
        // i = [0 .. gc0[
        for( int idx = threadIdx.x; idx < ext_nx.y * gcx0; idx += blockDim.x ) {
            const int i = idx % gcx0;
            const int j = idx / gcx0;
            local[ int_nx.x + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
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
    float * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcy0, int const gcy1 ) {

    // Find neighbours
    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int y_lcoord = y_coord - 1;
    int y_ucoord = y_coord + 1;

    if ( periodic ) {
        if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
        if ( y_ucoord > gridDim.x-1 ) y_ucoord -= gridDim.y;
    }

    // Add from lower neighbour
    if ( y_lcoord >= 0 ) {
        float * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;
        // j = [0 .. gcy1[
        // i = [0 .. ext_nx.x[
        for( int idx = threadIdx.x; idx < gcy1 * ext_nx.x; idx += blockDim.x ) {
            const int i = idx % ext_nx.x;
            const int j = idx / ext_nx.x;
            local[ i + (gcy0 + j) * ext_nx.x ] += y_lower[ i + (gcy0 + int_nx.y + j) * ext_nx.x ];
        }
    }

    // Add from upper neighbour
    if ( y_ucoord < gridDim.y ) {
        float * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;
        // j = [0 .. gcy0[
        // i = [0 .. ext_nx.x[
        for( int idx = threadIdx.x; idx < gcy0 * ext_nx.x; idx += blockDim.x ) {
            const int i = idx % ext_nx.x;
            const int j = idx / ext_nx.x;
            local[ i + ( int_nx.y + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
        }
    }
}

/**
 * @brief Adds values from neighboring guard cells to local data
 * 
 */
__host__
void Field::add_from_gc() {

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );

    _add_gcx_kernel <<< grid, block >>> (
        d_buffer, periodic.x, ext_nx(), nx, gc.x.lower, gc.x.upper
    );

    _add_gcy_kernel <<< grid, block >>> (
        d_buffer, periodic.y, ext_nx(), nx, gc.y.lower, gc.y.upper
    );

};


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
void _copy_gcx_kernel( 
    float * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    const int gcx0, const int gcx1 ) {

    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local   = buffer + (y_coord * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int x_lcoord = x_coord - 1;
    int x_ucoord = x_coord + 1;

    if ( periodic ) {
        if ( x_lcoord < 0 ) x_lcoord += gridDim.x;
        if ( x_ucoord > gridDim.x-1 ) x_ucoord -= gridDim.x;
    }

    // Copy data to lower guard cells
    if ( x_lcoord >= 0 ) {
        float * __restrict__ x_lower = buffer + (y_coord * gridDim.x + x_lcoord) * tile_vol;
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
        float * __restrict__ x_upper = buffer + (y_coord * gridDim.x + x_ucoord) * tile_vol;
        // j = [0 .. ext_nx.y[
        // i = [0 .. gc1[
        for( int idx = threadIdx.x; idx < ext_nx.y * gcx1; idx += blockDim.x ) {
            const int i = idx % gcx1;
            const int j = idx / gcx1;
            local[ gcx0 + int_nx.x + i + j * ext_nx.x ] = x_upper[ gcx0 + i + j * ext_nx.x ];
        }
    }
}

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
void _copy_gcy_kernel(
    float * const __restrict__ buffer,
    int const periodic,
    uint2 const ext_nx, uint2 const int_nx,
    int const gcy0, int const gcy1 ) {

    const int y_coord = blockIdx.y;
    const int x_coord = blockIdx.x;
    const int tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float * __restrict__ local   = buffer + (y_coord  * gridDim.x + x_coord ) * tile_vol;

    // Find neighbours
    int y_lcoord = y_coord - 1;
    int y_ucoord = y_coord + 1;

    if ( periodic ) {
        if ( y_lcoord < 0 ) y_lcoord += gridDim.y;
        if ( y_ucoord > gridDim.x-1 ) y_ucoord -= gridDim.y;
    }

    // Copy data to lower guard cells
    if ( y_lcoord >= 0 ) {
        float * __restrict__ y_lower = buffer + (y_lcoord * gridDim.x + x_coord ) * tile_vol;
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
        float * __restrict__ y_upper = buffer + (y_ucoord * gridDim.x + x_coord ) * tile_vol;
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
 * @brief Copies edge valies to neighboring guard cells
 * 
 */
__host__
void Field::copy_to_gc() {

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );

    _copy_gcx_kernel <<< grid, block >>> (
        d_buffer, periodic.x, ext_nx(), nx, gc.x.lower, gc.x.upper
    );

    _copy_gcy_kernel <<< grid, block >>> (
        d_buffer, periodic.y, ext_nx(), nx, gc.y.lower, gc.y.upper
    );
};

/**
 * @brief Save field values to disk
 * 
 * The field type float must be supported by ZDF file format
 * 
 */
__host__
void Field::save( zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

    // Fill in grid dimensions
    info.ndims = 2;
    info.count[0] = ntiles.x * nx.x;
    info.count[1] = ntiles.y * nx.y;

    // Allocate buffer on host to gather data
    float *h_data;
    malloc_host( h_data, info.count[0] * info.count[1] );

    if ( ! gather_host( h_data ) )
        zdf::save_grid( h_data, info, iter, path );

    free_host( h_data );

};
