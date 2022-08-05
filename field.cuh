#ifndef __FIELD__
#define __FIELD__

#include "tile_zdf.cuh"
#include <iostream>


/**
 * @brief 
 * 
 */
class Field {
    
    public:

    float *h_buffer;
    float *d_buffer;

    int2 nx;            // Tile grid size
    int2 gc[2];         // Tile guard cells
    int2 nxtiles;       // Number of tiles in each direction

    /**
     * @brief Class Field (float grid) constructor.
     * 
     * Data is allocated both on the host and the GPU. No data initialization is performed.
     * 
     * @param gnx       Global dimensions of grid
     * @param tnx       Tile dimensions
     * @param gc        Number of guard cells
     */
    __host__ Field( const int2 gnx_, const int2 tnx_, const int2 gc_[2]);

    /**
     * @brief Field destructor
     * 
     * Deallocates dynamic host and GPU memory
     */
    __host__ ~Field();

    /**
     * @brief zero host and device data on a Field grid
     * 
     * Note that the device data is zeroed using the `cudaMemset()` function that is
     * asynchronous with respect to the host.
     * 
     * @return int       Returns 0 on success, -1 on error
     */
    __host__ int zero() {

        size_t size = buffer_size( ) * sizeof(float);

        // zero GPU data
        cudaError_t err = cudaMemset( d_buffer, 0, size );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to zero device memory for tiled grid." << std::endl;
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // zero CPU data
        memset( h_buffer, 0, size );

        return 0;
    };

    /**
     * @brief Sets host and device data to a constant value
     * 
     * @param val       Type float value
     */
    __host__ void set( const float val );

    enum copy_direction { host_device, device_host };
    
    /**
     * @brief   Updates device/host data from host/device
     * 
     * @param direction     Copy direction ( host_device | device_host )
     * @return int          Returns 0 on success, -1 on error
     */
    __host__ int update_data( const copy_direction direction ) {
        cudaError_t err;
        size_t size = buffer_size( ) * sizeof(float);

        switch( direction ) {
            case host_device:  // Host to device
                err = cudaMemcpy( d_buffer, h_buffer, size, cudaMemcpyHostToDevice );
                break;
            case device_host: // Device to host
                err = cudaMemcpy( h_buffer, d_buffer, size, cudaMemcpyDeviceToHost );
                break;
        }

        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable copy data in Field_update()." << std::endl;
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        return 0;
    };

    /**
     * @brief Gather field component values from all tiles into a contiguous grid
     * 
     * Used mostly for diagnostic output
     * 
     * @param Field      Pointer to Field variable
     * @param data      Output buffer, must be pre-allocated
     */
    __host__ int gather_host( float * __restrict__ h_data );

    /**
     * @brief Adds another Field object on top of local object
     * 
     * Addition is done on device, data is not copied to CPU
     * 
     * @param rhs         Other object to add
     * @return Field&    Reference to local object
     */
    __host__ void add( const Field &rhs );


    /**
     * @brief Tile size
     * 
     * @return total number of cells in tile, including guard cells 
     */
    __host__
    std::size_t tile_size() {
        return ( gc[0].x + nx.x + gc[1].x ) * ( gc[0].y + nx.y + gc[1].y );
    };

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers
     */
    __host__
    std::size_t buffer_size() {
        return nxtiles.x * nxtiles.y * tile_size();
    };

    /**
     * @brief Global grid size
     * 
     * @return int2 
     */
    __host__
    int2 g_nx() {
        return make_int2 (
            nxtiles.x * nx.x,
            nxtiles.y * nx.y
        );
    };

    /**
     * @brief External size of tile (inc. guard cells)
     * 
     * @return      int2 value specifying external size of tile 
     */
    __host__
    int2 ext_nx() {
        return make_int2(
           gc[0].x +  nx.x + gc[1].x,
           gc[0].y +  nx.y + gc[1].y
        );
    };

    /**
     * @brief External volume of tile (inc. guard cells)
     * 
     * @return      size_t value of external volume
     */
    __host__
    size_t ext_vol() {
        return ( gc[0].x +  nx.x + gc[1].x ) *
               ( gc[0].y +  nx.y + gc[1].y );
    }

    /**
     * @brief Offset in cells between lower tile corner and position (0,0)
     * 
     * @return      Offset in cells 
     */
    __host__
    int offset() {
        return gc[0].y * (gc[0].x +  nx.x + gc[1].x) + gc[0].x;
    }

    /**
     * @brief Copies edge values to neighboring guard cells
     * 
     */
    __host__
    void copy_to_gc();

    __host__
    /**
     * @brief Adds values from neighboring guard cells to local data
     * 
     */
    void add_from_gc();

    __host__
    /**
     * @brief Save field values to disk
     * 
     * The field type float must be supported by ZDF file format
     * 
     */
    void save( t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path );
};

#endif