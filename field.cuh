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

    float *d_buffer;

    uint2 nx;            // Tile grid size
    uint2 gc[2];         // Tile guard cells
    uint2 ntiles;       // Number of tiles in each direction

    /**
     * @brief Class Field (float grid) constructor.
     * 
     * Data is allocated both on the host and the GPU. No data initialization is performed.
     * 
     * @param gnx       Global dimensions of grid
     * @param tnx       Tile dimensions
     * @param gc        Number of guard cells
     */
    __host__ Field( uint2 const ntiles, uint2 const nx, uint2 const gc[2]);

    __host__ Field( uint2 const ntiles, uint2 const nx );

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

        // zero device data
        cudaError_t err = cudaMemset( d_buffer, 0, size );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to zero device memory for tiled grid." << std::endl;
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        return 0;
    };

    /**
     * @brief Sets host and device data to a constant value
     * 
     * @param val       Type float value
     */
    __host__ void set( float const val );

    __host__ float operator=( float const val ) {
        set( val );
        return val;
    }

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
        return ntiles.x * ntiles.y * tile_size();
    };

    /**
     * @brief Global grid size
     * 
     * @return int2 
     */
    __host__
    uint2 g_nx() {
        return make_uint2 (
            ntiles.x * nx.x,
            ntiles.y * nx.y
        );
    };

    /**
     * @brief External size of tile (inc. guard cells)
     * 
     * @return      int2 value specifying external size of tile 
     */
    __host__
    uint2 ext_nx() {
        return make_uint2(
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
    unsigned int offset() {
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