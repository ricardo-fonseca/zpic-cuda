#ifndef __FIELD__
#define __FIELD__

#include "zdf-cpp.h"
#include <iostream>
#include "util.cuh"

/**
 * @brief 
 * 
 */
class Field {
    
    public:

    float *d_buffer;

    uint2 nx;            // Tile grid size
    bnd<unsigned int> gc;         // Tile guard cells
    uint2 ntiles;       // Number of tiles in each direction

    int2 periodic;

    /**
     * @brief Class Field (float grid) constructor.
     * 
     * Data is allocated both on the host and the GPU. No data initialization is performed.
     * 
     * @param gnx       Global dimensions of grid
     * @param tnx       Tile dimensions
     * @param gc        Number of guard cells
     */
    __host__ Field( uint2 const ntiles, uint2 const nx, bnd<unsigned int> const gc);

    __host__ Field( uint2 const ntiles, uint2 const nx );

    /**
     * @brief Field destructor
     * 
     * Deallocates dynamic host and GPU memory
     */
    __host__ ~Field();

    /**
     * @brief zero device data on a Field grid
     * 
     * Note that the device data is zeroed using the `cudaMemset()` function that is
     * asynchronous with respect to the host.
     * 
     * @return int       Returns 0 on success, -1 on error
     */
    __host__ int zero() {

        device::zero( d_buffer, buffer_size() );

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
    __host__ int gather_host( float * const __restrict__ h_data );

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
        return roundup4( ( gc.x.lower + nx.x + gc.x.upper ) * 
                         ( gc.y.lower + nx.y + gc.y.upper ) );
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
           gc.x.lower +  nx.x + gc.x.upper,
           gc.y.lower +  nx.y + gc.y.upper
        );
    };

    /**
     * @brief External volume of tile (inc. guard cells)
     * 
     * @return      size_t value of external volume
     */
    __host__
    size_t ext_vol() {
        return ( gc.x.lower +  nx.x + gc.x.upper ) *
               ( gc.y.lower +  nx.y + gc.y.upper );
    }

    /**
     * @brief Offset in cells between lower tile corner and position (0,0)
     * 
     * @return      Offset in cells 
     */
    __host__
    unsigned int offset() {
        return gc.y.lower * (gc.x.lower +  nx.x + gc.x.upper) + gc.x.lower;
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
    void save( zdf::grid_info &info, zdf::iteration &iter, std::string path );
};

#endif