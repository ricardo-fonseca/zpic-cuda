#ifndef __TILE_VFLD__
#define __TILE_VFLD__

#include <cstddef>
#include "tile_zdf.cuh"

/**
 * @brief VFLD class
 * 
 */
class VFLD {

    private:


    public:

    float3 *d_buffer;   // Data buffer (device)

    uint2 ntiles;       // Number of tiles in each direction
    uint2 nx;            // Tile grid size
    uint2 gc[2];         // Tile guard cells

    __host__ VFLD( uint2 const ntiles, uint2 const nx, uint2 const gc[2]);
    __host__ VFLD( uint2 const ntiles, uint2 const nx );
    __host__ ~VFLD();

    __host__ int zero();

    __host__ void set( float3 const val );

    __host__ float3 operator=( float3 const val ) { 
        set(val);
        return val;
    }
    __host__ void add( const VFLD &rhs );

    __host__ 
    int gather_host( const int fc, float * const __restrict__ h_data );

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
     * @brief  Save field values to disk
     * 
    * @param   fc          Field component to save
    * @param   info        Grid metadata (label, units, axis, etc.). Information is used to set file name
    * @param   iteration   Iteration metadata
    * @param   path        Path where to save the file
     */
    int save( const int fc, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path );
};

#endif