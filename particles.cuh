#ifndef __PARTICLES__
#define __PARTICLES__

#include "util.cuh"
#include "zdf-cpp.h"

namespace part {
    enum quant { x, y, ux, uy, uz };
}

/**
 * @brief Information on a single particle tile
 * 
 */
typedef struct ParticlesTile {

    // Tile ID
    uint2 tile;

    // Position on global buffer
    unsigned int pos;

    // Number of particles
    unsigned int n;

    // Number of particles (b)
    // Used when 2 different sets are in the same buffer
    unsigned int nb;

} t_part_tile;

class Particles {
    
    private:

    device::Var<unsigned int> _dev_tmp_uint;
    unsigned int max_np_tile;

    public:

    uint2 ntiles;
    uint2 nx;       // Tile grid size (valid ix is in the range 0 .. nx-1)

    int2 periodic;

    // Device data pointers
    int2 *ix;
    float2 *x;
    float3 *u;

    int *idx;

    t_part_tile *tiles;

    __host__
    Particles( uint2 const ntiles, uint2 const nx, unsigned int const max_np_tile );

    __host__
    ~Particles() {
        free_dev( u );
        free_dev( x );
        free_dev( ix );

        free_dev( idx );
    }

    /**
     * @brief Returns global grid size
     * 
     * @return uint2 
     */
    __host__
    auto g_nx() { return make_uint2 ( ntiles.x * nx.x, ntiles.y * nx.y ); };

    __host__
    /**
     * @brief Gets total number of particles on device
     * 
     * @return unsigned long long   Total number of particles
     */
    unsigned int np();

    __host__
    /**
     * @brief Gets maximum number of particles per tile
     * 
     * @return unsigned int 
     */
    unsigned int np_max_tile();

    __host__
    /**
     * @brief Gets minimum number of particles per tile
     * 
     * @return unsigned int 
     */
    unsigned int np_min_tile();

    __host__
    unsigned int np_exscan( unsigned int * const __restrict__ d_offset );

    __host__
    void gather( part::quant quant, float * const __restrict__ h_data );

    __host__
    void gather( part::quant quant, float * const __restrict__ h_data, float * const __restrict__ d_data, 
        unsigned int const np, unsigned int const * const __restrict__ d_np_scan );
    
    __host__
    /**
     * @brief Validates particle data
     * 
     * In case of invalid particle data prints out an error message and aborts
     * the program
     * 
     * @param msg   Message to print in case of error
     */
    void validate( std::string msg );
    void validate( std::string msg, int const over );

    __host__
    void cell_shift( int2 const shift );

    __host__
    /**
     * @brief 
     * 
     */
    void tile_sort( );
    void tile_sort( Particles &tmp );

    __host__
    void save( zdf::part_info &info, zdf::iteration &iter, std::string path );

    __host__
    void check_tiles();
};



#endif