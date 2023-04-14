#ifndef __PARTICLES__
#define __PARTICLES__

#include "util.cuh"
#include "zdf-cpp.h"

namespace part {
    enum quant { x, y, ux, uy, uz };
}

/**
 * @brief Individual particle data
 * 
 * Data is organized as a structure of arrays (SoA) and stored on device
 * 
 */
typedef struct ParticleData {
    /// @brief Particle position (cell index)
    int2 *ix;
    /// @brief Particle position (position inside cell)
    float2 *x;
    /// @brief Particle velocity
    float3 *u;
} t_part_data;

/**
 * @brief Tile information
 * 
 */
typedef struct ParticleTiles {

    /// @brief Number of particles in tile
    int * np;
    
    /// @brief Tile particle position on global array
    int * offset;

    /// @brief Secondary number of particles in tile
    int * np2;

    /// @brief Secondary tile particle position on global array
    int * offset2;

    /// @brief Number of particles in index list
    int * nidx;

    /// @brief Number of particles leaving tile in all directions
    int * npt;

} t_part_tiles;

class Particles {
    
    private:

    device::Var<unsigned int> _dev_tmp_uint;
    unsigned int max_np_tile;

    public:

    /// @brief Sets periodic boundaries (x,y)
    int2 periodic;

    /// @brief Number of tiles (x,y)
    uint2 ntiles;
    
    /**
     * @brief Tile grid size (x,y)
     * 
     * Valid ix is in the range 0 .. nx-1
     */
    uint2 nx;

    /// @brief Tile information
    t_part_tiles tiles;

    /// @brief Particle data
    t_part_data data;

    /**
     * @brief Indices of particles moving to another tile
     * 
     * (Device data)
     */
    int *idx;



    __host__
    Particles( uint2 const ntiles, uint2 const nx, unsigned int const max_np_tile );

    __host__
    ~Particles() {
        free_dev( data.u );
        free_dev( data.x );
        free_dev( data.ix );
        free_dev( idx );

        free_dev( tiles.nidx );
        free_dev( tiles.offset );
        free_dev( tiles.offset2 );
        free_dev( tiles.np );
        free_dev( tiles.np2 );
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

    // in-place large memory tile sort
    void tile_sort_mk4( Particles &tmp );

    // in-place low memory tile sort
    void tile_sort( Particles &tmp, bool offset_np2 );

    // out of place tile sort (used when growing buffer)
    void tile_sort_mk2( Particles &tmp );

    __host__
    void save( zdf::part_info &info, zdf::iteration &iter, std::string path );

    __host__
    /**
     * @brief Prints out 
     * 
     * @param d_n 
     */
    inline void tile_info( int * d_n ) {
        int * h_n;
        size_t size = ntiles.x * ntiles.y;
        malloc_host( h_n, size );
        devhost_memcpy( h_n, d_n, size );
        for( int j = 0; j < ntiles.y; j++ ) {
            printf("%5d | ", j );
            for( int i = 0; i < ntiles.x; i++ ) {
                int tid = i + j * ntiles.x;
                printf(" %5d", h_n[tid]);
            }
            printf("\n");
        }
        free_host( h_n );
    };
};



#endif