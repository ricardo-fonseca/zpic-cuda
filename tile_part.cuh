#ifndef __TILE_PART__
#define __TILE_PART__


class TilePart {

    public:


    // Global particle buffers (device/host)
    int2   *d_ix, *h_ix;
    float2 *d_x,  *h_x;
    float3 *d_u,  *h_u;

    // For each tile, position of first particle in global buffer and number of particles in tile
    int2 *d_tile, *h_tile;

    // Size of global buffer
    size_t buffer_max;

    // Number of tiles in each direction
    int2 nxtiles;

    // Tile grid size
    int2 nx;

    // Random number generator
    // Random rnd;

    int d_iter, h_iter;

    TilePart( const int2 global_nx, const int2 tile_nx, const int tile_np_max );
    
    ~TilePart();

    void bnd_sort();

    enum rep_quant { x, y, ux, uy, uz };
    void gather( rep_quant quant, float * __restrict__ data, const int np_max );

    size_t device_np();

    /**
     * @brief Returns global grid size
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
};


#endif