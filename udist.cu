#include "udist.cuh"
#include "random.cuh"

__global__
/**
 * @brief CUDA kernel for setting none(frozen) u distribution
 * 
 * @param d_tiles 
 * @param d_u 
 */
void _set_none( t_part_tile const * const __restrict__ d_tiles,
    float3 * const __restrict__ d_u ) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;
    float3 __restrict__ * const u  = &d_u[ offset ];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        u[i] = make_float3(0,0,0);
    }
}

/**
 * @brief Sets none(0 temperature, 0 fluid) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::None::set( Particles & part ) const {

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );
    
    _set_none <<< grid, block >>> ( part.tiles, part.u );
}

__global__
/**
 * @brief CUDA kernel for setting cold u distribution
 * 
 * @param d_tiles 
 * @param d_u 
 * @param ufl 
 */
void _set_cold( t_part_tile const * const __restrict__ d_tiles,
    float3 * const __restrict__ d_u, float3 const ufl ) {

    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;
    float3 __restrict__ * const u  = &d_u[ offset ];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        u[i] = ufl;
    }
}

/**
 * @brief Sets cold(0 temperatures) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::Cold::set( Particles & part ) const {

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );
    
    _set_none <<< grid, block >>> ( part.tiles, part.u );
}


__global__
/**
 * @brief Sets particle momentum
 * 
 * @param d_tile    Tile information
 * @param d_u       Particle buffer (momenta)
 * @param seed      Seed for random number generator
 * @param uth       Thermal distribution width
 * @param ufl       Fluid momentum
 */
void _set_thermal( 
    t_part_tile const * const __restrict__ d_tiles,
    float3 * const __restrict__ d_u, 
    const uint2 seed, const float3 uth, const float3 ufl ) {

    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    rand_init( seed, state, norm );

    // Set particle momenta
    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;
    float3 __restrict__ * const u  = &d_u[ offset ];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        u[i] = make_float3(
            ufl.x + uth.x * rand_norm( state, norm ),
            ufl.y + uth.y * rand_norm( state, norm ),
            ufl.z + uth.z * rand_norm( state, norm )
        );
    }
}

/**
 * @brief Sets momentum of all particles in object using uth / ufl
 * 
 */
void UDistribution::Thermal::set( Particles & part ) const {

    std::cout << "(*info*) Setting 'Thermal' u distribution\n";
    std::cout << "(*info*) uth(" << uth.x << "," << uth.y << "," << uth.z << ")\n";
    std::cout << "(*info*) ufl(" << ufl.x << "," << ufl.y << "," << ufl.z << ")\n";

    // Set thermal momentum
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );
    
    uint2 seed = {12345, 67890};
    _set_thermal <<< grid, block >>> ( 
        part.tiles, part.u, seed, uth, ufl
    );
}

__global__
/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 * 
 * @param d_tile    Tile information
 * @param d_u       Particle buffer (momenta)
 * @param d_ix      Particle buffer (momenta)
 * @param nx        Tile size
 * @param seed      Seed for random number generator
 * @param uth       Thermal distribution width
 * @param ufl       Fluid momentum
 * @param npmin     Minimum number of particles in cell to apply correction
 */
void _set_thermal_corr( 
    t_part_tile const * const __restrict__ d_tiles,
    float3 * const __restrict__ d_u, int2 const * const __restrict__ d_ix, uint2 const nx, 
    uint2 const seed, float3 const uth, float3 const ufl, int const npmin ) {

    auto block = cg::this_thread_block();

    extern __shared__ char buffer[];
    int * const __restrict__ npcell = (int*) buffer;
    float3 * const __restrict__ fluid = (float3*) (buffer + nx.x*nx.y*sizeof(int));

    for( int idx = threadIdx.x; idx < nx.x*nx.y; idx += blockDim.x ) {
        npcell[idx] = 0;
        fluid[idx].x = 0;
        fluid[idx].y = 0;
        fluid[idx].z = 0;
    }

    block.sync();

    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    rand_init( seed, state, norm );

    // Set particle momenta
    const int offset = d_tiles[tid].pos;
    const int np     = d_tiles[tid].n;
    float3 * const __restrict__ u  = &d_u[ offset ];
    int2 const * const __restrict__ ix = &d_ix[offset];

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        float3 upart = make_float3(
            uth.x * rand_norm( state, norm ),
            uth.y * rand_norm( state, norm ),
            uth.z * rand_norm( state, norm )
        );
        u[i] = upart;

        int const idx = ix[i].x + nx.x * ix[i].y;

        atomicAdd( &npcell[ idx ], 1 );
        atomicAdd( &fluid[ idx ].x, upart.x );
        atomicAdd( &fluid[ idx ].y, upart.y );
        atomicAdd( &fluid[ idx ].z, upart.z );
    }

    block.sync();

    for( int idx = threadIdx.x; idx < nx.x*nx.y; idx+= blockDim.x ) {
        if ( npcell[idx] > npmin ) {
            fluid[idx].x /= npcell[idx];
            fluid[idx].y /= npcell[idx];
            fluid[idx].z /= npcell[idx];
        } else {
            fluid[idx] = make_float3(0,0,0);
        }
    }

    block.sync();

    for( int i = threadIdx.x; i < np; i+= blockDim.x ) {
        float3 upart = u[i];
        int const idx = ix[i].x + nx.x * ix[i].y;
        
        upart.x += ufl.x - fluid[idx].x;
        upart.y += ufl.y - fluid[idx].y;
        upart.z += ufl.z - fluid[idx].z;

        u[i] = upart;
    }
}

/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 * 
 */
void UDistribution::ThermalCorr::set( Particles & part ) const {

    // Set thermal momentum
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    size_t shm_size = part.nx.x * part.nx.y * (sizeof(float3) + sizeof(int));
    
    uint2 seed = {12345, 67890};
    _set_thermal_corr <<< grid, block, shm_size >>> ( 
        part.tiles, part.u, part.ix, part.nx, seed, uth, ufl, npmin
    );
}