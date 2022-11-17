#include "cathode.cuh"
#include "random.cuh"

/**
 * ## Cathode algorithm
 * 
 * To minimize differences from a free streaming species with the same (fluid)
 * velocity, this implementation of the cathode stores `ppc.x` longitudinal
 * positions:
 * 1. These positions are initialized as if injecting a uniform density profile
 * 2. At each time-step they are advanced as if they are a free-streaming 
 *    particle species, i.e., positions are moved by v * dt / dx
 * 4. The complete set of positions is copied onto device memory; all positions
 *    >= 0.5 (for a lower edge cathode) correspond to particles that must be
 *    injected.
 * 5. Reference positions are then trimmed 
 * 
 * While this represents an overhead compared to the previous algorithm, it
 * ensures a much steadier flow of particles. Currently, the only limitation of
 * this implementation is that the algorithm requires a shared memory array of
 * `ppc.x` floats.
 */


__global__
/**
 * @brief CUDA kernel for initializing injection positions in device memeory
 * 
 * @param d_inj_pos 
 * @param ppc_x 
 */
void _cathode_init( float * d_inj_pos, int ppc_x ) {
    auto block = cg::this_thread_block();

    double dpcx = 1.0  / ppc_x;
    for( int i = block.thread_rank(); i < ppc_x; i += block.num_threads() ) {
        d_inj_pos[i] = dpcx * ( i + 0.5 ) - 0.5;
    }
}


/**
 * @brief Construct a new Cathode:: Cathode object
 * 
 * @param name      Cathode name
 * @param m_q       Mass over charge ratio
 * @param ppc       Number of particles per cell
 * @param ufl       Fluid velocity
 */
Cathode::Cathode( std::string const name, float const m_q, uint2 const ppc, float ufl ):
    Species( name, m_q, ppc ), ufl( abs(ufl) )
{ 
    if ( ufl == 0 ) {
        std::cerr << "(*error*) Cathodes cannot have ufl = 0, aborting...\n";
        exit(1);
    }

    // Default values
    wall  = edge::lower;
    start = 0;
    end   = std::numeric_limits<float>::infinity();
    uth   = make_float3( 0, 0, 0 );
    n0    = 1.0;
}

void Cathode::initialize( float2 const box_, uint2 const ntiles, uint2 const nx,
    float const dt_, int const id_ ) {

    // Cathode velocity (always > 0)
    vel = (ufl / sqrtf( ufl * ufl + 1.0f )) ;

    // Initialize position of cathode particles in the cell outside the box
    malloc_dev( d_inj_pos, ppc.x );
    _cathode_init <<< 1, ppc.x >>> ( d_inj_pos, ppc.x );

    // Complete species initialization
    Species::set_udist( UDistribution::Thermal( uth, make_float3( ufl, 0, 0 ) ) );
    Species::set_density( Density::None( n0 ) );
    Species::initialize( box_, ntiles, nx, dt_, id_ );

}


Cathode::~Cathode() {
    free_dev( d_inj_pos );
}

/**
 * @brief Inject particles inside the simulation box
 * 
 * This will only happen if iter == 0 and start < 0
 * This also sets the velocity of the injected particles
 */
void Cathode::inject() {
    
    if ( iter == 0 && start < 0 ) {

        uint2 g_nx = particles -> g_nx();
        bnd<unsigned int> range;
        range.x = { .lower = 0, .upper = g_nx.x - 1 };
        range.y = { .lower = 0, .upper = g_nx.y - 1 };

        Cathode::inject( range );
    }
}

/**
 * @brief Inject particles inside the specified range
 * 
 * @param range     Cell range in which to inject
 */
void Cathode::inject( bnd<unsigned int> range ) {

    if ( iter == 0 && start < 0 ) {
        float x0, x1, u;

        switch (wall)
        {
        case edge::lower:
            x0 = 0;
            x1 = -vel * start;
            u = ufl;
            break;
        
        case edge::upper:
            x0 = box.x + vel * start;
            x1 = box.x;
            u = - ufl;
            break;
        }

        Density::Slab cathode_density( coord::x, n0, x0, x1 );
        cathode_density.inject( particles, ppc, dx, make_float2(0,0), range );

        UDistribution::Thermal udist( uth, make_float3( u, 0, 0 ) );
        udist.set( *particles, id );
    }
}

__global__
/**
 * @brief CUDA kernel for updating injection positions (lower boundary injection)
 * 
 * @param d_inj_pos     Pointer to injection positions
 * @param ppc_x         Number of particles per cell along injection direction
 * @param motion        Particle motion ( vel * dt / dx )
 */
void _update_cathode_lower( float * const d_inj_pos, unsigned int ppc_x, float motion ) {
    auto block = cg::this_thread_block();

    for( int i = block.thread_rank(); i < ppc_x; i+= block.num_threads() ) {
        float x = d_inj_pos[i];
        if ( x >= 0.5f ) x -= 1.0f;
        d_inj_pos[i] = x + motion;
    }
}


__global__
/**
 * @brief CUDA kernel for updating injection positions (upper boundary injection)
 * 
 * @param d_inj_pos     Pointer to injection positions
 * @param ppc_x         Number of particles per cell along injection direction
 * @param motion        Particle motion ( vel * dt / dx > 0 )
 */
void _update_cathode_upper( float * const d_inj_pos, unsigned int ppc_x, float motion ) {
    auto block = cg::this_thread_block();

    for( int i = block.thread_rank(); i < ppc_x; i+= block.num_threads() ) {
        float x = d_inj_pos[i];
        if ( x < -0.5f ) x += 1.0f;
        d_inj_pos[i] = x - motion;
    }
}

__global__
/**
 * @brief CUDA kernel for injecting cathode particles from lower wall
 * 
 * @param d_inj_pos     Pointer to injection positions
 * @param ufl           Cathode flow generalized velocity
 * @param uth           Temperature for injected particles
 * @param seed          Seed for RNG
 * @param ppc           Number of particles per cell
 * @param ntiles        Number of tiles
 * @param nx            Tile grid size
 * @param d_tiles       Tile information (main buffer)
 * @param d_ix          Particle cells (main buffer)
 * @param d_x           Particle positions  (main buffer)
 * @param d_u           Particle generalized velocity (main buffer)
 */
void _inject_cathode_lower( 
    float * const d_inj_pos, float const ufl,
    float3 uth, uint2 seed,  uint2 const ppc,
    uint2 const ntiles, uint2 const nx, 
    int * const __restrict__ d_tile_np,
    int * const __restrict__ d_tile_offset,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    extern __shared__ float inj_pos[ ];

    int const tid = blockIdx.y * ntiles.x;

    // Initialize random state variables
    uint2 state;
    double norm;
    rand_init( seed, state, norm );

    const int offset =  d_tile_offset[ tid ];
    int2   * __restrict__ const ix = &d_ix[ offset ];
    float2 * __restrict__ const x  = &d_x[ offset ];
    float3 * __restrict__ const u  = &d_u[ offset ];

    int np = d_tile_np[ tid ];

    // Advance injection positions and count number of particles to inject

    __shared__ unsigned int _inj_np;
    _inj_np = 0;

    block.sync();

    unsigned int inj_np = 0;
    for( int idx = threadIdx.x; idx < ppc.x; idx += blockDim.x ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] >= 0.5f ) inj_np++;
    }

    inj_np = cg::reduce( warp, inj_np, cg::plus<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( &_inj_np, inj_np);

    block.sync();

    inj_np = _inj_np;

    // Inject particles
    double dpcy = 1.0 / ppc.y;

    // 1 thread per cell
    for( int idx = threadIdx.x; idx < nx.y; idx += blockDim.x ) {
        int2 const cell = make_int2( 0, idx );

        int part_idx = np + idx * ( inj_np * ppc.y );

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                if ( inj_pos[i0] >= 0.5f ) {
                    float2 const pos = make_float2(
                        inj_pos[i0] - 1.0f,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3( 
                        ufl + uth.x * rand_norm( state, norm ),
                              uth.y * rand_norm( state, norm ),
                              uth.z * rand_norm( state, norm )
                    );
                    part_idx++;
                }
            }
        }
    }

    // Update global number of particles in tile
    if ( threadIdx.x == 0 )
        d_tile_np[ tid ] += nx.y * inj_np * ppc.y;
}

__global__
/**
 * @brief CUDA kernel for injecting cathode particles from lower wall
 * 
 * @param d_inj_pos     Pointer to injection positions
 * @param ufl           Cathode flow generalized velocity
 * @param uth           Temperature for injected particles
 * @param seed          Seed for RNG
 * @param ppc           Number of particles per cell
 * @param ntiles        Number of tiles
 * @param nx            Tile grid size
 * @param d_tiles       Tile information (main buffer)
 * @param d_ix          Particle cells (main buffer)
 * @param d_x           Particle positions  (main buffer)
 * @param d_u           Particle generalized velocity (main buffer)
 */
void _inject_cathode_upper( 
    float * const d_inj_pos, float const ufl,
    float3 uth, uint2 seed,  uint2 const ppc,
    uint2 const ntiles, uint2 const nx, 
    int * const __restrict__ d_tile_np,
    int * const __restrict__ d_tile_offset,
    int2 * __restrict__ d_ix, float2 * __restrict__ d_x, float3 * __restrict__ d_u )
{
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    extern __shared__ float inj_pos[ ];

    int const tid = blockIdx.y * ntiles.x + (ntiles.x-1);

    // Initialize random state variables
    uint2 state;
    double norm;
    rand_init( seed, state, norm );

    const int offset =  d_tile_offset[ tid ];
    int2   * __restrict__ const ix = &d_ix[ offset ];
    float2 * __restrict__ const x  = &d_x[ offset ];
    float3 * __restrict__ const u  = &d_u[ offset ];

    int np = d_tile_np[ tid ];

    // Advance injection positions and count number of particles to inject

    __shared__ unsigned int _inj_np;
    _inj_np = 0;

    block.sync();

    unsigned int inj_np = 0;
    for( int idx = threadIdx.x; idx < ppc.x; idx += blockDim.x ) {
        inj_pos[idx] = d_inj_pos[idx];
        if ( inj_pos[idx] < -0.5f ) inj_np++;
    }

    inj_np = cg::reduce( warp, inj_np, cg::plus<unsigned int>());
    if ( warp.thread_rank() == 0 ) atomicAdd( &_inj_np, inj_np);

    block.sync();

    inj_np = _inj_np;

    // Inject particles
    double dpcy = 1.0 / ppc.y;

    // 1 thread per cell
    for( int idx = threadIdx.x; idx < nx.y; idx += blockDim.x ) {
        int2 const cell = make_int2( nx.x-1, idx );

        int part_idx = np + idx * ( inj_np * ppc.y );

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                if ( inj_pos[i0] < -0.5f ) {
                    float2 const pos = make_float2(
                        inj_pos[i0] + 1.0f,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3( 
                        -ufl + uth.x * rand_norm( state, norm ),
                               uth.y * rand_norm( state, norm ),
                               uth.z * rand_norm( state, norm )
                    );
                    part_idx++;
                }
            }
        }
    }

    // Update global number of particles in tile
    if ( threadIdx.x == 0 )
        d_tile_np[ tid ] += nx.y * inj_np * ppc.y;
}

/**
 * @brief Advance cathode species.
 * 
 * Advances existing particles and injects new particles if needed.
 * 
 * @param emf 
 * @param current 
 */
void Cathode::advance( EMF const &emf, Current &current ) 
{

    // Advance particles using superclass methods
    Species::advance( emf, current );

    double t = ( iter - 1 ) * dt;
    if (( t >= start ) && ( t < end ) ) { 

        dim3 grid( 1, particles -> ntiles.y );
        dim3 block( 32 );
        size_t shm_size = ppc.x * sizeof( float );

        uint2 rnd_seed = {12345 + (unsigned int) iter, 67890 + (unsigned int ) id };

        switch (wall)
        {
        case edge::lower:
            _update_cathode_lower <<< 1, ppc.x >>> ( d_inj_pos, ppc.x, vel * dt / dx.x );

            _inject_cathode_lower <<< grid, block, shm_size >>> (
                d_inj_pos, ufl, uth, rnd_seed, ppc, 
                particles -> ntiles, particles -> nx,
                particles -> tile_np, particles -> tile_offset, 
                particles -> ix, particles -> x, particles -> u
            );
            break;
        
        case edge::upper:
            _update_cathode_upper <<< 1, ppc.x >>> ( d_inj_pos, ppc.x, vel * dt / dx.x );

            _inject_cathode_upper <<< grid, block, shm_size >>> (
                d_inj_pos, ufl, uth, rnd_seed, ppc, 
                particles -> ntiles, particles -> nx, 
                particles -> tile_np, particles -> tile_offset, 
                particles -> ix, particles -> x, particles -> u
            );
            break;
        }

    }
}