#include "particles.cuh"
#include <iostream>
#include "tile_zdf.cuh"

#include "util.cuh"

/**
 * @brief Construct a new Species:: Species object
 * 
 * @param name 
 * @param m_q 
 * @param ppc 
 * @param n0
 * @param ufl 
 * @param uth 
 * @param gnx 
 * @param tnx 
 * @param dt 
 */
Species::Species(const std::string name, const float m_q, const int2 ppc,
        const float n0, const float3 ufl, const float3 uth,
        const float2 box, const int2 gnx, const int2 tnx, const float dt ) :
        name{name}, m_q{m_q}, ppc{ppc}, ufl{ufl}, uth{uth}, box{box}
{
    std::cout << "(*info*) Initializing species " << name << " ..." << std::endl;

    q = copysign( 1.0f, m_q ) / (ppc.x * ppc.y);

    // Maximum number of particles per tile
    int np_max = tnx.x * tnx.y * ppc.x * ppc.y * 2;
    
    particles = new TilePart( gnx, tnx, np_max );

    // Create RNG object
    // random = 

    // Inject particles
    const int2 range[2] = { 
        {0,0},
        {gnx.x-1, gnx.y-1}
    };
    inject_particles( range );

    // Reset iteration numbers
    d_iter = 0;
    h_iter = -1;

}

/**
 * @brief Destroy the Species:: Species object
 * 
 */
Species::~Species() {

    delete( particles );
    //delete( random );

}

__global__ 
/**
 * @brief Injects particles in tile
 * 
 * Currently only the following is supported:
 * - injecting particles on the whole tile
 * - 0 momenta
 * Buffer overflow is not verified!
 * 
 * @param d_tile 
 * @param buffer_max 
 * @param d_ix 
 * @param d_x 
 * @param ppc 
 */
void _inject_part_kernel(
    int2* __restrict__ d_tile, size_t buffer_max,
    int2* __restrict__ d_ix, float2* __restrict__ d_x, float3* __restrict__ d_u,
    int2 nx, int2 ppc
) {

    // Tile ID
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Pointers to tile particle buffers
    const int offset =  d_tile[ tid ].x;
    int2   __restrict__ *ix = &d_ix[ offset ];
    float2 __restrict__ *x  = &d_x[ offset ];
    float3 __restrict__ *u  = &d_u[ offset ];
    
    const int np = d_tile[ tid ].y;

    const int np_cell = ppc.x * ppc.y;

    double dpcx = 1.0 / ppc.x;
    double dpcy = 1.0 / ppc.y;

    // Each thread takes 1 cell
    for( int idx = threadIdx.x; idx < nx.x*nx.y; idx += blockDim.x ) {
        const int2 cell = { 
            idx % nx.x,
            idx / nx.x
        };

        int part_idx = np + idx * np_cell;

        for( int i1 = 0; i1 < ppc.y; i1++ ) {
            for( int i0 = 0; i0 < ppc.x; i0++) {
                const float2 pos = {
                    static_cast<float>(dpcx * ( i0 + 0.5 )),
                    static_cast<float>(dpcy * ( i1 + 0.5 ))
                };
                ix[ part_idx ] = cell;
                x[ part_idx ] = pos;
                u[ part_idx ] = {0};
                part_idx++;
            }
        }
    }

    // Update global number of particles in tile
    if ( threadIdx.x == 0 ) {
        d_tile[ tid ].y = np + nx.x * nx.y * np_cell ;
    }
}

/**
 * @brief 
 * 
 * @param range 
 */
void Species::inject_particles( const int2 range[2] ) {

    // Only full range injection is currently implemented
    
    // Create particles
    dim3 grid( particles->nxtiles.x, particles->nxtiles.y );
    dim3 block( 64 );

    _inject_part_kernel <<< grid, block >>> ( 
        particles -> d_tile, particles -> buffer_max,
        particles -> d_ix, particles -> d_x, particles -> d_u, 
        particles -> nx, ppc
    );

}

/**
 * @brief 
 * 
 * @param emf 
 * @param current 
 */
void Species::advance( EMF &emf, Current &current ) {

    std::cerr << __func__ << " not implemented yet." << std::endl;

}


void Species::move_window() {

    std::cerr << __func__ << " not implemented yet." << std::endl;

}

/**
 * @brief Deposit phasespace density
 * 
 * @param rep_type 
 * @param pha_nx 
 * @param pha_range 
 * @param buf 
 */
void Species::deposit_phasespace( const int rep_type, const int2 pha_nx, const float2 pha_range[2],
        float buf[]) {

    std::cerr << __func__ << " not implemented yet." << std::endl;

}

__global__
void _dep_charge_kernel( float* __restrict__ d_charge, int offset, int2 int_nx, int2 ext_nx,
    int2* __restrict__ d_tile, int2* __restrict__ d_ix, float2* __restrict__ d_x, const float q ) {

    extern __shared__ float buffer[];

    // Zero shared memory and sync.
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        buffer[i] = 0;
    }
    float *charge = &buffer[ offset ];

    __syncthreads();

    const int tid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int part_off =  d_tile[ tid ].x;
    const int np = d_tile[ tid ].y;
    int2   __restrict__ *ix = &d_ix[ part_off ];
    float2 __restrict__ *x  = &d_x[ part_off ];
    const int nrow = ext_nx.x;

    for( int i = threadIdx.x; i < np; i += blockDim.x ) {
        const int idx = ix[i].y * nrow + ix[i].x;
        const float w1 = x[i].x;
        const float w2 = x[i].y;

        atomicAdd( &charge[ idx            ], ( 1.0f - w1 ) * ( 1.0f - w2 ) * q );
        atomicAdd( &charge[ idx + 1        ], (        w1 ) * ( 1.0f - w2 ) * q );
        atomicAdd( &charge[ idx     + nrow ], ( 1.0f - w1 ) * (        w2 ) * q );
        atomicAdd( &charge[ idx + 1 + nrow ], (        w1 ) * (        w2 ) * q );
    }

    __syncthreads();

    // Copy data to global memory
    const int tile_off = tid * ext_nx.x * ext_nx.y;
    for( int i = threadIdx.x; i < ext_nx.x * ext_nx.y; i += blockDim.x ) {
        d_charge[tile_off + i] = buffer[i];
    } 

}

__host__
/**
 * @brief Deposit charge density
 * 
 * @param charge 
 */
void Species::deposit_charge( Field &charge ) {

    int2 ext_nx = charge.ext_nx();
    dim3 grid( charge.nxtiles.x, charge.nxtiles.y );
    dim3 block( 64 );

    size_t shm_size = ext_nx.x * ext_nx.y * sizeof(float);

    _dep_charge_kernel <<< grid, block, shm_size >>> (
        charge.d_buffer, charge.offset(), charge.nx, ext_nx,
        particles -> d_tile, particles -> d_ix, particles -> d_x, q
    );

}


/**
 * @brief Save particle data to file
 * 
 */
void Species::save_particles( ) {

    const char * quants[] = {
        "x","y",
        "ux","uy","uz"
    };

    const char * qlabels[] = {
        "x","y",
        "u_x","u_y","u_z"
    };

    const char * qunits[] = {
        "c/\\omega_p", "c/\\omega_p",
        "c","c","c"
    };

    t_zdf_iteration iter = {
        .n = d_iter,
        .t = d_iter * dt,
        .time_units = (char *) "1/\\omega_p"
    };

    // Get number of particles on device
    const size_t np = particles->device_np();

    t_zdf_part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .np = np,
        .nquants = 5,
        .quants = (char **) quants,
        .qlabels = (char **) qlabels,
        .qunits = (char **) qunits,
    };

    // Create file and add description
    t_zdf_file part_file;

    std::string path = "PARTICLES/";
    path += name;
    zdf_open_part_file( &part_file, &info, &iter, path.c_str() );

    float * h_data;
    cudaError_t err = cudaMallocHost( &h_data, np * sizeof(float) );
    CHECK_ERR( err, "Failed to allocate host memory for h_data" );
    
    particles -> gather( TilePart::x, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[0], h_data, np );

    particles -> gather( TilePart::y, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[1], h_data, np );

    particles -> gather( TilePart::ux, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[2], h_data, np );

    particles -> gather( TilePart::uy, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[3], h_data, np );

    particles -> gather( TilePart::uz, h_data, np );
    zdf_add_quant_part_file( &part_file, quants[4], h_data, np );

    err = cudaFreeHost( h_data );
    CHECK_ERR( err, "Failed to free host memory for h_data" );

    zdf_close_file( &part_file );
}

/**
 * @brief Saves charge density to file
 * 
 */
void Species::save_charge() {

    const int2 gnx = particles -> g_nx();
    const int2 nx = particles -> nx;
    const int2 gc[2] = {
        {0,0},
        {1,1}
    };

    // Deposit charge on device
    Field charge( gnx, nx, gc );

    // This will also zero the charge density initially
    deposit_charge( charge );

    charge.add_from_gc();

    // Prepare file info
    t_zdf_grid_axis axis[2];
    axis[0] = (t_zdf_grid_axis) {
        .name = (char *) "x",
        .min = 0.,
        .max = box.x,
        .label = (char *) "x",
        .units = (char *) "c/\\omega_p"
    };

    axis[1] = (t_zdf_grid_axis) {
        .name = (char *) "y",
        .min = 0.,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_p"
    };

    std::string grid_name = name + "-charge";
    std::string grid_label = name + " \\rho";

    t_zdf_grid_info info = {
        .name = (char *) grid_name.c_str(),
        .label = (char *) grid_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    info.ndims = 2;
    info.count[0] = gnx.x;
    info.count[1] = gnx.y;

    t_zdf_iteration iter = {
        .name = (char *) "ITERATION",
        .n = d_iter,
        .t = d_iter * dt,
        .time_units = (char *) "1/\\omega_p"
    };

    std::string path = "CHARGE/";
    path += name;
    
    charge.save( info, iter, path.c_str() );
}