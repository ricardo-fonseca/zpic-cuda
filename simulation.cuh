#ifndef __SIMULATION__
#define __SIMULATION__

#include "emf.cuh"
#include "current.cuh"
#include "species.cuh"

#include "timer.cuh"
#include "util.cuh"

#include <vector>

class Simulation {

    private:

    uint2 ntiles;
    uint2 nx;
    float2 box;
    float dt;
    unsigned int iter;

    public:

    EMF * emf;
    Current * current;
    std::vector <Species*> species;

    /**
     * @brief Construct a new Simulation object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Tile grid size
     * @param box       Simulation box size
     * @param dt        Time step
     */
    Simulation( uint2 const ntiles, uint2 const nx, float2 const box, float dt ):
        ntiles( ntiles ), nx( nx ), box( box ), dt( dt ), iter(0) {

        emf = new EMF( ntiles, nx, box, dt );
        current = new Current( ntiles, nx, box, dt );
    }

    /**
     * @brief Destroy the Simulation object
     * 
     */
    ~Simulation() {
        delete current;
        delete emf;

        for (int i = 0; i < species.size(); i++)
            delete species[i];
        
        // Check for any device errors that went under the radar
        auto err_sync = cudaPeekAtLastError();
        auto err_async = cudaDeviceSynchronize();
        if (( err_sync != cudaSuccess ) || ( err_async != cudaSuccess )) {
            std::cerr << "(*error*) CUDA device in on an error state:\n";
            if ( err_sync != cudaSuccess )
                std::cerr << "(*error*) Sync. error message: " << cudaGetErrorString(err_sync)
                    << " (" << err_sync << ") \n";
            if ( err_async != cudaSuccess )
                std::cerr << "(*error*) Async. error message: " << cudaGetErrorString(err_async)
                     << " (" << err_async << ") \n";
            exit(1);
        }
    };

    /**
     * @brief Adds a new particle species to the simulation object
     * 
     * @param name      Species name
     * @param m_q       mass over charge ratio
     * @param ppc       number of particles per cell
     * @param n0        Reference density value
     * @param uth       Initial thermal velocity
     * @param ufl       Initial fluid velocity
     */
    void add_species( std::string const name, float const m_q, uint2 const ppc,
        float const n0, density::parameters const & dens, float3 const uth, float3 const ufl ) {
        Species * s = new Species( name, m_q, ppc, n0,
            box, ntiles, nx, dt );
        species.push_back( s );

        s -> inject_particles( dens );
        s -> set_u( uth, ufl );
    }

    /**
     * @brief Gets a pointer to a specific species object
     * 
     * @param id                Species ID
     * @return Species const* 
     */
    Species const * get_species( int const id ) {
        return ( id >= 0 && id < species.size()) ? species[id] : nullptr;
    }

    Species const * get_species( std::string name ) {
        int id = 0;
        for( id = 0; id < species.size(); id++ )
            if ( (species[id])->name == name ) break;
        return ( id < species.size() ) ? species[id] : nullptr;
    }

    void add_laser( Laser & laser ) {
        emf -> add_laser( laser );
    }

    void advance() {

        // Zero global current
        current -> zero();

        // Advance aoo species
        for (int i = 0; i < species.size(); i++) {
            species[i] -> advance( *emf, *current );
        }

        // Update current edge values and guard cells
        current -> advance();
        
        // Advance EM fields
        emf -> advance( *current );

        iter++;
    }

    auto   get_box() { return box; };
    auto   get_iter() { return iter; };
    auto   get_dt() { return dt; };
    double get_t() { return iter * double(dt); };
};


#endif