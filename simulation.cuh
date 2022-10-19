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

    void set_moving_window() {
        emf -> set_moving_window();
        current -> set_moving_window();
        for (int i = 0; i < species.size(); i++)
            species[i]->set_moving_window();
    }

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
        Density::Profile const & dens, UDistribution::Type const & udist )
    {
        Species * s = new Species( 
            name, m_q, ppc, dens, 
            box, ntiles, nx, dt );
        species.push_back( s );

        // Inject particles
        s -> inject( );

        // Set momentum distribution
        s -> set_udist( udist, species.size() );
    }

    /**
     * @brief Gets a pointer to a specific species object
     * 
     * @param name                Species name
     * @return Species const* 
     */
    Species * get_species( std::string name ) {
        int id = 0;
        for( id = 0; id < species.size(); id++ )
            if ( (species[id])->name == name ) break;
        return ( id < species.size() ) ? species[id] : nullptr;
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

    void energy_info() {
        std::cout << "(*info*) Energy at n = " << iter << ", t = " << iter * double(dt)  << "\n";
        double part_ene = 0;
        for (int i = 0; i < species.size(); i++) {
            double kin = species[i]->get_energy();
            std::cout << "(*info*) " << species[i]->name << " = " << kin << "\n";
            part_ene += kin;
        }

        if ( species.size() > 1 )
            std::cout << "(*info*) Total particle energy = " << part_ene << "\n";

        double3 ene_E, ene_B;
        emf -> get_energy( ene_E, ene_B );
        std::cout << "(*info*) Electric field = " << ene_E.x + ene_E.y + ene_E.z << "\n";
        std::cout << "(*info*) Magnetic field = " << ene_B.x + ene_B.y + ene_B.z << "\n";

        double total = part_ene + ene_E.x + ene_E.y + ene_E.z + ene_B.x + ene_B.y + ene_B.z;
        std::cout << "(*info*) total = " << total << "\n";
    }
};


#endif
