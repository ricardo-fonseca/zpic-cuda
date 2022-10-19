#ifndef __EMF__
#define __EMF__

#include "vector_field.cuh"
#include "laser.cuh"
#include "current.cuh"
#include "util.cuh"
#include "moving_window.cuh"

namespace emf {
    enum field  { e, b };

    namespace bc {
        enum type { none, periodic, pec, pmc };
    }

    typedef bnd<bc::type> bc_type;
}

class EMF {

    /// @brief Boundary condition
    emf::bc_type bc;

    /// @brief Simulation box size
    float2 box;

    /// @brief cell size
    float2 dx;

    /// @brief time step
    float dt;

    /// @brief Iteration number
    int iter;

    /// @brief Moving window information
    MovingWindow moving_window;

    __host__
    /**
     * @brief Move simulation window if needed
     * 
     */
    void move_window();

    __host__
    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    /// @brief Device buffer for field energy calculations
    double * d_energy;

    public:

    /// @brief Electric field
    VectorField* E;
    /// @brief Magnetic field
    VectorField* B;

    __host__
    /**
     * @brief Construct a new EMF object
     * 
     * @param ntiles    Number of tiles in x,y direction
     * @param nx        Tile size (#cells)
     * @param box       Simulation box size (sim. units)
     * @param dt        Time step
     */
    EMF( uint2 const ntiles, uint2 const nx, float2 const box, float const dt );
    
    __host__
    /**
     * @brief Destroy the EMF object
     * 
     */
    ~EMF();

    __host__
    emf::bc_type get_bc( ) { return bc; }

    __host__
    void set_bc( emf::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == emf::bc::periodic) || (new_bc.x.upper == emf::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) EMF boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == emf::bc::periodic) || (new_bc.y.upper == emf::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) EMF boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;

        // Set periodic flags on tile grids
        E->periodic.x = B->periodic.x = ( bc.x.lower == emf::bc::periodic );
        E->periodic.y = B->periodic.y = ( bc.y.lower == emf::bc::periodic );
    }

    __host__
    /**
     * @brief Sets moving window algorithm
     * 
     * This method can only be called before the simulation has started (iter = 0)
     * 
     * @return int  0 on success, -1 on error
     */
    int set_moving_window() { 
        if ( iter == 0 ) {
            moving_window.init( dx.x );

            E->periodic.x = false;
            B->periodic.x = false;

            bc.x.lower = emf::bc::none;
            bc.x.upper = emf::bc::none;

            return 0;
        } else {
            std::cerr << "(*error*) set_moving_window() called with iter != 0\n";
            return -1; 
        }
    }

    __host__
    void advance( Current & current );
    __host__
    void advance( );

    __host__ void add_laser( Laser::Pulse & laser );

    __host__ void save( emf::field const field, const fcomp::cart fc );
    __host__ void get_energy( double3 & ene_E, double3 & ene_b );

};

#endif