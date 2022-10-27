#ifndef __CURRENT__
#define __CURRENT__

#include "vector_field.cuh"
#include "util.cuh"
#include "moving_window.cuh"
#include "filter.cuh"

namespace current {
    namespace bc {
        enum type { none = 0, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;
}

class Current {

    /// @brief Simulation box size
    float2 box;

    /// @brief cell size
    float2 dx;
    
    /// @brief time step
    float dt;

    /// @brief Moving window information
    MovingWindow moving_window;

    /// @brief Boundary condition
    current::bc_type bc;

    /// @brief Iteration number
    int iter;

    __host__
    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

    public:

    /// @brief Current density
    VectorField * J;

    /// @brief Filtering parameters
    Filter::Digital *filter;

    __host__ Current( uint2 const ntiles, uint2 const nx, float2 const box, float const dt );
    __host__ ~Current() {
        delete (J);
        delete (filter);
    }

    __host__
    current::bc_type get_bc( ) { return bc; }

    __host__
    void set_bc( current::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == current::bc::periodic) || (new_bc.x.upper == current::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Current boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to current::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == current::bc::periodic) || (new_bc.y.upper == current::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) Current boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to emf::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;


        std::string bc_name[] = {"none", "periodic", "reflecting"};
        std::cout << "(*info*) Current boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";

        // Set periodic flags on tile grids
        J->periodic.x = ( bc.x.lower == current::bc::periodic );
        J->periodic.y = ( bc.y.lower == current::bc::periodic );
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

            bc.x.lower = bc.x.upper = current::bc::none;
            J->periodic.x = false;

            return 0;
        } else {
            std::cerr << "(*error*) set_moving_window() called with iter != 0\n";
            return -1; 
        }
    }

    __host__
    void set_filter( Filter::Digital const & new_filter ) {
        delete filter;
        filter = new_filter.clone();
    }

    __host__ void advance();
    __host__ void zero();
    __host__ void save( fcomp::cart const jc );

};


#endif