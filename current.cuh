#ifndef __CURRENT__
#define __CURRENT__

#include "vector_field.cuh"
#include "util.cuh"
#include "moving_window.cuh"
#include "filter.cuh"

class Current {

    // Simulation box info
    float2 box;
    float2 dx;

    // Time step
    float dt;

    // Moving window information
    MovingWindow moving_window;

    public:

    // Current density
    VectorField * J;

    // Filtering parameters
    Filter::Digital *filter;

    // Iteration number
    int iter;

    __host__ Current( uint2 const ntiles, uint2 const nx, float2 const box, float const dt );
    __host__ ~Current() {
        delete (J);
        delete (filter);
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