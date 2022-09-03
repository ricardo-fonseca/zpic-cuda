#ifndef __EMF__
#define __EMF__

#include "vector_field.cuh"
#include "laser.cuh"
#include "current.cuh"
#include "util.cuh"
#include "moving_window.cuh"

namespace emf {
    enum field  { e, b };
}

class EMF {

    // Simulation box info
    float2 box;
    float2 dx;

    // time step
    float dt;

    // Iteration number
    int iter;

    // Moving window information
    MovingWindow moving_window;

    __host__
    void move_window();

    public:

    // Electric and magnetic fields
    VectorField* E;
    VectorField* B;

    __host__ EMF( uint2 const ntiles, uint2 const nx, float2 const box, float const dt );
    __host__ ~EMF();

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
    __host__ void get_energy( double energy[6] );

};

#endif