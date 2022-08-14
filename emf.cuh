#ifndef __EMF__
#define __EMF__

#include "tile_vfld.cuh"
#include "laser.cuh"
#include "current.cuh"
#include "util.cuh"

namespace emf {
    enum field  { e, b };
}

class EMF {
    public:

    // Electric and magnetic fields
    VFLD* E;
    VFLD* B;

    // Simulation box info
    float2 box;
    float2 dx;

    // time step
    float dt;

    // Iteration number
    int iter;

    __host__ EMF( uint2 const ntiles, uint2 const nx, float2 const box, float const dt );
    __host__ ~EMF();

    __host__
    void advance( Current & current );
    __host__
    void advance( );

    __host__ void add_laser( Laser& laser );

    __host__ void save( emf::field const field, const fcomp::cart fc );
    __host__ void get_energy( double energy[6] );

};

#endif