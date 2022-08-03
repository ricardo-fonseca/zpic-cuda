#ifndef __EMF__
#define __EMF__

#include "tile_vfld.cuh"
#include "laser.cuh"

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

    // Iteration number (device / host)
    int d_iter, h_iter;

    __host__ EMF( const int2 gnx, const int2 tnx, const float2 box, const float dt );
    __host__ ~EMF();

    __host__ void update_data( const VFLD::copy_direction direction );

    __host__ void advance();

    __host__ void add_laser( Laser& laser );

    enum diag_fld { EFLD, BFLD };
    __host__ void report( const diag_fld field, const int fc );
    __host__ void get_energy( double energy[6] );

};

#endif