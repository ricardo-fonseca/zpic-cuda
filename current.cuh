#ifndef __CURRENT__
#define __CURRENT__

#include "tile_vfld.cuh"

class Current {

    public:

    // Current density
    VFLD* J;
            
    // Simulation box info
    float2 box;
    float2 dx;

    // Time step
    float dt;

    // Filtering parameters
    //Filter filter;

    // Iteration number (device / host)
    int d_iter, h_iter;

    __host__ Current( const int2 gnx, const int2 tnx, const float2 box, const float dt );
    __host__ ~Current();

    __host__ void update_data( const VFLD::copy_direction direction );

    __host__ void advance();
    __host__ void zero();
    __host__ void report( const int jc );

};


#endif