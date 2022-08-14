#ifndef __CURRENT__
#define __CURRENT__

#include "vector_field.cuh"
#include "util.cuh"

class Current {

    public:

    // Current density
    VectorField * J;
            
    // Simulation box info
    float2 box;
    float2 dx;

    // Time step
    float dt;

    // Filtering parameters
    //Filter filter;

    // Iteration number
    int iter;

    __host__ Current( uint2 const ntiles, uint2 const nx, float2 const box, float const dt );
    __host__ ~Current();

    __host__ void advance();
    __host__ void zero();
    __host__ void save( fcomp::cart const jc );

};


#endif