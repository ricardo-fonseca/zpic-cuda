#ifndef __PARTICLES__
#define __PARTICLES__

#include <string>
#include "tile_part.cuh"

#include "field.cuh"

#include "emf.cuh"
#include "current.cuh"

class Species {
    public:

    // Species name
    std::string name;

    // Particle data buffer
    TilePart *particles;

    // Mass over charge ratio
    float m_q;

    // Total kinetic energy
    double energy;

    // Charge of individual partilce
    float q;

    // Number of particles per cell
    int2 ppc;

    // Initial density profile
    // Density density;

    // Initial fluid and thermal momenta
    float3 ufl, uth;

    // Cell and simulation box size
    float2 dx,  box;

    // Time step
    float dt;

    // Iteration
    int d_iter, h_iter;

    __host__
    Species( const std::string name, const float m_q, const int2 ppc,
        const float n0, const float3 ufl, const float3 uth,
        const float2 box, const int2 gnx, const int2 tnx, const float dt );
    
    __host__
    ~Species();

    __host__
    void inject_particles( );

    __host__
    void set_u();

    __host__
    void advance( EMF &emf, Current &current );

    __host__
    void move_deposit( VFLD &current );

    __host__
    void deposit_charge( Field &charge );

    __host__
    void deposit_phasespace( const int rep_type, const int2 pha_nx, const float2 pha_range[2],
        float buf[]);
        
    __host__
    void report( const int rep_type, const int2 pha_nx, const float2 pha_range[2]);

    __host__
    void save_particles();

    __host__
    void save_charge();

};


#endif