/**
 * @file species.cuh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __SPECIES__
#define __SPECIES__

#include <string>
#include "particles.cuh"

#include "field.cuh"

#include "emf.cuh"
#include "current.cuh"

class Species {
    public:

    // Species name
    std::string name;

    // Particle data buffer
    Particles *particles;

    // Secondary data buffer to speed up some calculations
    Particles *tmp;

    // Mass over charge ratio
    float m_q;

    // Total kinetic energy
    double energy;

    // Charge of individual partilce
    float q;

    // Number of particles per cell
    uint2 ppc;

    // Cell and simulation box size
    float2 dx,  box;

    // Time step
    float dt;

    // Iteration
    int iter;

    __host__
    Species( std::string const name, float const m_q, uint2 const ppc,
        float const n0, float3 const uth, float3 const ufl,
        float2 const box, uint2 const ntiles, uint2 const nx, const float dt );
    
    __host__
    ~Species();

    __host__
    void inject_particles( );

    __host__
    void set_u( float3 const uth, float3 const ufl );

    __host__
    void advance( EMF &emf, Current &current );

    __host__
    void push( VFLD &E, VFLD &B );

    __host__
    void move_deposit( VFLD * const current );

    __host__
    void tile_sort( );

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