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


/**
 * @brief Density parameters
 * 
 */
namespace density {
    
    namespace type {
        enum dens { uniform, step, slab, sphere };
    }

    typedef struct parameters {
        type::dens type;
        float2 pos;
        float radius;
    } t_parameters;

    static inline parameters uniform( ) { 
        parameters d = { .type = type::uniform };
        return d;
    }

    static inline parameters step( float const start ) { 
        parameters d = { .type = type::step, .pos = make_float2(start,0) };
        return d;
    }

    static inline parameters slab( float const start, float const finish ) { 
        parameters d = { .type = type::slab, .pos = make_float2(start,finish) };
        return d;
    }

    static inline parameters sphere( float2 const center, float const radius ) { 
        parameters d = { .type = type::slab, .pos = center, .radius = radius };
        return d;
    }
}

/**
 * @brief Charged particles class
 * 
 */
class Species {

private:

    float n0;

    uint2 ppc;


    // Total kinetic energy per tile
    double * d_energy_tile;

    float q;

    // Secondary data buffer to speed up some calculations
    Particles *tmp;

    // Mass over charge ratio
    float m_q;

    // Cell and simulation box size
    float2 dx,  box;

    // Time step
    float dt;

    // Iteration
    int iter;


public:

    // Species name
    std::string name;

    // Particle data buffer
    Particles *particles;

    __host__
    Species( std::string const name, float const m_q, uint2 const ppc, float const n0,
        float2 const box, uint2 const ntiles, uint2 const nx, const float dt );
    
    __host__
    ~Species();

    __host__
    void inject_particles( density::parameters const & dens );

    __host__
    void set_u( float3 const uth, float3 const ufl );

    __host__
    void advance( EMF const &emf, Current &current );

    __host__
    void deposit_charge( Field &charge ) const;

    __host__
    void deposit_phasespace( const int rep_type, const int2 pha_nx, const float2 pha_range[2],
        float buf[]);
        
    __host__
    void report( const int rep_type, const int2 pha_nx, const float2 pha_range[2]);

    __host__
    /**
     * @brief Returns total time centered kinetic energy
     * 
     * Note that this will always trigger a reduction operation of the per tile
     * energy data.
     * 
     * @return double 
     */
    double get_energy() {
        // uint2 ntiles = particles -> ntiles;
        // return device::reduction( d_energy_tile, ntiles.x * ntiles.y );

        std::cerr << "(*warn*) " << __func__ << " not implemented yet." << std::endl;
        return 0;
    }

    __host__
    void save() const;

    __host__
    void save_charge() const;

    __host__
    void push( VFLD * const E, VFLD * const B );

    __host__
    void move( VFLD * const current );

    __host__
    void move( );

};


#endif