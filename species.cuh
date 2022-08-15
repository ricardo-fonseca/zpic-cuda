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
        parameters d = { .type = type::sphere, .pos = center, .radius = radius };
        return d;
    }
}

namespace phasespace {
    enum quant { x, y, ux, uy, uz };

    static inline void qinfo( quant q, std::string & name, std::string & label, std::string & units ) {
        switch(q) {
        case x :
            name = "x"; label = "x"; units = "c/\\omega_n";
            break;
        case y :
            name = "y"; label = "y"; units = "c/\\omega_n";
            break;
        case ux :
            name = "ux"; label = "u_x"; units = "m_e c";
            break;
        case uy :
            name = "uy"; label = "u_y"; units = "m_e c";
            break;
        case uz :
            name = "uz"; label = "u_y"; units = "m_e c";
            break;
        }
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

    __host__
    void dep_phasespace( float * const d_data, 
        phasespace::quant q, float2 const range, unsigned const size );

    __host__
    void dep_phasespace( float * const d_data,
        phasespace::quant quant0, float2 range0, unsigned const size0,
        phasespace::quant quant1, float2 range1, unsigned const size1 );

public:

    // Iteration
    int iter;

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
    void save_phasespace( 
        phasespace::quant quant, float2 const range, unsigned size );

    __host__
    void save_phasespace( 
        phasespace::quant quant0, float2 const range0, unsigned size0,
        phasespace::quant quant1, float2 const range1, unsigned size1 );

    __host__
    void push( VectorField * const E, VectorField * const B );

    __host__
    void move( VectorField * const current );

    __host__
    void move( );

};


#endif