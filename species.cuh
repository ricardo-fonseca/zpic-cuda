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

#include "density.cuh"
#include "moving_window.cuh"

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
            name = "ux"; label = "u_x"; units = "c";
            break;
        case uy :
            name = "uy"; label = "u_y"; units = "c";
            break;
        case uz :
            name = "uz"; label = "u_y"; units = "c";
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

    uint2 ppc;

    float n0;

    Density::Profile * density;

    // Total kinetic energy per tile
    double * d_energy_tile;

    float q;

    // Secondary data buffer to speed up some calculations
    Particles *tmp;

    // Mass over charge ratio
    float m_q;

    // Cell and simulation box size
    float2 dx, box;

    // Time step
    float dt;

    // Moving window information
    MovingWindow moving_window;

    __host__
    void move_window_shift();

    __host__
    void move_window_inject();

    __host__
    void dep_phasespace( float * const d_data, 
        phasespace::quant q, float2 const range, unsigned const size ) const;

    __host__
    void dep_phasespace( float * const d_data,
        phasespace::quant quant0, float2 range0, unsigned const size0,
        phasespace::quant quant1, float2 range1, unsigned const size1 ) const;

public:

    // Iteration
    int iter;

    // Species name
    std::string name;

    // Particle data buffer
    Particles *particles;

    __host__
    Species( std::string const name, float const m_q, 
        uint2 const ppc, Density::Profile const & density,
        float2 const box, uint2 const ntiles, uint2 const nx,
        const float dt );
    
    __host__
    ~Species();

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
            particles->periodic.x = false;
            return 0;
        } else {
            std::cerr << "(*error*) set_moving_window() called with iter != 0\n";
            return -1; 
        }
    }

    __host__
    void inject();
    void inject( bnd<unsigned int> range );

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
    void save_phasespace ( 
        phasespace::quant quant, float2 const range, unsigned size ) const;

    __host__
    void save_phasespace ( 
        phasespace::quant quant0, float2 const range0, unsigned size0,
        phasespace::quant quant1, float2 const range1, unsigned size1 ) const;

    __host__
    void push( VectorField * const E, VectorField * const B );

    __host__
    void move( VectorField * const current );

    __host__
    void move( );

};


#endif