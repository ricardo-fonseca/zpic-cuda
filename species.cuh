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
#include "udist.cuh"

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

namespace species {
    enum pusher { boris, euler };
    namespace bc {
        enum type { open = 0, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;

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

    float q;

    // Secondary data buffer to speed up some calculations
    Particles *tmp;

    // Mass over charge ratio
    float m_q;

    // Cell and simulation box size
    float2 dx, box;

    // Time step
    float dt;

    /// @brief Boundary condition
    species::bc_type bc;

    __host__
    /**
     * @brief Process boundary conditions
     * 
     */
    void process_bc();

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

    double *d_energy;

public:

    // Iteration
    int iter;

    // Species name
    std::string name;

    // Particle data buffer
    Particles *particles;

    species::pusher push_type;

    __host__
    Species( std::string const name, float const m_q, 
        uint2 const ppc, Density::Profile const & density,
        float2 const box, uint2 const ntiles, uint2 const nx,
        const float dt );
    
    __host__
    ~Species();

    __host__
    species::bc_type get_bc( ) { return bc; }

    __host__
    void set_bc( species::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == species::bc::periodic) || (new_bc.x.upper == species::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Species boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to species::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == species::bc::periodic) || (new_bc.y.upper == species::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) Species boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to species::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;


        std::string bc_name[] = {"open", "periodic", "reflecting"};
        std::cout << "(*info*) Species " << name << " boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";

        // Set periodic flags on tile grids
        particles->periodic.x = ( bc.x.lower == species::bc::periodic );
        particles->periodic.y = ( bc.y.lower == species::bc::periodic );
    }

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

            bc.x.lower = bc.x.upper = species::bc::open;
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
    void set_udist( UDistribution::Type const & udist, unsigned int seed ) { udist.set(*particles, seed );};

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
    double get_energy() const {
        
        // Get energy from device memory
        double h_energy;
        devhost_memcpy( &h_energy, d_energy, 1 );

        // Normalize and return
        return h_energy * q * m_q * dx.x * dx.y;
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