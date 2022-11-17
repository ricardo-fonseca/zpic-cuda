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

protected:

     /// @brief Species name
    std::string name;

     /// @brief Unique species identifier
    int id;

     /// @brief  Mass over charge ratio
    float m_q;

    /// @brief Nunber of particles per cell
    uint2 ppc;

    /// @brief reference particle charge
    float q;

    /// @brief Cell dize
    float2 dx;

    /// @brief Simulation box size
    float2 box;

    /// @brief Time step
    float dt;

     /// @brief Iteration
    int iter;

     /// @brief Particle data buffer
    Particles *particles;

     /// @brief Secondary data buffer to speed up some calculations
    Particles *tmp;

private:

    /// @brief Boundary condition
    species::bc_type bc;

     /// @brief Moving window information
    MovingWindow moving_window;

    /// @brief Initial density profile
    Density::Profile * density;

    /// @brief Initial velocity distribution
    UDistribution::Type * udist;

    /// @brief Total species energy on device
    double *d_energy;

    /// @brief Total number of particles moved
    unsigned long long *d_nmove;

    __host__
    /**
     * @brief Process (physical) boundary conditions
     * 
     */
    void process_bc();

    __host__
    /**
     * @brief Shift particle positions due to moving window motion
     * 
     */
    void move_window_shift();

    __host__
    /**
     * @brief Inject new particles due to moving window motion
     * 
     */
    void move_window_inject();

    __host__
    /**
     * @brief Deposit 1D phasespace density
     * 
     * @param d_data    Data buffer
     * @param q         Quantity for axis
     * @param range     Value range
     * @param size      Number of grid points
     */
    void dep_phasespace( float * const d_data, 
        phasespace::quant q, float2 const range, unsigned const size ) const;

    __host__
    /**
     * @brief Deposit 2D phasespace density
     * 
     * @param d_data    Data buffer
     * @param quant0    axis 0 quantity
     * @param range0    axis 0 value range
     * @param size0     axis 0 number of points
     * @param quant1    axis 1 quantity
     * @param range1    axis 1 value range
     * @param size1     axis 1 number of points
     */
    void dep_phasespace( float * const d_data,
        phasespace::quant quant0, float2 range0, unsigned const size0,
        phasespace::quant quant1, float2 range1, unsigned const size1 ) const;

public:

    /// @brief Type of particle pusher to use
    species::pusher push_type;

    __host__
    /**
     * @brief Construct a new Species object
     * 
     * @param name  Name for the species object (used for diagnostics)
     * @param m_q   Mass over charge ratio
     * @param ppc   Number of particles per cell
     */
    Species( std::string const name, float const m_q, uint2 const ppc );

    /**
     * @brief Initialize data structures
     * 
     * @param box       Simulation global box size
     * @param ntiles    Number of tiles
     * @param nx        Title grid dimension
     * @param dt        
     * @param id 
     */
    virtual void initialize( float2 const box, uint2 const ntiles, uint2 const nx,
        float const dt, int const id_ );

    __host__
    /**
     * @brief Destroy the Species object
     * 
     */
    ~Species();

    /**
     * @brief Get the species name
     * 
     * @return std::string 
     */
    auto get_name() { return name; }

    /**
     * @brief Set the density profile object
     * 
     * @param new_density   New density object to be cloned
     */
    virtual void set_density( Density::Profile const & new_density ) {
        delete density;
        density = new_density.clone();
    }

    /**
     * @brief Get the density object
     * 
     * @return Density::Profile& 
     */
    auto & get_density() {
        return * density;
    }

    /**
     * @brief Set the velocity distribution object
     * 
     * @param new_udist     New udist object to be cloned
     */
    virtual void set_udist( UDistribution::Type const & new_udist ) {
        delete udist;
        udist = new_udist.clone();
    }

    /**
     * @brief Get the udist object
     * 
     * @return UDistribution::Type& 
     */
    auto & get_udist() {
        return *udist;
    } 

    __host__
    /**
     * @brief Sets the boundary condition type
     * 
     * @param new_bc 
     */
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

/*
        std::string bc_name[] = {"open", "periodic", "reflecting"};
        std::cout << "(*info*) Species " << name << " boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
*/

        // Set periodic flags on tile grids
        if ( particles ) {
            particles->periodic.x = ( bc.x.lower == species::bc::periodic );
            particles->periodic.y = ( bc.y.lower == species::bc::periodic );
        }
    }

    __host__
    /**
     * @brief Get the current boundary condition types
     * 
     * @return species::bc_type 
     */
    auto get_bc( ) { return bc; }

    __host__
    /**
     * @brief Sets moving window algorithm
     * 
     * This method can only be called before the simulation has started (iter = 0)
     * 
     * @return int  0 on success, -1 on error
     */
    virtual int set_moving_window() { 
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
    /**
     * @brief Inject particles in the simulation box
     * 
     */
    virtual void inject();

    __host__
    /**
     * @brief Inject particles in the specified range of the simulation
     * 
     * @param range     Range in which to inject particles
     */
    virtual void inject( bnd<unsigned int> range );

    __host__
    /**
     * @brief Advance particle velocities
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     */
    void push( VectorField * const E, VectorField * const B );

    __host__
    /**
     * @brief Move particles (advance positions) and deposit current
     * 
     * @param current   Electric current density
     */
    void move( VectorField * const current );

    __host__
    /**
     * @brief Move particles (advance positions) without depositing current
     * 
     */
    void move( );

    __host__
    /**
     * @brief Advance particles 1 timestep
     * 
     * @param emf       EM fields
     * @param current   Electric current density
     */
    virtual void advance( EMF const &emf, Current &current );

    __host__
    /**
     * @brief Deposit species charge
     * 
     * @param charge    Charge density grid
     */
    void deposit_charge( Field &charge ) const;

    __host__
    /**
     * @brief Returns total time centered kinetic energy
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
    /**
     * @brief Returns total number of particles moved
     * 
     * @return unsigned long long 
     */
    unsigned long long get_nmove() const {
        // Get total number of pushes from device memory
        unsigned long long h_nmove;
        devhost_memcpy( &h_nmove, d_nmove, 1 );

        return h_nmove;
    }

    __host__
    /**
     * @brief Returns the maximum number of particles per tile
     * 
     * @return auto 
     */
    auto np_max_tile() const {
        return particles -> np_max_tile();
    }

    __host__
    /**
     * @brief Save particle data to file
     * 
     * Saves positions and velocities for all particles. Positions are currently
     * normalized to cell size
     */
    void save() const;

    __host__
    /**
     * @brief Save charge density for species to file
     * 
     */
    void save_charge() const;

    __host__
    /**
     * @brief Save 1D phasespace density to file
     * 
     * @param quant     Phasespace quantity
     * @param range     Value range
     * @param size      Number of grid points
     */
    void save_phasespace ( 
        phasespace::quant quant, float2 const range, unsigned size ) const;

    __host__
    /**
     * @brief Save 2D phasespace density to file
     * 
     * @param quant0    axis 0 quantity
     * @param range0    axis 0 value range
     * @param size0     axis 0 number of points
     * @param quant1    axis 1 quantity
     * @param range1    axis 1 value range
     * @param size1     axis 1 number of points
     */
    void save_phasespace ( 
        phasespace::quant quant0, float2 const range0, unsigned size0,
        phasespace::quant quant1, float2 const range1, unsigned size1 ) const;

};


#endif