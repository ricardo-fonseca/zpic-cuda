

#ifndef __CATHODE__
#define __CATHODE__

#include "species.cuh"

class Cathode : public Species {

private:

    /// @brief velocity (beta) of cathode flow
    float vel;

    /// @brief generalized velocity of cathode flow
    float ufl;

    /// @brief Position of injection particles (device)
    float *d_inj_pos;

public:

    /// @brief wall to inject particles from
    edge::pos wall;

    /// @brief Time to start cathode injection
    float start;

    /// @brief Time to end cathode injection
    float end;

    /// @brief temperature of injected particles
    float3 uth;

    /// @brief Reference density for cathode
    float n0;

    Cathode( std::string const name, float const m_q, uint2 const ppc, float ufl );

    void initialize( float2 const box, uint2 const ntiles, uint2 const nx,
        float const dt, int const id_ ) override;
    
    ~Cathode();

    void advance( EMF const &emf, Current &current ) override;

    int set_moving_window() override {
        std::cerr << "(*error*) Cathodes cannot be used with moving windows, aborting...\n";
        exit(1);
    }

    void set_udist( UDistribution::Type const & new_udist ) override {
        std::cerr << "(*error*) Cathodes do not support the set_udist() method, use the uth parameter instead\n";
        exit(1);
    }

    void set_density( Density::Profile const & new_density ) override {
        std::cerr << "(*error*) Cathodes do not support the set_density() method, use the n0 parameter instead\n";
        exit(1);
    }

    virtual void inject() override;

    virtual void inject( bnd<unsigned int> range )  override;


};

#endif