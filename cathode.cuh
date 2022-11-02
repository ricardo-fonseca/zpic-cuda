

#ifndef __CATHODE__
#define __CATHODE__

#include "species.cuh"

class Cathode : public Species {

private:

    /// @brief velocity (beta) of cathode flow
    float vel;

    /// @brief generalized velocity of cathode flow
    float ufl;

    float *h_inj_pos;
    float *d_inj_pos;

    /// @brief wall to inject particles from
    edge::pos wall;

public:

    /// @brief Time to start cathode injection
    float start;

    /// @brief Time to end cathode injection
    float end;

    /// @brief temperature of injected particles
    float3 uth;

    Cathode( std::string const name, float ufl, edge::pos wall, 
        float const m_q, uint2 const ppc, float n0, 
        float2 const box, uint2 const ntiles, uint2 const nx,
        const float dt );
    
    ~Cathode();

    void advance( EMF const &emf, Current &current ) override;

    int set_moving_window() override {
        std::cerr << "(*error*) Cathodes cannot be used with moving windows, aborting...\n";
        exit(1);
    }

    virtual void inject() override;

    virtual void inject( bnd<unsigned int> range )  override;


};

#endif