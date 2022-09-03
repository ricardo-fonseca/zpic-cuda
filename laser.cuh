#ifndef __LASER__
#define __LASER__

#include "vector_field.cuh"

namespace Laser {

class Pulse {
    public:

    float start;    // Front edge of the laser pulse, in simulation units
    float fwhm;     // FWHM of the laser pulse duration, in simulation units
    float rise, flat, fall; // Rise, flat and fall time of the laser pulse, in simulation units 

    float a0;       // Normalized peak vector potential of the pulse
    float omega0;   // Laser frequency, normalized to the plasma frequency

    float polarization;

    float cos_pol;
    float sin_pol;

    __host__ Pulse() : start(0), fwhm(0), rise(0), flat(0), fall(0),
        a0(0), omega0(0),
        polarization(0), cos_pol(0), sin_pol(0) {};

    __host__ virtual int validate();
    __host__ virtual int launch( VectorField& E, VectorField& B, float2 box ) = 0;

};

class PlaneWave : public Pulse {

    public:
    __host__ PlaneWave() : Pulse() {};

    __host__ int validate() { return Pulse::validate(); };
    __host__ int launch( VectorField& E, VectorField& B, float2 box );

};

class Gaussian : public Pulse {

    public:

    float W0;
    float focus;
    float axis;

    __host__ Gaussian() : Pulse(), W0(0), focus(0), axis(0) {};

    __host__ int validate();
    __host__ int launch( VectorField& E, VectorField& B, float2 box );
};

}

#endif