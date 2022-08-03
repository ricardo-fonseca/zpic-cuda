#ifndef __LASER__
#define __LASER__

#include "tile_vfld.cuh"

class Laser {
    public:

    float start;    // Front edge of the laser pulse, in simulation units
    float fwhm;     // FWHM of the laser pulse duration, in simulation units
    float rise, flat, fall; // Rise, flat and fall time of the laser pulse, in simulation units 

    float a0;       // Normalized peak vector potential of the pulse
    float omega0;   // Laser frequency, normalized to the plasma frequency

    float polarization;

    float cos_pol;
    float sin_pol;

    __host__ Laser();

    __host__ virtual int validate();
    __host__ virtual int launch( VFLD& E, VFLD& B, float2 box );


    private :
    
    __device__ float lon_env( const float z );

};

class Gaussian : public Laser {

    public:

    float W0;
    float focus;
    float axis;

    __host__ int validate();
    __host__ int launch( VFLD& E, VFLD& B, float2 box );
};


#endif