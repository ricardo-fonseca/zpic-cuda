#include <stdio.h>

#include "zpic.h"
#include "timer.cuh"
#include "emf.cuh"
#include "species.cuh"


void test_emf() {

    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    float tmax = 4.0;

    EMF emf( ntiles, nx, box, dt );

/*
    Laser laser;
    laser.start = 16;
    laser.fwhm = 4;
    laser.a0 = 1.0f;
    laser.polarization = 0.f;
    laser.omega0 = 5.0;
*/

    Gaussian laser;
    laser.start = 16;
    laser.fwhm = 4;
    laser.a0 = 1.0f;
    laser.polarization = M_PI_4;
    laser.omega0 = 5.0;

    laser.W0 = 4;
    laser.focus = 12.8;
    laser.axis = 12.8;

    Timer timer;

    timer.start();

    emf.add_laser( laser );

    emf.report( EMF::EFLD, 0 );
    emf.report( EMF::EFLD, 1 );
    emf.report( EMF::EFLD, 2 );

    emf.report( EMF::BFLD, 0 );
    emf.report( EMF::BFLD, 1 );
    emf.report( EMF::BFLD, 2 );

    while( emf.iter * emf.dt < tmax ) {
        emf.advance();
        emf.report( EMF::EFLD, 0 );
        emf.report( EMF::EFLD, 1 );
        emf.report( EMF::EFLD, 2 );

        emf.report( EMF::BFLD, 0 );
        emf.report( EMF::BFLD, 1 );
        emf.report( EMF::BFLD, 2 );

    }

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}


int main() {

    Timer timer;

    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    Current current( ntiles, nx, box, dt );

    uint2 ppc = {8,8};
    //float3 ufl = {1.0, 2.0, 3.0};
    // float3 uth = {0.1, 0.2, 0.3};
    float3 ufl = {1000., 1000., 1000.};
    float3 uth = {0};

    Species electrons( "electrons", -1, ppc, 1.0, 
        ufl, uth, box, ntiles, nx, dt );

    timer.start();

    electrons.save_particles();
    electrons.save_charge();

    float tmax = 4.0;
    while( electrons.iter * electrons.dt < tmax ) {
        printf(" i = %3d, t = %g \n", electrons.iter, electrons.iter * electrons.dt );
       
        current.zero();
       
        electrons.move_deposit( current.J );
        electrons.tile_sort();
        electrons.iter ++;

        current.advance();
    }

    current.report(0);
    current.report(1);
    current.report(2);
    electrons.save_particles();
    electrons.save_charge();

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());

    return 0;
}
