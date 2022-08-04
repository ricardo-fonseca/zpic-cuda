#include <stdio.h>

#include "zpic.h"
#include "timer.cuh"
#include "emf.cuh"
#include "particles.cuh"


void test_emf() {

    int2 gnx = {256, 256};
    int2 tnx = {16,16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    float tmax = 4.0;

    EMF emf( gnx, tnx, box, dt );

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

    while( emf.d_iter * emf.dt < tmax ) {
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

    int2 gnx = {256, 256};
    int2 tnx = {16,16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    float tmax = 4.0;

    int2 ppc = {8,8};
    float3 ufl = {1.0, 2.0, 3.0};
    float3 uth = {0.1, 0.2, 0.3};

    Species electrons( "electrons", -1, ppc, 1.0, 
        ufl, uth, box, gnx, tnx, dt );

    timer.start();

    electrons.save_particles();
    electrons.save_charge();

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());

    return 0;
}
