#include <stdio.h>

#include "zpic.h"
#include "timer.cuh"
#include "emf.cuh"
#include "species.cuh"
#include "current.cuh"

#include "simulation.cuh"

/**
 * @brief Tests EM solver and laser injection
 * 
 * Injects a laser pulse and propagates for a given time.
 * 
 */
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

    emf.save( emf::e, fcomp::x );
    emf.save( emf::e, fcomp::y );
    emf.save( emf::e, fcomp::z );

    emf.save( emf::b, fcomp::x );
    emf.save( emf::b, fcomp::y );
    emf.save( emf::b, fcomp::z );

    while( emf.iter * emf.dt < tmax ) {
        emf.advance();
        emf.save( emf::e, fcomp::x );
        emf.save( emf::e, fcomp::y );
        emf.save( emf::e, fcomp::z );

        emf.save( emf::b, fcomp::x );
        emf.save( emf::b, fcomp::y );
        emf.save( emf::b, fcomp::z );
    }

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}

/**
 * @brief Tests the tile sort and current deposit
 * 
 * Creates a sphere of particles and free streams it to check if the tile sort
 * is operating correctly
 * 
 */
void test_sort_deposit() {
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

    Species electrons( "electrons", -1, ppc, 1.0, box, ntiles, nx, dt );


    electrons.inject_particles( density::sphere( make_float2(9.6,16.0), 3.2) );
    electrons.set_u( uth, ufl );

    current.save( fcomp:: x );
    current.save( fcomp:: y );
    current.save( fcomp:: z );

    electrons.save();
    electrons.save_charge();

    timer.start();

    int iter = 0;
    int iter_max = 100;
    while( iter < iter_max ) {
        printf(" i = %3d, t = %g \n", iter, iter * dt );

        current.zero();

        electrons.move( current.J );

        current.advance();

        iter++;
    }

    printf(" i = %3d, t = %g (finished)\n", iter, iter * dt );

    timer.stop();

    current.save( fcomp:: x );
    current.save( fcomp:: y );
    current.save( fcomp:: z );

    electrons.save();
    electrons.save_charge();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}


void test_weibel() {

    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};
    float2 box = {25.6, 25.6};

    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    // Add particles species
    uint2 ppc  = {8,8};
    float3 ufl = {0., 0., 0.6};
    float3 uth = {0.1, 0.1, 0.1};

    sim.add_species( "electrons", -1.0f, ppc, 1.0f, density::uniform(), uth, ufl );
    ufl.z = -ufl.z;
    sim.add_species( "positrons", +1.0f, ppc, 1.0f, density::uniform(), uth, ufl );

    // Run simulation
    int const imax = 500;

    printf("Running simulation up to i = %d...\n", imax );

    Timer timer;

    timer.start();

    while( sim.get_iter() < imax ) {

        sim.advance();
    }

    timer.stop();

    printf("Simulation complete at i = %d\n", sim.get_iter());
    

    sim.emf -> save( emf::e, fcomp::x );
    sim.emf -> save( emf::e, fcomp::y );
    sim.emf -> save( emf::e, fcomp::z );

    sim.emf -> save( emf::b, fcomp::x );
    sim.emf -> save( emf::b, fcomp::y );
    sim.emf -> save( emf::b, fcomp::z );

    printf("Elapsed time was: %.3f ms\n", timer.elapsed());
}


int main() {

    // test_emf();

    // test_sort_deposit();

    test_weibel();

    return 0;
}