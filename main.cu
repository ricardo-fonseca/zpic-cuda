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
    
    std::cout << "Running sort/deposit test...\n";
    
    Timer timer;

    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    Current current( ntiles, nx, box, dt );

    uint2 ppc = {8,8};
    float3 uth = {0.1, 0.2, 0.3 };

    float3 ufl = {0.,    0.,    0.};
//    float3 ufl = {1000.,    0.,    0.};
//    float3 ufl = {   0., 1000.,    0.};
//   float3 ufl = {   0.,    0., 1000.};
//    float3 ufl = {1000., 1000., 1000.};

    Species electrons( "electrons", -1, ppc, 1.0, box, ntiles, nx, dt );


    // electrons.inject_particles( density::uniform() );
    electrons.inject_particles( density::sphere( make_float2(12.8,12.8), 3.2) );
    // electrons.inject_particles( density::slab( 9.6,16.0 ) );
    electrons.set_u( uth, ufl );

    current.save( fcomp:: x );
    current.save( fcomp:: y );
    current.save( fcomp:: z );

    electrons.save();
    electrons.save_charge();

    timer.start();

    int iter = 0;
    int iter_max = 10;
    while( iter < iter_max ) {
        printf(" i = %3d, t = %g \n", iter, iter * dt );

        current.zero();

        electrons.move( current.J );
        electrons.particles->tile_sort();
        electrons.iter++;

        current.advance();

        electrons.save_charge();

        electrons.save_phasespace( phasespace::x, make_float2( 0., 25.6), 200 ); 
        // electrons.save_phasespace( phasespace::y, make_float2( 0., 25.6), 200 ); 

        // electrons.save_phasespace( phasespace::ux, make_float2( -1., 1.), 128, 
        //                           phasespace::uy, make_float2( -1., 1.), 128 );

        electrons.save_phasespace( phasespace::x, make_float2( -1., 26.6), 150, 
                                   phasespace::y, make_float2( -1., 26.6), 150 );

        // electrons.save_phasespace( phasespace::x, make_float2( -1., 26.6), 128, 
        //                            phasespace::ux, make_float2( -1., 2.), 128 );

        // electrons.save_phasespace( phasespace::y, make_float2( -1., 26.6), 128, 
        //                           phasespace::ux, make_float2( -1., 2.), 128 );

        // current.save( fcomp:: x );
        // current.save( fcomp:: y );
        // current.save( fcomp:: z );

        iter++;
    }

    printf(" i = %3d, t = %g (finished)\n", iter, iter * dt );

    timer.stop();

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
    float const imax = 500;

    printf("Running Weibel test up to n = %g...\n", imax );

    Timer timer;

    timer.start();

    while( sim.get_iter() < imax ) {

        sim.advance();
    }

    timer.stop();

    printf("Simulation complete at i = %d\n", sim.get_iter());
    
    sim.current -> save( fcomp::x );
    sim.current -> save( fcomp::y );
    sim.current -> save( fcomp::z );

    sim.emf -> save( emf::e, fcomp::x );
    sim.emf -> save( emf::e, fcomp::y );
    sim.emf -> save( emf::e, fcomp::z );

    sim.emf -> save( emf::b, fcomp::x );
    sim.emf -> save( emf::b, fcomp::y );
    sim.emf -> save( emf::b, fcomp::z );

    printf("Elapsed time was: %.3f s\n", timer.elapsed( timer::s ));
}


int main() {

    // test_emf();

    test_sort_deposit();

    // test_weibel();

    return 0;
}