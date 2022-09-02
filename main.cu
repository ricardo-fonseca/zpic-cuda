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

    emf.set_moving_window();

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

    int iter = 0;

    while( iter * dt < tmax ) {
        emf.advance();
        emf.save( emf::e, fcomp::x );
        emf.save( emf::e, fcomp::y );
        emf.save( emf::e, fcomp::z );

        emf.save( emf::b, fcomp::x );
        emf.save( emf::b, fcomp::y );
        emf.save( emf::b, fcomp::z );

        iter++;
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

    float3 uth = { 0 };
//    float3 uth = {0.1, 0.2, 0.3 };

//    float3 ufl = {0.,    0.,    0.};
//   float3 ufl = {   0.,    0., 1000.};
//    float3 ufl = {1000.,    0.,  1000.};
//    float3 ufl = {   0., 1000.,    1000.};
    float3 ufl = {1000., 1000., 1000.};


    bnd<unsigned int> range;
    range.x = { .lower = 128, .upper = 255 };
    range.y = { .lower = 128, .upper = 255 };

    // auto density = Density::Step( 1.0, 6.4 );
    // auto density =  Density::Slab( 1.0, 9.6, 16.0 );
    auto density = Density::Sphere( 1.0, make_float2(12.8,12.8), 3.2 );

    Species electrons( "electrons", -1, ppc, density, box, ntiles, nx, dt );

    electrons.inject( range );
    electrons.particles->validate( "After injection");
    electrons.set_u( uth, ufl );

    electrons.save_charge();
    current.save( fcomp::x );
    current.save( fcomp::y );
    current.save( fcomp::z );

    timer.start();

    int iter = 0;
    int iter_max = 10;
    while( iter < iter_max ) {
        printf(" i = %3d, t = %g \n", iter, iter * dt );

        current.zero();

        electrons.move( current.J );
        electrons.particles->tile_sort();

        electrons.particles->validate("after tile sort");
        electrons.iter++;

        current.advance();

        electrons.save_charge();
        current.save( fcomp::x );
        current.save( fcomp::y );
        current.save( fcomp::z );

        iter++;
    }

    printf(" i = %3d, t = %g (finished)\n", iter, iter * dt );

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}


void test_move_window() {
    
    std::cout << "Running move window test...\n";
    
    Timer timer;

    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    // Add particles species
    uint2 ppc  = {8,8};
    float3 ufl = {0., 0., 0.};
    float3 uth = {0., 0., 0.};
    
    auto density = Density::Sphere( 1.0, make_float2(25.6,12.8), 6.4 );
    // auto density = Density::Uniform( 1.0 );

    sim.add_species( "electrons", -1.0f, ppc, density, uth, ufl );

    sim.get_species(0) -> save_charge();

    sim.set_moving_window();

    timer.start();

    int iter_max = 50;
    while( sim.get_iter() < iter_max ) {
        printf(" i = %3d, t = %g \n", sim.get_iter(), sim.get_t() );

        sim.advance();

        sim.get_species(0) -> save_charge();
    }

    printf(" i = %3d, t = %g (finished)\n", sim.get_iter(), sim.get_t() );

    timer.stop();

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

    sim.add_species( "electrons", -1.0f, ppc, Density::Uniform(1.0f), uth, ufl );
    ufl.z = -ufl.z;
    sim.add_species( "positrons", +1.0f, ppc, Density::Uniform(1.0f), uth, ufl );

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

void test_lwfa() {

    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {64,16};
    float2 box = {20.48, 25.6};

    float dt = 0.014;

    Simulation sim( ntiles, nx, box, dt );

    // Add particles species
    uint2 ppc  = {8,8};
    float3 ufl = {0., 0., 0.};
    float3 uth = {0., 0., 0.};

    sim.add_species( "electrons", -1.0f, ppc, Density::Step(1.0f,20.48), uth, ufl );

    Gaussian laser;
    laser.start = 17.0;
    laser.fwhm = 2.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = 20.28;
    laser.axis = 12.8;
    laser.polarization = M_PI_2;

    sim.add_laser( laser );

    sim.set_moving_window();

    // Run simulation
    float const tmax = 22.;

    printf("Running LWFA test up to t = %g...\n", tmax );

    Timer timer;

    timer.start();

    while( sim.get_t() < tmax ) {

        sim.advance();
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());
    
    sim.emf -> save( emf::e, fcomp::x );
    sim.emf -> save( emf::e, fcomp::y );
    sim.emf -> save( emf::e, fcomp::z );

    sim.get_species(0)-> save_phasespace ( 
        phasespace::x,  make_float2( 0., 20.48 ), 1024,
        phasespace::ux, make_float2( -2., 2 ), 512
        );
    sim.get_species(0)->save();

    printf("Elapsed time was: %.3f s\n", timer.elapsed( timer::s ));
}

int main() {

    // test_emf();

    // test_sort_deposit();

    // test_move_window();

    // test_weibel();

    test_lwfa();

    return 0;
}