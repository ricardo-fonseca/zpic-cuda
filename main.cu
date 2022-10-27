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


    Laser::PlaneWave laser;
    laser.start = 16;
    laser.fwhm = 4;
    laser.a0 = 1.0f;
    laser.polarization = 0.f;
    laser.omega0 = 5.0;

/*
    Laser::Gaussian laser;
    laser.start = 16;
    laser.fwhm = 4;
    laser.a0 = 1.0f;
    laser.polarization = M_PI_4;
    laser.omega0 = 5.0;

    laser.W0 = 4;
    laser.focus = 12.8;
    laser.axis = 12.8;
*/

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
//    auto udist = UDistribution::Cold( make_float3(1000,1000,1000));
    auto udist = UDistribution::Cold( make_float3(+1000,+1000,+1000));

    bnd<unsigned int> range;
    range.x = { .lower = 128, .upper = 255 };
    range.y = { .lower = 128, .upper = 255 };

    // auto density = Density::Step( 1.0, 6.4 );
    // auto density =  Density::Slab( 1.0, 9.6, 16.0 );
    //auto density = Density::Sphere( 1.0, make_float2(12.8,12.8), 3.2 );
    auto density = Density::Uniform( 1.0 );

    Species electrons( "electrons", -1, ppc, density, box, ntiles, nx, dt );

    electrons.inject( range );
    electrons.set_udist( udist, 0 );
    electrons.particles->validate( "After injection");



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
    
    auto density = Density::Sphere( 1.0, make_float2(25.6,12.8), 6.4 );
    // auto density = Density::Uniform( 1.0 );

    sim.add_species( "electrons", -1.0f, ppc, density, UDistribution::None() );

    sim.species[0] -> save_charge();

    sim.set_moving_window();

    timer.start();

    int iter_max = 50;
    while( sim.get_iter() < iter_max ) {
        printf(" i = %3d, t = %g \n", sim.get_iter(), sim.get_t() );

        sim.advance();

        sim.species[0] -> save_charge();
    }

    printf(" i = %3d, t = %g (finished)\n", sim.get_iter(), sim.get_t() );

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}

void test_filter() {

    // Create simulation box
    uint2 ntiles = {32, 16};
    uint2 nx = {32,16};
    float2 box = {20.48, 25.6};
    float dt = 0.014;

    EMF emf( ntiles, nx, box, dt );

    Laser::Gaussian laser;
    laser.start = 17.0;
    laser.fwhm = 2.0;
    laser.a0 = 3.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = 20.28;
    laser.axis = 12.8;
    laser.polarization = M_PI_2;

    emf.add_laser( laser );

    Filter::Compensated filter( coord::x, 1 );
    filter.apply(*emf.E);

    emf.save( emf::e, fcomp::z );
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

    auto udist = UDistribution::Thermal( uth, ufl );
    //auto udist = UDistribution::ThermalCorr( uth, ufl, 16 );

    sim.add_species( "electrons", -1.0f, ppc, Density::Uniform(1.0f), udist );
    udist.ufl.z = -udist.ufl.z;
    sim.add_species( "positrons", +1.0f, ppc, Density::Uniform(1.0f), udist );


    sim.species[0]-> save_phasespace ( 
        phasespace::ux, make_float2( -1, 1 ), 128,
        phasespace::uz, make_float2( -1, 1 ), 128
        );

    sim.species[1]-> save_phasespace ( 
        phasespace::ux, make_float2( -1, 1 ), 128,
        phasespace::uz, make_float2( -1, 1 ), 128
        );


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
    
    sim.energy_info();

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
    uint2 ntiles = {32, 16};
    uint2 nx = {32,16};
    float2 box = {20.48, 25.6};

    float dt = 0.014;

    Simulation sim( ntiles, nx, box, dt );

    // Add particles species
    uint2 ppc  = {4,4};

    sim.add_species( "electrons", -1.0f, ppc, Density::Step( coord::x, 1.0, 20.48 ), UDistribution::None() );

    Laser::Gaussian laser;
    laser.start = 17.0;
    laser.fwhm = 2.0;
    laser.a0 = 3.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = 20.28;
    laser.axis = 12.8;
    laser.polarization = M_PI_2;

    sim.emf -> add_laser( laser );
    // sim.current -> set_filter( Filter::Compensated() );

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

    sim.species[0]-> save_phasespace ( 
        phasespace::x,  make_float2( 0., 20.48 ), 1024,
        phasespace::ux, make_float2( -2., 2 ), 512
        );
    sim.species[0]->save();

    printf("Elapsed time was: %.3f s\n", timer.elapsed( timer::s ));
}


void diag_mushroom( Simulation & sim ) {

    sim.species[0] -> save_charge();
    sim.species[1] -> save_charge();
    sim.species[2] -> save_charge();
    sim.species[3] -> save_charge();

    sim.current -> save( fcomp::x );
    sim.current -> save( fcomp::y );
    sim.current -> save( fcomp::z );

    sim.emf -> save( emf::e, fcomp::x );
    sim.emf -> save( emf::e, fcomp::y );
    sim.emf -> save( emf::e, fcomp::z );

    sim.emf -> save( emf::b, fcomp::x );
    sim.emf -> save( emf::b, fcomp::y );
    sim.emf -> save( emf::b, fcomp::z );

}

void test_mushroom() {

    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};
    float2 box = {25.6, 25.6};

    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    // Add particles species
    uint2 ppc   = {8,8};
    float3 ufl  = {0., 0., 0.2};
    float3 uth_e = {0.001, 0.001, 0.001};
    float3 uth_i = {0.0001, 0.0001, 0.0001};

    auto udist_e = UDistribution::ThermalCorr( uth_e, ufl, 16 );
    auto udist_i = UDistribution::ThermalCorr( uth_i, ufl, 16 );

    sim.add_species( "electrons-up", -1.0f, ppc, Density::Slab(coord::y,1.0,0,box.y/2), udist_e );
    sim.add_species( "ions-up",    +100.0f, ppc, Density::Slab(coord::y,1.0,0,box.y/2), udist_i );

    udist_e.ufl.z = - udist_e.ufl.z;
    udist_i.ufl.z = - udist_i.ufl.z;

    sim.add_species( "electrons-down", -1.0f, ppc, Density::Slab(coord::y,1.0,box.y/2,box.y), udist_e );
    sim.add_species( "ions-down",    +100.0f, ppc, Density::Slab(coord::y,1.0,box.y/2,box.y), udist_i );

    // Run simulation
    int const nmax = 2000 ;

    printf("Running Mushroom test up to n = %d...\n", nmax );

    Timer timer;

    timer.start();

    diag_mushroom( sim );

    while( sim.get_iter() < nmax ) {
        sim.advance();
        if ( sim.get_iter() % 100 == 0 ) diag_mushroom( sim );
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());


    printf("Elapsed time was: %.3f s\n", timer.elapsed( timer::s ));
}


void diag_kh( Simulation & sim ) {

    sim.species[0] -> save_charge();
    sim.species[1] -> save_charge();
    sim.species[2] -> save_charge();
    sim.species[3] -> save_charge();

    sim.current -> save( fcomp::x );
    sim.current -> save( fcomp::y );
    sim.current -> save( fcomp::z );

    sim.emf -> save( emf::e, fcomp::x );
    sim.emf -> save( emf::e, fcomp::y );
    sim.emf -> save( emf::e, fcomp::z );

    sim.emf -> save( emf::b, fcomp::x );
    sim.emf -> save( emf::b, fcomp::y );
    sim.emf -> save( emf::b, fcomp::z );

}

void test_kh() {

    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};
    float2 box = {25.6, 25.6};

    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    // Add particles species
    uint2 ppc   = {8,8};
    float3 ufl  = {0.2, 0., 0.};
    float3 uth_e = {0.001, 0.001, 0.001};
    float3 uth_i = {0.0001, 0.0001, 0.0001};

    auto udist_e = UDistribution::ThermalCorr( uth_e, ufl, 16 );
    auto udist_i = UDistribution::ThermalCorr( uth_i, ufl, 16 );

    sim.add_species( "electrons-r", -1.0f, ppc, Density::Slab(coord::y,1.0,0,box.y/2), udist_e );
    sim.add_species( "ions-r",    +100.0f, ppc, Density::Slab(coord::y,1.0,0,box.y/2), udist_i );

    udist_e.ufl.x = - udist_e.ufl.x;
    udist_i.ufl.x = - udist_i.ufl.x;

    sim.add_species( "electrons-l", -1.0f, ppc, Density::Slab(coord::y,1.0,box.y/2,box.y), udist_e );
    sim.add_species( "ions-l",    +100.0f, ppc, Density::Slab(coord::y,1.0,box.y/2,box.y), udist_i );

    sim.current->set_filter( Filter::Binomial ( coord::x, 2 ) );

    // Run simulation
    float const tmax = 100 ;

    printf("Running Kelvin-Helmholtz test up to t = %g...\n", tmax );

    Timer timer;

    timer.start();

    diag_kh( sim );

    while( sim.get_t() < tmax ) {
        sim.advance();
        if ( sim.get_iter() % 100 == 0 ) diag_kh( sim );
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());


    printf("Elapsed time was: %.3f s\n", timer.elapsed( timer::s ));
}

void test_bcemf() {

    // Create simulation box
    uint2 ntiles = {32, 16};
    uint2 nx = {32,16};
    float2 box = {20.48, 25.6};

    float dt = 0.014;

    Simulation sim( ntiles, nx, box, dt );

    // emf::bc_type emf_bc(emf::bc::pmc);
    emf::bc_type emf_bc(emf::bc::pec);

    sim.emf->set_bc(emf_bc); 


    Laser::Gaussian laser;
    laser.start = 17.0;
    laser.fwhm = 2.0;
    laser.a0 = 3.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = 20.28;
    laser.axis = 12.8;

    laser.cos_pol = 1;
    laser.sin_pol = 0;
    
    sim.emf -> add_laser( laser );

    sim.emf -> save( emf::e, fcomp::x );
    sim.emf -> save( emf::e, fcomp::y );
    sim.emf -> save( emf::e, fcomp::z );
    sim.emf -> save( emf::b, fcomp::x );
    sim.emf -> save( emf::b, fcomp::y );
    sim.emf -> save( emf::b, fcomp::z );


    // Run simulation
    float const tmax = 42 ;


    Timer timer;

    timer.start();

    while( sim.get_t() < tmax ) {
        sim.advance();
        if ( sim.get_iter() % 100 == 0 ) {
            sim.emf -> save( emf::e, fcomp::x );
            sim.emf -> save( emf::e, fcomp::y );
            sim.emf -> save( emf::e, fcomp::z );
            sim.emf -> save( emf::b, fcomp::x );
            sim.emf -> save( emf::b, fcomp::y );
            sim.emf -> save( emf::b, fcomp::z );
        };
    }

    timer.stop();


    printf("Simulation complete at t = %g\n", sim.get_t());


    printf("Elapsed time was: %.3f s\n", timer.elapsed( timer::s ));

}

void test_bcspec() {
    
    
    uint2 ntiles = {16, 16};
    uint2 nx = {16,16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );


    sim.add_species( "electrons", -1.0f, 
        make_uint2( 8, 8 ), // ppc
        Density::Sphere( 1.0, make_float2(12.8,12.8), 6.4 ),
        UDistribution::Cold( make_float3( -1.0e8, 0, 0 ) )
    );

    auto bc = sim.species[0] -> get_bc();
    bc.x = { 
        .lower = species::bc::reflecting,
        .upper = species::bc::reflecting
    };
    sim.species[0] -> set_bc( bc );

    sim.species[0] -> save_charge();
    sim.current -> save( fcomp::x );
    sim.emf -> save( emf::e, fcomp::x );

    float const tmax = 51.2 ;

    printf("Running Species boundary condition test up to t = %g...\n", tmax );

    Timer timer;
    timer.start();

    while( sim.get_t() < tmax ) {

        sim.advance();
        if ( sim.get_iter() % 50 == 0 ) {
            sim.species[0] -> save_charge();
            sim.current -> save( fcomp::x );
            sim.emf -> save( emf::e, fcomp::x );
        }
    }

    printf("Simulation complete at t = %g\n", sim.get_t());

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}

int main() {

    // test_emf();

    // test_sort_deposit();

    // test_move_window();

    // test_filter();

    // test_weibel();

    // test_lwfa();

    // test_mushroom();

    // test_kh();

    // test_bcemf();

    test_bcspec();

    return 0;
}