#include <stdio.h>
#include <list>

#include "zpic.h"

#include <vector>


/**
 * @brief Tests absorbing boundary conditions for particles
 * 
 */
void test_absorb() {
    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {16, 16};

    float2 box = {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};

    float dt = 0.07;

    Simulation sim(ntiles, nx, box, dt);

    // Add particles species
    uint2 ppc = {8, 8};

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_density(Density::Step(coord::x, 1.0, 5.0));

    sim.add_species(electrons);
    sim.set_moving_window();
    
    electrons.save_charge();

    // Run simulation
    int const imax = 256;

    printf("Running mov.window absorption test up to n = %d...\n", imax);

    Timer timer;

    timer.start();

    while (sim.get_iter() < imax)
    {
        sim.advance();

        if ( sim.get_iter() % 16 == 0 ) electrons.save_charge();
    }

    timer.stop();

    printf("Simulation complete at i = %d\n", sim.get_iter());

    printf("Elapsed time was: %.3f s\n", timer.elapsed(timer::s));
}

/**
 * @brief Tests EM solver and laser injection
 *
 * Injects a laser pulse and propagates for a given time.
 *
 */
void test_emf()
{
    uint2 ntiles = {16, 16};
    uint2 nx = {16, 16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    float tmax = 4.0;

    EMF emf(ntiles, nx, box, dt);

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

    emf.add_laser(laser);

    emf.save(emf::e, fcomp::x);
    emf.save(emf::e, fcomp::y);
    emf.save(emf::e, fcomp::z);

    emf.save(emf::b, fcomp::x);
    emf.save(emf::b, fcomp::y);
    emf.save(emf::b, fcomp::z);

    int iter = 0;

    while (iter * dt < tmax)
    {
        emf.advance();
        emf.save(emf::e, fcomp::x);
        emf.save(emf::e, fcomp::y);
        emf.save(emf::e, fcomp::z);

        emf.save(emf::b, fcomp::x);
        emf.save(emf::b, fcomp::y);
        emf.save(emf::b, fcomp::z);

        iter++;
    }

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}

void test_move_window()
{

    std::cout << "Running move window test...\n";

    Timer timer;

    uint2 ntiles = {16, 16};
    uint2 nx = {16, 16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    Simulation sim(ntiles, nx, box, dt);

    // Add particles species
    uint2 ppc = {8, 8};

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_density(Density::Sphere(1.0, make_float2(25.6, 12.8), 6.4));
    sim.add_species(electrons);

    electrons.save_charge();

    sim.set_moving_window();

    timer.start();

    int iter_max = 50;
    while (sim.get_iter() < iter_max)
    {
        printf(" i = %3d, t = %g \n", sim.get_iter(), sim.get_t());

        sim.advance();
        electrons.save_charge();
    }

    printf(" i = %3d, t = %g (finished)\n", sim.get_iter(), sim.get_t());

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}

void test_filter()
{

    // Create simulation box
    uint2 ntiles = {32, 16};
    uint2 nx = {32, 16};
    float2 box = {20.48, 25.6};
    float dt = 0.014;

    EMF emf(ntiles, nx, box, dt);

    Laser::Gaussian laser;
    laser.start = 17.0;
    laser.fwhm = 2.0;
    laser.a0 = 3.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = 20.28;
    laser.axis = 12.8;
    laser.polarization = M_PI_2;

    emf.add_laser(laser);

    Filter::Compensated filter(coord::x, 1);
    filter.apply(*emf.E);

    emf.save(emf::e, fcomp::z);
}

void test_weibel()
{

    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {16, 16};

    float2 box = {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};

    float dt = 0.07;

    Simulation sim(ntiles, nx, box, dt);

    // Add particles species
    uint2 ppc = {8, 8};
    float3 ufl = {0., 0., 0.6};
    float3 uth = {0.1, 0.1, 0.1};

    UDistribution::Thermal udist(uth, ufl);

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(udist);

    Species positrons("positrons", +1.0f, ppc);
    udist.ufl.z = -udist.ufl.z;
    positrons.set_udist(udist);

    sim.add_species(electrons);
    sim.add_species(positrons);

    electrons.save_phasespace(
        phasespace::ux, make_float2(-1, 1), 128,
        phasespace::uz, make_float2(-1, 1), 128);

    positrons.save_phasespace(
        phasespace::ux, make_float2(-1, 1), 128,
        phasespace::uz, make_float2(-1, 1), 128);

    // Run simulation
    float const imax = 500;

    printf("Running Weibel test up to n = %g...\n", imax);

    Timer timer;

    timer.start();

    while (sim.get_iter() < imax)
    {
        sim.advance();
    }

    timer.stop();

    printf("Simulation complete at i = %d\n", sim.get_iter());

    sim.energy_info();

    sim.current->save(fcomp::x);
    sim.current->save(fcomp::y);
    sim.current->save(fcomp::z);

    sim.emf->save(emf::e, fcomp::x);
    sim.emf->save(emf::e, fcomp::y);
    sim.emf->save(emf::e, fcomp::z);

    sim.emf->save(emf::b, fcomp::x);
    sim.emf->save(emf::b, fcomp::y);
    sim.emf->save(emf::b, fcomp::z);

    printf("Elapsed time was: %.3f s\n", timer.elapsed(timer::s));
    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void test_weibel_large()
{

    // Create simulation box
    uint2 ntiles = {32, 32};
    uint2 nx = {40, 32};

    float2 box = {nx.x * ntiles.x * 0.1f, nx.y * ntiles.y * 0.1f};

    float dt = 0.07;

    Simulation sim(ntiles, nx, box, dt);

    // Add particles species
    uint2 ppc = {8, 8};
    float3 ufl = {0., 0., 0.6};
    float3 uth = {0.1, 0.1, 0.1};

    UDistribution::Thermal udist(uth, ufl);

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(udist);

    Species positrons("positrons", +1.0f, ppc);
    udist.ufl.z = -udist.ufl.z;
    positrons.set_udist(udist);

    sim.add_species(electrons);
    sim.add_species(positrons);

    // Run simulation
    float const imax = 500;

    printf("Running Weibel test up to n = %g...\n", imax);

    Timer timer;

    timer.start();

    while (sim.get_iter() < imax)
    {
        sim.advance();
    }

    timer.stop();

    printf("Simulation complete at i = %d\n", sim.get_iter());

    sim.energy_info();

    sim.emf->save(emf::b, fcomp::x);
    sim.emf->save(emf::b, fcomp::y);
    sim.emf->save(emf::b, fcomp::z);

    printf("Elapsed time was: %.3f s\n", timer.elapsed(timer::s));
    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void test_lwfa()
{

    // Create simulation box
    uint2 ntiles = {32, 16};
    uint2 nx = {64, 16};
    float2 box = {40.96, 25.6};

    float dt = 0.014;

    Simulation sim(ntiles, nx, box, dt);

    // Add particles species
    uint2 ppc = {4, 4};

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_density(Density::Step(coord::x, 1.0, 40.96));

    sim.add_species(electrons);

    Laser::Gaussian laser;
    laser.start = 40.8;
    laser.fwhm = 2.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.W0 = 4.0;
    laser.focus = 40.96;
    laser.axis = 12.8;
    laser.polarization = M_PI_2;

    sim.emf->add_laser(laser);
    sim.current -> set_filter( Filter::Compensated( coord::x ) );

    sim.set_moving_window();

    // Run simulation
    float const tmax = 44.;

    printf("Running LWFA test up to t = %g...\n", tmax);

    Timer timer;

    timer.start();

    while (sim.get_t() < tmax)
    {
        sim.advance_mov_window();
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());

    sim.emf->save(emf::e, fcomp::x);
    sim.emf->save(emf::e, fcomp::y);
    sim.emf->save(emf::e, fcomp::z);

    electrons.save_phasespace(
        phasespace::x, make_float2(0., 20.48), 1024,
        phasespace::ux, make_float2(-4., 8), 1024);
    electrons.save();

    printf("Elapsed time was: %.3f s\n", timer.elapsed(timer::s));
    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void diag_mushroom(Simulation &sim)
{

    sim.species[0]->save_charge();
    sim.species[1]->save_charge();
    sim.species[2]->save_charge();
    sim.species[3]->save_charge();

    sim.current->save(fcomp::x);
    sim.current->save(fcomp::y);
    sim.current->save(fcomp::z);

    sim.emf->save(emf::e, fcomp::x);
    sim.emf->save(emf::e, fcomp::y);
    sim.emf->save(emf::e, fcomp::z);

    sim.emf->save(emf::b, fcomp::x);
    sim.emf->save(emf::b, fcomp::y);
    sim.emf->save(emf::b, fcomp::z);
}

void test_mushroom()
{

    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {16, 16};
    float2 box = {25.6, 25.6};

    float dt = 0.07;

    Simulation sim(ntiles, nx, box, dt);

    // Add particles species
    uint2 ppc = {8, 8};
    float3 ufl = {0., 0., 0.2};
    float3 uth_e = {0.001, 0.001, 0.001};
    float3 uth_i = {0.0001, 0.0001, 0.0001};

    UDistribution::ThermalCorr udist_e(uth_e, ufl, 16);
    UDistribution::ThermalCorr udist_i(uth_i, ufl, 16);

    Species e_up("electrons-up", -1.0f, ppc);
    e_up.set_density(Density::Slab(coord::y, 1.0, 0, box.y / 2));
    e_up.set_udist(udist_e);
    sim.add_species(e_up);

    Species i_up("ions-up", +100.0f, ppc);
    i_up.set_density(Density::Slab(coord::y, 1.0, 0, box.y / 2));
    i_up.set_udist(udist_i);
    sim.add_species(i_up);

    udist_e.ufl.z = -udist_e.ufl.z;
    udist_i.ufl.z = -udist_i.ufl.z;

    Species e_down("electrons-down", -1.0f, ppc);
    e_down.set_density(Density::Slab(coord::y, 1.0, box.y / 2, box.y));
    e_down.set_udist(udist_e);
    sim.add_species(e_down);

    Species i_down("ions-down", +100.0f, ppc);
    i_down.set_density(Density::Slab(coord::y, 1.0, box.y / 2, box.y));
    i_down.set_udist(udist_i);
    sim.add_species(i_down);

    // Run simulation
    int const nmax = 2000;

    printf("Running Mushroom test up to n = %d...\n", nmax);

    Timer timer;

    timer.start();

    diag_mushroom(sim);

    while (sim.get_iter() < nmax)
    {
        sim.advance();
        if (sim.get_iter() % 100 == 0)
            diag_mushroom(sim);
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());

    sim.energy_info();

    printf("Elapsed time was: %.3f s\n", timer.elapsed(timer::s));
    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void diag_kh(Simulation &sim)
{

    sim.species[0]->save_charge();
    sim.species[1]->save_charge();
    sim.species[2]->save_charge();
    sim.species[3]->save_charge();

    sim.current->save(fcomp::x);
    sim.current->save(fcomp::y);
    sim.current->save(fcomp::z);

    sim.emf->save(emf::e, fcomp::x);
    sim.emf->save(emf::e, fcomp::y);
    sim.emf->save(emf::e, fcomp::z);

    sim.emf->save(emf::b, fcomp::x);
    sim.emf->save(emf::b, fcomp::y);
    sim.emf->save(emf::b, fcomp::z);
}

void test_kh()
{

    // Create simulation box
    uint2 ntiles = {16, 16};
    uint2 nx = {16, 16};
    float2 box = {25.6, 25.6};

    float dt = 0.07;

    Simulation sim(ntiles, nx, box, dt);

    // Add particles species
    uint2 ppc = {8, 8};
    float3 ufl = {0.2, 0., 0.};
    float3 uth_e = {0.001, 0.001, 0.001};
    float3 uth_i = {0.0001, 0.0001, 0.0001};

    UDistribution::ThermalCorr udist_e(uth_e, ufl, 16);
    UDistribution::ThermalCorr udist_i(uth_i, ufl, 16);

//    UDistribution::Thermal udist_e( uth_e, ufl );
//    UDistribution::Thermal udist_i( uth_i, ufl );


    Species e_right("electrons-r", -1.0f, ppc);
    e_right.set_density(Density::Slab(coord::y, 1.0, 0, box.y / 2));
    e_right.set_udist(udist_e);
    sim.add_species(e_right);

    Species i_right("ions-r", +100.0f, ppc);
    i_right.set_density(Density::Slab(coord::y, 1.0, 0, box.y / 2));
    i_right.set_udist(udist_i);
    sim.add_species(i_right);

    udist_e.ufl.x = -udist_e.ufl.x;
    udist_i.ufl.x = -udist_i.ufl.x;

    Species e_left("electrons-l", -1.0f, ppc);
    e_left.set_density(Density::Slab(coord::y, 1.0, box.y / 2, box.y));
    e_left.set_udist(udist_e);
    sim.add_species(e_left);

    Species i_left("ions-l", +100.0f, ppc);
    i_left.set_density(Density::Slab(coord::y, 1.0, box.y / 2, box.y));
    i_left.set_udist(udist_i);
    sim.add_species(i_left);

    sim.current->set_filter(Filter::Binomial(coord::x, 2));

    // Run simulation
    float const tmax = 100;

    printf("Running Kelvin-Helmholtz test up to t = %g...\n", tmax);

    Timer timer;

    timer.start();

    diag_kh(sim);

    while (sim.get_t() < tmax)
    {
        sim.advance();
        if (sim.get_iter() % 100 == 0)
            diag_kh(sim);
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());

    sim.energy_info();

    printf("Elapsed time was: %.3f s\n", timer.elapsed(timer::s));
    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void test_bcemf()
{

    // Create simulation box
    uint2 ntiles = {32, 16};
    uint2 nx = {32, 16};
    float2 box = {20.48, 25.6};

    float dt = 0.014;

    Simulation sim(ntiles, nx, box, dt);

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

    sim.emf->add_laser(laser);

    sim.emf->save(emf::e, fcomp::x);
    sim.emf->save(emf::e, fcomp::y);
    sim.emf->save(emf::e, fcomp::z);
    sim.emf->save(emf::b, fcomp::x);
    sim.emf->save(emf::b, fcomp::y);
    sim.emf->save(emf::b, fcomp::z);

    // Run simulation
    float const tmax = 42;

    Timer timer;

    timer.start();

    while (sim.get_t() < tmax)
    {
        sim.advance();
        if (sim.get_iter() % 100 == 0)
        {
            sim.emf->save(emf::e, fcomp::x);
            sim.emf->save(emf::e, fcomp::y);
            sim.emf->save(emf::e, fcomp::z);
            sim.emf->save(emf::b, fcomp::x);
            sim.emf->save(emf::b, fcomp::y);
            sim.emf->save(emf::b, fcomp::z);
        };
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());

    printf("Elapsed time was: %.3f s\n", timer.elapsed(timer::s));
}

void test_bcspec()
{

    uint2 ntiles = {16, 16};
    uint2 nx = {16, 16};

    float2 box = {25.6, 25.6};
    float dt = 0.07;

    Simulation sim(ntiles, nx, box, dt);

    Species electrons("electrons", -1.0f, make_uint2(8, 8));
    electrons.set_density(Density::Sphere(1.0, make_float2(12.8, 12.8), 6.4));
    electrons.set_udist(UDistribution::Cold(make_float3(-1.0e8, 0, 0)));

    auto bc = electrons.get_bc();
    bc.x = {
        .lower = species::bc::reflecting,
        .upper = species::bc::reflecting};
    electrons.set_bc(bc);

    electrons.save_charge();
    sim.current->save(fcomp::x);
    sim.emf->save(emf::e, fcomp::x);

    float const tmax = 51.2;

    printf("Running Species boundary condition test up to t = %g...\n", tmax);

    Timer timer;
    timer.start();

    while (sim.get_t() < tmax)
    {

        sim.advance();
        if (sim.get_iter() % 50 == 0)
        {
            electrons.save_charge();
            sim.current->save(fcomp::x);
            sim.emf->save(emf::e, fcomp::x);
        }
    }

    printf("Simulation complete at t = %g\n", sim.get_t());

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());
}

void test_cathode()
{

    Simulation sim(
        make_uint2(16, 16),      // ntiles
        make_uint2(16, 16),      // nx
        make_float2(25.6, 25.6), // box
        0.07                     // dt
    );

    // Create cathode
    Cathode cathode(
        "cathode",
        +1.0f,            // m_q
        make_uint2(4, 4), // ppc
        1.0e5f            // ufl
    );

    // Set additional cathode parameters
    cathode.n0 = 1.0f;
    cathode.wall = edge::lower;
    cathode.start = -6.4;
    cathode.uth = make_float3(0.1, 0.1, 0.1);
    ;

    auto bc = cathode.get_bc();
    bc.x = {
        .lower = species::bc::open,
        .upper = species::bc::open};

    cathode.set_bc(bc);
    sim.add_species(cathode);

    cathode.save_charge();
    cathode.save();
    sim.current->save(fcomp::x);
    sim.emf->save(emf::e, fcomp::x);

    float const tmax = 51.2;

    printf("Running Cathode test up to t = %g...\n", tmax);

    Timer timer;
    timer.start();

    while (sim.get_t() < tmax)
    {

        sim.advance();

        if (sim.get_iter() % 50 == 0)
        {
            cathode.save_charge();
            cathode.save();
            sim.current->save(fcomp::x);
            sim.emf->save(emf::e, fcomp::x);
        }
    }

    printf("Simulation complete at t = %g\n", sim.get_t());

    timer.stop();

    printf("Elapsed time: %.3f ms\n", timer.elapsed());

    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void test_frozen()
{

    Simulation sim(
        make_uint2(32, 32),        // ntiles
        make_uint2(40, 40),        // nx
        make_float2(128.0, 128.0), // box
        0.07                       // dt
    );

    // Create cathode
    Species electrons("electrons", -1.0f, make_uint2(8, 4));
    sim.add_species(electrons);

    float const tmax = 30.0;

    printf("Running frozen test up to t = %g...\n", tmax);

    Timer timer;
    timer.start();

    while (sim.get_t() < tmax)
    {
        sim.advance();
    }

    timer.stop();

    printf("Simulation complete at t = %g\n", sim.get_t());

    printf("Elapsed time: %.3f ms\n", timer.elapsed());

    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void test_warm()
{

    Simulation sim(
        make_uint2(32, 32),        // ntiles
        make_uint2(32, 32),        // nx
        make_float2(102.4, 102.4), // box
        0.07                       // dt
    );

    uint2 ppc = make_uint2(8, 8);
    float3 uth = make_float3(0.01, 0.01, 0.01);
    float3 ufl = make_float3(0.0, 0.0, 0.0);

    UDistribution::Thermal udist(uth, ufl);

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(udist);

    sim.add_species(electrons);

    electrons.save();

    int const imax = 500;

    printf("Running warm test up to iteration = %d...\n", imax);

    Timer timer;
    timer.start();

    while (sim.get_iter() < imax)
    {
        sim.advance();
    }

    timer.stop();

    sim.current->save(fcomp::x);
    sim.current->save(fcomp::y);
    sim.current->save(fcomp::z);
    electrons.save();

    printf("Simulation complete at t = %g\n", sim.get_t());

    printf("Elapsed time: %.3f ms\n", timer.elapsed());

    std::cout << "Performance: " << sim.get_nmove() / timer.elapsed(timer::s) / 1.e9 << " GPart/s\n";
}

void bench_weibel_sim(FILE *output, uint2 ppc, uint2 nx, uint2 ntiles)
{

    Simulation sim(
        ntiles,                                                      // ntiles
        nx,                                                          // nx
        make_float2(ntiles.x * nx.x * 0.1f, ntiles.y * nx.y * 0.1f), // box
        0.07                                                         // dt
    );

    float3 ufl = {0., 0., 0.6};
    float3 uth = {0.1, 0.1, 0.1};

    UDistribution::Thermal udist(uth, ufl);

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(udist);

    Species positrons("positrons", +1.0f, ppc);
    udist.ufl.z = -udist.ufl.z;
    positrons.set_udist(udist);

    sim.add_species(electrons);
    sim.add_species(positrons);

    float const imax = 500;

    Timer timer;

    timer.start();

    while (sim.get_iter() < imax)
    {
        sim.advance();
    }

    timer.stop();

    fprintf(output, " %d, %d, %d, %d, %d, %d, %.3f, %.3f\n",
            ppc.x, ppc.y, nx.x, nx.y, ntiles.x, ntiles.y,
            timer.elapsed(timer::s),
            sim.get_nmove() / timer.elapsed(timer::s) / 1.e9);
}

void benchmark_weibel()
{

    std::vector<uint2> ppc_list{
        {1, 1}, {2, 1}, {2, 2}, {4, 2}, {4, 4}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}};

    std::vector<uint2> ntiles_list{
        {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}, {64, 32}, {64, 64}, {128, 64}, {128, 128}, {256, 128}, {256, 256}, {512, 256}, {512, 512}, {1024, 512}, {1024, 1024}};

    std::vector<uint2> nx_list{
        {4, 4}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {32, 16}, {32, 32}, {40, 32}, {40, 40}};

    FILE *output = fopen("benchmark_weibel.csv", "w");
    fprintf(output, "ppc.x, ppc.y, nx.x, nx.y, ntiles.x, ntiles.y, time, perf\n");

    /*
        uint2 ppc = {8,8};
        uint2 nx = {16,16};
        uint2 ntiles = {16,16};
    */

    int ntests = ppc_list.size() *
                 nx_list.size() *
                 ntiles_list.size();

    int i = 0;
    for (uint2 ppc : ppc_list)
        for (uint2 nx : nx_list)
            for (uint2 ntiles : ntiles_list)
            {
                int64_t vol = nx.x * nx.y * ntiles.x * ntiles.y;
                int64_t npart = vol * ppc.x * ppc.y;

                printf("[%3d/%3d] %d, %d, %d, %d, %d, %d (%ld)\n", i++, ntests,
                       ppc.x, ppc.y, nx.x, nx.y, ntiles.x, ntiles.y, npart);

                if (npart < 104857600)
                {
                    cudaDeviceReset();
                    bench_weibel_sim(output, ppc, nx, ntiles);
                    fflush(output);
                }
            }
    fclose(output);
    printf("tests complete.\n");
}

class Test
{
public:
    std::string name;
    void (*func)();

    Test(std::string name, void (*func)()) : name(name), func(func) {}

    void run()
    {
        if (func != NULL)
        {
            std::cout << "[" << name << "]\n";
            (*func)();
            std::cout << "[test complete]\n";

            cudaDeviceReset();
        }
    }
};

int main(int argc, char *argv[])
{

    std::vector<Test> tests;

    tests.push_back(Test("absorb", &test_absorb));
    tests.push_back(Test("emf", &test_emf));
    tests.push_back(Test("move_window", &test_move_window));
    tests.push_back(Test("filter", &test_filter));
    tests.push_back(Test("weibel", &test_weibel));
    tests.push_back(Test("weibel_large", &test_weibel_large));
    tests.push_back(Test("lwfa", &test_lwfa));
    tests.push_back(Test("mushroom", &test_mushroom));
    tests.push_back(Test("kh", &test_kh));
    tests.push_back(Test("bcemf", &test_bcemf));
    tests.push_back(Test("bcspec", &test_bcspec));
    tests.push_back(Test("cathode", &test_cathode));
    tests.push_back(Test("frozen", &test_frozen));
    tests.push_back(Test("warm", &test_warm));
    tests.push_back(Test("benchmark", &benchmark_weibel));
    
    switch (argc)
    {
    case 2: {
        std::string input(argv[1]);
        bool found = false;
        for (auto test : tests)
        {
            if (input == test.name)
            {
                test.run();
                found = true;
                break;
            }
        }

        if (!found)
        {
            std::cerr << "Test '" << input << "' not found.\n";
            std::cerr << "Available tests are:\n";
            for (auto test : tests)
            {
                std::cerr << "- " << test.name << "\n";
            }
            exit(1);
        }
        break;
    }
    case 1: {
        Test("weibel", &test_weibel).run();
        break;
    }
    default:
        std::cerr << "Usage: " << argv[0] << " <test-name>\n";
        exit(1);
    }

    return 0;
}