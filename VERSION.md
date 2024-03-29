# ZPIC CUDA Development

## TO-DO

* Thermal boundaries for particles
* Absorbing boundaries for fields

## 2023.07.24

* Modifies particle injection strategy to minimize memory collisions in electric current
  deposit. Consecutive particles in the particle buffer are now injected in different
  cells.
* Adds performance / energy information to some tests
* Adds `Species::np()` method to get number of particles directly from species object
* Updates kernel filtering to process periodic boundaries correctly

## 2023.04.27

* Improves on moving window implementation, particle shift is now done inside the advance
  deposit. There is a new method `Species::advance_deposit_mov_window()` that should be
  done for this purpose.
* Makes `mk3` particle tile sort the default
* Improves performance of the `kernel3_x` CUDA kernel code used for current filtering
* Improves performance of the `copy_gcx` and `copy_gcy` CUDA kernels
* Species initialization now sets the tile `offset` values according to the number of
  particles to be injected at t = 0 
* Increases (longitudinal) size of LWFA test
* Cosmetic update to LWFA notebook

## 2023.04.14

* Implements moving window support for low-memory particle sorter
* Implements `Species::np_inject()` method that gets the number of particles to be injected
  per tile.
* Sets the particle buffer size to 1.2 times the grid volume times ppc
* Fixes issues with the LWFA test
* Adds `filter.cuh` to `zpic.cuh`
* Adds `ABORT()` macro, and includes additional information in the `CHECK_ERR()` macro.

## 2023.04.01

* Changes the initial particle ordering. The new version places particles in different
  cells in contiguous positions in the particle buffer, which reduces memory collisions
  at electric current deposit
* Minor optimizations to current deposition: access buffer as float[] (instead of 
  float3[]), reorder operations for (near) linear memory access 
* Adds test selection through (basic) command line option
* Adds `benchmark` script to repeat a test several times and get average performance

## 2023.01.10

* Implements large particle sorter mk4

## 2023.01.9

* Implements low-memory particles sorter mk1, mk2, and mk3. mk1 and mk2 copy all
  particles to a new buffer, mk3 does it in place.
* Fixes minor issue with `Particles::validate()` (would not work properly after calling
  `Particles::np()`)
* Adds `cudaDeviceReset()` before calls to `exit()`. This prevents the process from 
  sometimes hanging when `deviceCheck()` and other routines would fail, forcing a reboot.

## 2022.12.27

* Improves on particle tile sorter. Particles are checked for leaving tile inside the
  pusher.

## 2022.11.22

* Groups particle data into `t_part_data` structure for simpler kernel calls

## 2022.11.17

* Adds performance measurements
* Adds "frozen" (0 temperature, 0 fluid velocity) test
* Implements new particle tile sorter using a temporary buffer for storing the indices
  of particles moving from tile

## 2022.11.7

__CRITICAL:__ Fixes critical issues

* Fixes a critical issue in non-uniform density profiles
  * A `block.sync()` was missing after setting the shared variable `np`, randomly
    leading to improper injection.
* Changes `Species` class initialization behavior
  * Adds a `Species::initialize()` method, to be called by the simulation object
    to complete initialization.
  * After the species is created the user may freely change the species initialization
    parameters
  * When the species is added to the simulation (`Simulation::add_species()`) the
    `Simulation` object will call the `Species::initialize()` method, creating the
    `particles` data structure and initializing species particles.

## 2022.11.2

__CRITICAL:__ Fixes critical issues

* Fixes critical issue in particle tile sorter (`_bnd_out()`)
  * A shared variable was being initialized after synchronization, randomly 
    leading to corrupted particle positions i
* Adds Wall injection for particles
* Adds `Density::None` class
* Adds `edge::pos` (lower, upper) constants


## 2022.10.27

* Reflecting boundaries for currents
* Reflecting boundaries for particles

## 2022.10.26

* PEC and PMC boundaries for EM fields
* Use `bnd<T>` type for boundary-related information (number of guard cells, bc type)

## 2022.10.19

* Adds documentation file
* Adds support for step and slab profiles along y
* Adds total particle energy output to energy info

## 2022.9.9

__CRITICAL:__ Fixes critical issues

* Fixes critical issue in Field and VectorField `::add_from_gc()` methods.
  * Adding in values from an upper neighbor was not done properly.
* Fixes critical issue with `pusher::euler`, rotation matrix was not properly implemented

* Adds Courant condition checking to EMF class
* Adds electric current filtering along y
* Adds seeding parameter to "Species::set_u". This enables different species to use a different seed for the random number generator.


## 2022.9.6

* Implements energy diagnostics for both particles and fields
  * This requires some minor changes to the `dudt_*` routines that add negligible calculations
* Implements simple `Simulation::energy_info()` method to report global simulation energy
* Removes `Simulation::get_species(int)` method, it was redundant (`Simulation.species[i]` is accessible)
* Removes const qualifier from `Simulation::get_species(string)`

## 2022.9.5

* Particle pusher can now be set using the `Species.push_type` member variable
  * Defaults to `pusher::boris` (the traditional Boris push)
* Fixes issue with particle tile sort with periodic boundaries
* Implements `UDistribution::*` classes to handle temperature distributions:
  * Implements `None` (0), `Cold` (`ufl` only), and `Thermal` (`uth`, `ufl`)
  * Also implements `ThermalCorr` which includes a correction for local `ufl` fluctuations due to a low number of particles per cell
* Laser pulse filtering can now be controlled by the `laser.filter` parameter.
  * This defaults to 1, which corresponds to a 1-level compensated binomial filter. Setting it to 0 disables filtering.

## 2022.9.3

* Adds electric current filtering (along x only)
  * Filtering parameters defined by `Filter::` classes: `Filter::None`, `Filter::Binomial`, and `Filter::Compensated`
  * Controlled by `Current::set_filter()` method. Can be changed throughout the simulation
* Moves laser pulses into `Laser::` namespace, e.g. `Laser::Gaussian`
* Lasers pulses are now filtered with `Filter::Compensated(1)` before injection
* Fixes inconsistency in phase-space units (`ux`, `uy`, and `uz` units are now "c")
* Adds `visxd.grid2d_fft()` routine to plot the FFT of 2D grid data

## 2022.9.2

__MILESTONE:__ The code has completed the LWFA simulation test.

* Implements moving window algorithm
* New MovingWindow class
  * Handles moving window data (needs to move, distance moved, etc.)
* New Density class
  * Describes all density profiles
  * Handles particle injection
* Particles class
  * Adds `cell_shift` method to shift particle cells by a specified amount
  * Adds `periodic` member to control `tile_sort`()` behavior (global periodic boundaries were previously enforced)
* Field and VectorField classes
  * Adds `x_shift_left` method to shift data left by a specified amount
  * Adds `periodic` member to control `copy_to_gc()` and `add_from_gc()` method behaviour (global periodic boundaries were previously enforced)
* Adds math constants to `utils.h`
* Adds `bnd` class to describe boundary data (x/y, lower/upper)

## 2022.8.15 - Suplemental

* Optimizes `particles::tile_sort()` routine
  * See `optimizations.md` for details
  * Local speedup of 5.54 x
* Optimizes `Species::move_deposit()`
  * See `optimizations.md` for details
  * Local speedup of 1.55 x
* Minor changes
  * Cleans up info messages.
  * Replaces multiple `_gather_*_kernel()` functions with a single templated `_gather_quant < quant >()` function

## 2022.8.15

* Adds phasespace diagnostics
  * Both 1D and 2D phasespaces can be generated with any of (x,y,ux,uy,uz) values
* Makes Field and VectorField `::zero()` method asynchronous (using `cudaMemsetAsync()`)
* Fixes ouput units inconsistency (reference frequency is now $\omega_n$)
* Updates visualization tools
  * Adds `visxd.grid1d()` routine to generate line plots from 1d data
  * Adds `visxd.grid()` routine that automatically chooses between `visxd.grid1d()` and `visxd.grid2d()` based on file grid dimensions
  * Renames `plot_sfield()` to `plot_data()`. It now works for both 2d and 1d files.

## 2022.8.14

__MILESTONE:__ The code has successfully completed the Weibel simulation test.

* Species class
  * Fixes a critical issue with the trajectory splitter and out of plane current deposition (this issue was due to the change to [-0.5,.05[ positions)
  * Fixes a critical issue with E field interpolation, again due to the change to [-0.5,.05[ positions
  * Fixes issue with non-uniform particle injection. There was a block sync. operation missing before storing the total number of particles so some particles were being left out
  * Fixes issue with sphere profile injection (was injecting a slab)
* Implements new ZDF c++ interface, removing the `tile_zdf*` files
  * The new interface, defined in the `zdf-cpp.h` file, puts C zdf declarations inside a namespace (`zdf::`), and exposes a minimal amount of types with new names (e.g. `zdf::file` corresponding to the C `t_zdf_file`)
* Minor changes
  * Timer class can now return timing information in other time units (s, ms, us, or ns)
  * Renamed `VFLD` class `VectorField` class
* visxd
  * Add option to linearly scale data for `grid2d` plots, and other plot formatting options
  * Adds new `vfield2d()` routine to plot the magnitude of a vector field

## 2022.8.13

* Adds new top level Simulation class
  * Holds one EMF and one Current objects
  * Holds a vector of Species objects
  * Implements methods for adding / getting Species objects
  * Implements method for adding laser pulses
  * Implements advance method that advances the simulation 1 timestep
* main file
  * Has been restructured using new classes
    * Previous tests moved into `test_emf` and `test_sort_deposit`
  * Implements the canonical Weibel simulation test!
* Species class
  * Updates code to the new [-0.5,0.5[ particle position scheme (distance from center of the cell)
  * Removes `inject_particles` and `set_u` calls from constructor, objects are now created without any particles
  * Fixes the uniform density injection kernel
  * Implements new `density::parameters` type to specify injection profiles (which are no longer hardcoded)
  * Implements new `Species::move()` method where no current is deposited
* EMF class
  * Adds field advance including electric current
  * Renames `report` -> `save` (finally!)
* Current class
  * Fixes issue in `Current::advance()`, data was not being copied to gc causing EM field solver issue
  * Renames `report` -> `save` (finally!)
* Particles classe
  * Particles positions now use the OSIRIS [-0.5,0.5[ position scheme (distance from center of the cell) for improved numerics
  * Improves `Particles::validate()` method, now checks all particle data and allows for overflow
  * Adds `Particles::check_tiles()` method to check on the number of particle per tiles (debug only)
* VFLD class
  * Improves the `VFLD::save()` method, all internal info is now set in this routine
* Field class
  * Improves the `Field::save()` method, all internal info is now set in this routine
  * Moved memory allocation now uses `malloc_host()` and `free_host()` routines everywhere
* Random class
  * Removed host support (was breaking device code), may be reinstated at a later time
  * Improved the per cell seed initialization
* Minor changes
  * adds `clean` script to remove diagnostic output
  * enum values defining multiple parameters have been moved into namespaces
  * replaces the `deviceSynchronize()` macro with a better `deviceCheck()` one that reports both synchronous and assynchronous errors (if any)
* Moves visualization routines to new `visxd` python module
  * `plot2d` routine has been renamed `grid2d` (and it may change again in the future ;-) and adds a new `vsim` (symmetric color scale) parameter
  * Replaces `FileNotFoundError` exceptions with simple error message.

## 2022.8.12

* Renames `tile_part` to `Particles` class, `particles` to `Species` class
  * Adds save method to `Particles` to save particle data
  * Implements device np and np_exscan functions
  * Implements overloaded versions of gather() function, improved performance when gathering multiple quantities sequentially
  * Adds validade method (currently only checks cells/positions)
  * Adds tile_sort() method: reassigns particles that crossed tile boundaries

* Implements full species advance deposit
  * Implements particle push (velocity advance)
    * Field interpolation using linear splines
    * Velocity advance using traditional relativistic Boris pusher (linearized rotation tangent)
  * Implements move deposit
    * Particle motion using traditional leap-frog method
    * Current deposition using linear splines and Villasenor-Buneman method

* Changes to VFLD + Field classes
  * Removed CPU copies, all calculations on GPU
  * Makes tile structure data unsigned
  * Constructor now takes ntiles and nx. Adds overloaded constructor for gc = 0

* Minor improvements
  * Adds `const` and `__restrict__` qualifiers all over
  * Uses cooperative groups sync() instead of __syncthreads() for thread block synchronization
  * Adds a number of utility functions for data allocation / deallocation, copy, simple reduction / scan operations, and scalar variables in device memory (accessible from host)

## 2022.8.5

* Implements particle move_deposit
  * Fully relativistic leap-frog implementation complete
  * Acceleration (push) will be implemented in a separate kernel to save on shared memory
  * Initial implementation of current deposit equal to OpenACC version
    * Am convinced that trajectory splitting (aka virtual particles), with virtual particles stored in shared memory is the way to go
* Improvements on the `random` module
  * Enables use on host code
  * Adds `real1`, `real2` and `real3` flavours
  * Adds state initializer. On `__device__` code this ensures a different state for each thread
* Adds `copy_to_gc()` and `add_from_gc()` methods to VFLD
  * Removes the old `update_gc()` method
  * Also fixed a bug on `Field::add_from_gc()`
* Fixes initializers for Current, EMF and Species objects
* Adds `VFLD::save()` method
  * Removes the old `zdf_save_tile_vfld()` routine
* Adds histogram plots and several plot options for all visXD notebook routines
* Adds code documentation in several places

## 2022.8.4

* Improves particle initialization
  * Implements step, slab, and sphere profiles
  * Implements uth / ufl intialization
* Implements simple random module for use in CUDA kernels
  * Each thread must use its own state

## 2022.8.3

* Implements initial Species class
  * Only uniform density / 0 temperature particle injection
  * No pusher or current deposit yet
  * Charge deposit
  * Particle data and charge density diagnostics
* Implements tile_part class
  * Includes gather function for particle data
    * Can be optimized by caching the result of the scan operation to find positions in output buffer when gathering multiple quantities sequentially
* Implements Field class for float grids (e.g. Charge)
  * Will be made into a template class, may replace VFLD in the future
  * Implements copy_to_gc() (copy to guard cells from neighboring tile data) and add_from_gc() (adds to tile data from neighboring tiles guard cells) methods for periodic boundaries
  * Implements save() method

## 2022.7.14

* Implements initial EMF class, including
  * Yee solver
  * Basic reporting (field save)
* Implements plane wave and Gaussian beams
* Adds update_gc method to VFLD object to copy guard-cell values from neighboring tiles

## 2022.7.9

Moves the code to C++:
* Implements C++ classes for Timer (using CUDA events) and VFLD (tiled float3 grid)
* Gather function organized by tile, kernel used 2D block grid (nxtiles.x,nxtiles.y)
* Modifies ZDF header for use with C++ code

## 2022.7.7

Initial version of tile VFLD object:
* C code
* Gather function organized by global cell index, kernel used 1D block grid (ncells / nthreads)
* ZDF output implemented, includes simple notebook for visualizing 2D scalar grid data.
