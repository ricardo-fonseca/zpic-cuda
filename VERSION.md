# ZPIC CUDA Development

## TO-DO

* Refelecting boundaries for currents
* Reflecting and thermal boundaries for particles
* Improve particle tile sorter, make it production ready

## 2022.10.26

* PEC and PMC boundaries for EM fields
* Use bnd<T> type for boundary related information (number of guard cells, bc type)

## 2022.10.19

* Adds documentation file
* Adds support for step and slab profiles along y
* Adds total particle energy output to energy info

## 2022.9.9

__CRITICAL:__ Fixes critical issues

* Fixes critical issue in Field and VectorField `::add_from_gc()` method.
  * Adding in values from upper neighbour was not done properly.
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
  * Implements `None` (0), `Cold` (ufl only), and `Thermal` (uth, ufl)
  * Also implements `ThermalCorr` that includes a correction for local ufl fluctuations due to low number of particles per cell
* Laser pulse filtering can now be controlled by the `laser.filter` parameter.
  * This defaults to 1, which corresponds to a 1 level compensated binomial filter. Setting it to 0 disables filtering.

## 2022.9.3

* Adds electric current filtering (along x only)
  * Filtering parameters defined by `Filter::` classes: `Filter::None`, `Filter::Binomial` and `Filter::Compensated`
  * Controled by `Current::set_filter()` method. Can be changed over the course of the simulation
* Moves laser pulses into `Laser::` namespace, e.g. `Laser::Gaussian`
* Lasers pulses are now filtered with `Filter::Compensated(1)` before injection
* Fixes inconsistency in phasespace units (ux, uy, and uz units are now "c")
* Adds `visxd.grid2d_fft()` routine to plot the FFT of 2D grid data

## 2022.9.2

__MILESTONE:__ The code has successfully completed the LWFA simulation test.

* Implements moving window algorithm
* New MovingWindow class
  * Handles moving window data (needs to move, distance moved, etc.)
* New Density class
  * Describes all density profiles
  * Handles particle injection
* Particles class
  * Adds `cell_shift` method to shift particle cells by a specified amount
  * Adds `periodic` member to control `tile_sort()` behaviour (global periodic boundaries were previously enforced)
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
