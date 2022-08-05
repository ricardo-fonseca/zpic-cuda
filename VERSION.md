# ZPIC CUDA Development

## 2022.8.5

* Implements particle move_deposit
  * Fully relativistic leap-frog implementation complete
  * Acceleration (du_dt) will be implemented in a separate kernel to save on shared memory
  * Initial implementation of current deposit equal to OpenACC version
    * Am convinced that trajectory splitting (aka virtual particles), with virtual particles stored in shared memory is the way to go
* Improvements on the random module
  * Enables use on host code
  * Adds real1, real2 and real3 flavours
  * Adds state initializer. On `__device__` code this ensures a different state for each thread
* Adds copy_to_gc() and add_from_gc() methods to VFLD
  * Removes the old update_gc() method
  * Also fixed a bug on Field::add_from_gc()
* Fixes initializers for Current, EMF and Species objects
* Adds VFLD::save() method
  * Removes the old zdf_save_tile_vfld() routine
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
