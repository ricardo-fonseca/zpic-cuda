# ZPIC CUDA Development

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
