# ZPIC-CUDA optimizations

## New boundary move routines

Reference:

```text
Elapsed time was: 5.148 s
==45708== Profiling application: ./zpic-cuda
==45708== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   30.79%  1.58683s      1000  1.5868ms  1.0473ms  9.8765ms  _move_deposit_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, unsigned int, uint2, float2, float, float2)
                   29.89%  1.54037s      1000  1.5404ms  697.75us  2.7238ms  _bnd_out_y(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
                   29.87%  1.53952s      1000  1.5395ms  724.02us  2.7872ms  _bnd_out_x(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
                    8.00%  412.40ms      1000  412.40us  362.72us  492.73us  _push_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, float3*, unsigned int, uint2, float)
````

1. Replace the scan / reduce operations with atomic adds
    * Atomics are now very fast (especially in shared memory), and they will only be called for particles crossing the node
    * Immediately allows for arbitrary number of threads (previous version limited to 1 warp)

```text
Elapsed time was: 2.619 s
==45885== Profiling application: ./zpic-cuda
==45885== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.19%  1.57969s      1000  1.5797ms  1.0398ms  9.8454ms  _move_deposit_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, unsigned int, uint2, float2, float, float2)
                   15.75%  413.34ms      1000  413.34us  362.33us  496.28us  _push_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, float3*, unsigned int, uint2, float)
                   10.63%  279.05ms      1000  279.05us  235.58us  349.05us  _bnd_out_r1_x(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
                   10.58%  277.69ms      1000  277.69us  227.65us  348.06us  _bnd_out_r1_y(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
```

Speedup was __5.54 x__ for the `bnd_out` routines, __1.97 x__ overall.

2. Replace the different `bnd_out_x` and `bnd_in_y` routines with a template based one (and also `bnd_in_*`)
    * This makes the code easier to read, it should have 0 impact on performance

```text
Elapsed time was: 2.617 s
==46694== Profiling application: ./zpic-cuda
==46694== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.22%  1.57901s      1000  1.5790ms  1.0445ms  9.8669ms  _move_deposit_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, unsigned int, uint2, float2, float, float2)
                   15.70%  411.58ms      1000  411.58us  362.75us  494.56us  _push_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, float3*, unsigned int, uint2, float)
                   10.65%  279.18ms      1000  279.18us  235.97us  344.09us  void _bnd_out<coord::cart>(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
                   10.59%  277.69ms      1000  277.69us  228.64us  345.44us  void _bnd_out<coord::cart>(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
````

3. Prefetching all particle data before conditional copy to tmp memory
    * This allows coallesced access to particle data which can improve on the random access currently in use
    * Altough requiring more memory reads memory access performance can be better; also this will move particle data into L2

```text
Elapsed time was: 2.603 s
==49526== Profiling application: ./zpic-cuda
==49526== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.66%  1.58256s      1000  1.5826ms  1.0436ms  9.8451ms  _move_deposit_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, unsigned int, uint2, float2, float, float2)
                   15.81%  412.38ms      1000  412.38us  359.13us  498.04us  _push_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, float3*, unsigned int, uint2, float)
                   10.35%  270.03ms      1000  270.03us  229.09us  340.51us  void _bnd_out_r1<coord::cart>(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
                   10.32%  269.26ms      1000  269.26us  223.20us  335.26us  void _bnd_out_r1<coord::cart>(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
```

This leads to a marginal speedup of 1.03x.

3. Does a n1, n2 reduce operation at warp level, then only each warp.thread_rank 0 calls atomicAdd()
    * No noticeable difference, but it looks nicer so we keep it.


## Current deposit

1. Use 3 virtual particles per thread stored on local variables (hopefully registers)
    * Implemented using a switch() based splitter. Variations on the splitter had no impact on performance
    * After splitting each thread runs 1, 2 or 3 dep_current_seg calls.
    * Divergence may only happen on 2nd and 3rd calls so this allows for much better thread divergence management:

```text
Elapsed time was: 2.039 s
==62592== Profiling application: ./zpic-cuda
==62592== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.81%  1.01818s      1000  1.0182ms  698.26us  10.880ms  _move_deposit_kernel_b(ParticlesTile const *, int2*, float2*, float3*, float3*, unsigned int, uint2, float2, float, float2)
                   20.20%  412.86ms      1000  412.86us  361.18us  501.63us  _push_kernel(ParticlesTile const *, int2*, float2*, float3*, float3*, float3*, unsigned int, uint2, float)
                   13.21%  269.98ms      1000  269.98us  229.12us  335.45us  void _bnd_out_r1<coord::cart>(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
                   13.15%  268.84ms      1000  268.84us  222.33us  332.89us  void _bnd_out_r1<coord::cart>(int, ParticlesTile*, int2*, float2*, float3*, ParticlesTile*, int2*, float2*, float3*)
```

This leads to speedup of __1.55 x__ in `move_deposit`, __1.28 x__ overall.