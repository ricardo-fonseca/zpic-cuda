#ifndef __UTIL__
#define __UTIL__

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

#include <iostream>

namespace fcomp {
    enum cart  { x = 0, y, z };
}

namespace coord {
    enum cart  { x = 0, y };
}


#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   ///< pi
#endif

#ifndef M_PI_2
#define M_PI_2      1.57079632679489661923132169163975144   ///< pi/2
#endif

#ifndef M_PI_4
#define M_PI_4      0.785398163397448309615660845819875721  ///< pi/4
#endif

#define CHECK_ERR( err_, msg_ ) { \
    if ( err_ != cudaSuccess ) { \
        std::cerr << "(*error*) " << msg_ << std::endl; \
        std::cerr << "(*error*) code: " << err_ << ", reason: " << cudaGetErrorString(err_) << std::endl; \
        exit(1); \
    } \
}

/**
 * @brief Checks if there are any synchronous or asynchronous errors from CUDA calls
 * 
 * If any errors are found the routine will print out the error messages and exit
 * the program
 */
#define deviceCheck() { \
    auto err_sync = cudaPeekAtLastError(); \
    auto err_async = cudaDeviceSynchronize(); \
    if (( err_sync != cudaSuccess ) || ( err_async != cudaSuccess )) { \
        std::cerr << "(*error*) CUDA device is on error state at " << __func__ << "()\n"; \
        std::cerr << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        if ( err_sync != cudaSuccess ) \
            std::cerr << "(*error*) Sync. error message: " << cudaGetErrorString(err_sync) << " (" << err_sync << ") \n"; \
        if ( err_async != cudaSuccess ) \
            std::cerr << "(*error*) Async. error message: " << cudaGetErrorString(err_async) << " (" << err_async << ") \n"; \
        exit(1); \
    } \
}

/**
 * @brief Allocates page-locked memory on the host 
 * 
 * In case of failure the routine will isse an error and abort.
 * 
 * @tparam T        Type of data
 * @param buffer    (out) Pointer to allocated memory
 * @param size      Number of T elements to allocate
 * @return T*       Pointer to allocated memory
 */
template < typename T >
T * malloc_host( T * & buffer, size_t const size, std::string file, int line ) {
    auto err = cudaMallocHost( &buffer, size * sizeof(T) );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Allocation failed on file " << file << ":" << line << "\n";
        std::cerr << "(*error*) Unable to allocate " << size << " elements of type " << typeid(T).name() << " on host.\n";
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
    return buffer;
}

#define malloc_host( buffer, size ) malloc_host( (buffer), (size), __FILE__, __LINE__ )

/**
 * @brief Frees host memory previously allocated by malloc_host
 * 
 * @tparam T        Type of data
 * @param buffer    Pointer to allocated memory
 */
template < typename T >
void free_host( T * buffer, std::string file, int line ) {
    if ( buffer != nullptr ) {
        auto err = cudaFreeHost( buffer );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) deallocation failed on file " << file << ":" << line << "\n";
            std::cerr << "(*error*) Unable to deallocate " << typeid(T).name() << " buffer at " << buffer << " from host.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            exit(1);
        }
    }
}

#define free_host( buffer ) free_host( (buffer), __FILE__, __LINE__ )


/**
 * @brief Allocates memory on the device 
 * 
 * @tparam T        Type of data
 * @param buffer    (out) Pointer to allocated memory
 * @param size      Number of T elements to allocate
 * @return T*       Pointer to allocated memory
 */
template < typename T >
T * malloc_dev( T * & buffer, size_t const size, std::string file, int line ) {
    auto err = cudaMalloc( &buffer, size * sizeof(T) );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Allocation failed on file " << file << ":" << line << "\n";
        std::cerr << "(*error*) Unable to allocate " << size << " elements of type " << typeid(T).name() << " on device.\n";
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
    return buffer;
}

#define malloc_dev( buffer, size ) malloc_dev( (buffer), (size), __FILE__, __LINE__ )

/**
 * @brief Frees device memory previously allocated by malloc_dev 
 * 
 * @tparam T        Type of data
 * @param buffer    Pointer to allocated memory
 */
template < typename T >
void free_dev( T * buffer , std::string file, int line ) {
    if ( buffer != nullptr ) {
        auto err = cudaFree( buffer );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) deallocation failed on file " << file << ":" << line << "\n";
            std::cerr << "(*error*) Unable to deallocate " << typeid(T).name() << " buffer at " << buffer << " from device.\n";
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
            exit(1);
        }
    }
}

#define free_dev( buffer ) free_dev( (buffer), __FILE__, __LINE__ )

template < typename T >
void devhost_memcpy( T * const __restrict__ h_out, T const * const __restrict__ d_in, size_t const size ) {
    auto err = cudaMemcpy( h_out, d_in, size * sizeof(T), cudaMemcpyDeviceToHost );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to copy " << size << " elements of type " << typeid(T).name() << " from device to host.\n";
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
}

template < typename T >
void hostdev_memcpy( T * const __restrict__ d_out, T const * const __restrict__ h_in, size_t const size ) {
    auto err = cudaMemcpy( d_out, h_in, size * sizeof(T), cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) {
        std::cerr << "(*error*) Unable to copy " << size << " elements of type " << typeid(T).name() << " from host to device.\n";
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
}

namespace device {

namespace {

template < typename T >
__global__ 
/**
 * @brief CUDA Kernel for reduction
 * 
 * The reduction result is added onto the output variable (i.e. you may want to set it to 0
 * before calling this kernel)
 * 
 * @param data          Data to scan
 * @param size          Size of data to scan
 * @param reduction     Result of reduction operation on entire dataset
 */
void _reduction_kernel( T * __restrict__ d_data, unsigned int const size, T * __restrict__ reduction ) {

    auto grid = cg::this_grid();
    auto warp = cg::tiled_partition<32>(grid);

    // In case there are fewer threads than data points
    T v = 0;
    for( auto i = grid.thread_rank(); i < size; i += grid.num_threads() )
        v += d_data[ i ];

    v = cg::reduce( warp, v, cg::plus<T>());

    if ( warp.thread_rank() == 0 ) atomicAdd( reduction, v );
}

template < typename T >
__global__ 
/**
 * @brief CUDA Kernel for inclusive scan
 * 
 * Kernel must be launched with a single block and arbitrary number of threads
 * 
 * @param data          Data to scan
 * @param size          Size of data to scan
 * @param reduction     Result of reduction operation on entire dataset
 */
void _inclusive_scan_kernel( T * __restrict__ data, unsigned int const size, T * __restrict__ reduction ) {

    // 32 is the current maximum number of warps
    __shared__ T tmp[ 32 ];
    __shared__ T prev;
    

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Contribution from previous warp
    prev = 0;
    
    for( unsigned int i = block.thread_rank(); i < size; i += block.num_threads() ) {
        auto v = data[i];

        v = cg::inclusive_scan( warp, v, cg::plus<T>());
        if ( warp.thread_rank() == warp.num_threads() - 1 ) tmp[ warp.meta_group_rank() ] = v;
        block.sync();

        // Only 1 warp does this
        if ( warp.meta_group_rank() == 0 ) {
            auto t = tmp[ warp.thread_rank() ];
            t = cg::exclusive_scan( warp, t, cg::plus<T>());
            tmp[ warp.thread_rank() ] = t + prev ;
        }
        block.sync();

        v += tmp[ warp.meta_group_rank() ];
        data[i] = v;

        if ((block.thread_rank() == block.num_threads() - 1) || ( i + 1 == size ) ) prev = v;
        block.sync();
    }

    if ( block.thread_rank() == 0 ) *reduction = prev;
}

template < typename T >
__global__ 
/**
 * @brief CUDA Kernel for inclusive scan
 * 
 * Kernel must be launched with a single block and arbitrary number of threads
 * 
 * @param data          Data to scan
 * @param size          Size of data to scan
 * @param reduction     Result of reduction operation on entire dataset
 */
void _exclusive_scan_kernel( T * __restrict__ data, unsigned int const size, T * __restrict__ reduction ) {

    // 32 is the current maximum number of warps
    __shared__ T tmp[ 32 ];
    __shared__ T prev;

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Contribution from previous warp
    prev = 0;

    for( unsigned int i = block.thread_rank(); i < size; i += block.num_threads() ) {
        auto s = data[i];

        auto v = cg::exclusive_scan( warp, s, cg::plus<T>());
        if ( warp.thread_rank() == warp.num_threads() - 1 ) tmp[ warp.meta_group_rank() ] = v + s;
        block.sync();

        // Only 1 warp does this
        if ( warp.meta_group_rank() == 0 ) {
            auto t = tmp[ warp.thread_rank() ];
            t = cg::exclusive_scan( warp, t, cg::plus<T>());
            tmp[ warp.thread_rank() ] = t + prev;
        }
        block.sync();

        // Add in contribution from previous threads
        v += tmp[ warp.meta_group_rank() ];
        data[i] = v;

        if ((block.thread_rank() == block.num_threads() - 1) || ( i + 1 == size ) )
            prev = v + s;

        block.sync();
    }

    if ( block.thread_rank() == 0 ) *reduction = prev;
}

}

/**
 * @brief Class representing a scalar variable in device memory
 * 
 * This class simplifies the creation of scalar variables in unified memory.
 * Note that getting the variable in the host (get()) will always trigger a 
 * device synchronization.
 * 
 * @tparam T    Variable datatype
 */
template< typename T> class Var {
    private:

    T * data;

    public:

    __host__
    /**
     * @brief Construct a new Var<T> object
     * 
     */
    Var() {
        auto err = cudaMallocManaged( &data, sizeof(T) );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to allocate managed memory for device::Var" << std::endl;
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
    }

    __host__
    /**
     * @brief Construct a new Var<T> object and set value to val
     * 
     * @param val 
     */
    Var( const T val ) : Var() {
        set( val );
    }

    __host__
    /**
     * @brief Destroy the Var<T> object
     * 
     */
    ~Var() {
        auto err = cudaFree( data );
        if ( err != cudaSuccess ) {
            std::cerr << "(*error*) Unable to free managed memory for device::Var" << std::endl;
            std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
    }

    __host__
    /**
     * @brief Sets the value of the Var<T> object
     * 
     * @param val       value to set
     * @return T const  returns same value
     */
    T const set( const T val ) {
        *data = val;
        return val;
    }

    __host__
    /**
     * @brief Overloaded assignment operation for setting the object value
     * 
     * @param val 
     * @return T 
     */
    T operator= (const T val) {
        return set(val);
    }

    __host__ T
    /**
     * @brief Get object value. Device operations will be synchcronized first.
     * 
     */
    const get() const { 
        // Ensure any kernels still running terminate
        cudaDeviceSynchronize();
        return *data;
    }

    __host__ 
    /**
     * @brief Pointer to variable data
     * 
     * @return T* 
     */
    T * ptr() const { return data; }

#if 0
    /**
     * @brief 
     * 
     * @tparam U 
     * @param os 
     * @param d 
     * @return std::ostream& 
     */
    template< class U >
    friend auto operator<< (std::ostream& os, device::Var<U> const & d) -> std::ostream& { 
        return os << d.get();
    }
#endif

};

/**
 * @brief Perform reduction operation on CUDA device, return result on host
 * 
 * @tparam T    Datatype
 * @param data  Data buffer on device
 * @param size  Number of data points
 * @return T    Reduction of data buffer
 */
template< typename T >
T reduction( T const * const __restrict__ data, unsigned int const size ) {

    device::Var<T> sum(0);
    unsigned int grid = (size-1) >> 5 + 1;
    _reduction_kernel <<< grid, 32 >>> ( data, size, sum.ptr() );
    return sum.get();
}

/**
 * @brief Perform inclusive scan operation on CUDA device, return reduction on host
 * 
 * @tparam T    Datatype
 * @param data  Data buffer on device
 * @param size  Number of data points
 * @return T    Reduction of data buffer
 */
template< typename T >
__host__
T inclusive_scan( T * const __restrict__ data, unsigned int const size ) {

    device::Var<T> sum;
    unsigned int block = ( size < 1024 ) ? size : 1024 ;
    _inclusive_scan_kernel <<< 1, block >>> ( data, size, sum.ptr() );

    return sum.get();
}

/**
 * @brief Perform exclusive scan operation on CUDA device, return reduction on host
 * 
 * @tparam T    Datatype
 * @param data  Data buffer on device
 * @param size  Number of data points
 * @return T    Reduction of data buffer
 */
template< typename T >
__host__
T exclusive_scan( T * const __restrict__ data, unsigned int const size ) {

    device::Var<T> sum;

    unsigned int block = ( size < 1024 ) ? size : 1024 ;
    _exclusive_scan_kernel <<< 1, block >>> ( data, size, sum.ptr());

    return sum.get();
}

}

#endif