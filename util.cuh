#ifndef __UTIL__
#define __UTIL__

#include <iostream>

#define CHECK_ERR( err_, msg_ ) { \
    if ( err_ != cudaSuccess ) { \
        std::cerr << "(*error*) " << msg_ << std::endl; \
        std::cerr << "(*error*) code: " << err_ << ", reason: " << cudaGetErrorString(err_) << std::endl; \
        exit(1); \
    } \
}

/**
 * @brief Calls cudaDeviceSynchronize(), and exits code in case of error
 * 
 * This can be used to check if there were any errors on earlier kernel calls. In case of an error
 * the routine will printout the error message and the function / file / line where the function
 * was called.
 */
#define deviceSynchronize() { \
    cudaError_t err = cudaDeviceSynchronize(); \
    if ( err != cudaSuccess ) { \
        std::cerr << "(*error*) Unable to synchronize on " << __func__ << "()"; \
        std::cerr << " - " << __FILE__ << "(" << __LINE__ << ")" << std::endl; \
        std::cerr << "(*error*) code: " << err << ", reason: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}


#endif