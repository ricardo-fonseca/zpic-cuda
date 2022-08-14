#ifndef __TIMER__
#define __TIMER__

#include <iostream>

namespace timer {
    enum units { s, ms, us, ns };
}

class Timer {
    private:

    cudaEvent_t startev, stopev;
    int status;

    public:

    __host__
    /**
     * @brief Construct a new Timer object
     * 
     */
    Timer(){
        cudaEventCreate( &startev );
        cudaEventCreate( &stopev );
        status = -1;
    }

    __host__
    /**
     * @brief Destroy the Timer object
     * 
     */
    ~Timer(){
        cudaEventDestroy( startev );
        cudaEventDestroy( stopev );
        status = -2;
    }

    __host__
    /**
     * @brief Starts the timer
     * 
     */
    void start(){
        status = 0;
        cudaEventRecord( startev );
    }

    __host__
    /**
     * @brief Stops the timer
     * 
     */
    void stop(){
        cudaEventRecord( stopev );
        if ( ! status ) {
            status = 1;
        } else {
            std::cerr << "(*error*) Timer was not started\n";
        }
    }

    __host__ 
    /**
     * @brief Returns elapsed time in milliseconds
     * 
     * @return float 
     */
    float elapsed(){
        if ( status < 1 ) {
            std::cerr << "(*error*) Timer was not complete\n";
            return -1;
        } else {
            if ( status == 1 ) {
                cudaEventSynchronize( stopev );
                status = 2;
            }
            float delta;
            cudaEventElapsedTime(&delta, startev, stopev );
            return delta;
        }
    }

    __host__
    float elapsed( timer::units units ) {
        float ms = elapsed();

        float t;
        switch( units ) {
        case timer::s:  t = 1.e-3 * ms; break;
        case timer::ms: t =         ms; break;
        case timer::us: t = 1.e+3 * ms; break;
        case timer::ns: t = 1.e+6 * ms; break;
        }
        return t;
    }

    __host__
    /**
     * @brief Printout timer 
     * 
     */
    void report( std::string msg = "" ) {
        if ( status < 1 ) {
            std::cout << msg << " timer is not complete." << std::endl;
        } else {
            auto time = elapsed();
            std::cout << msg << " elapsed time was " << time << " ms." << std::endl;
        }
    }
};

#endif