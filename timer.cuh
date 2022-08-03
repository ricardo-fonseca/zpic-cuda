#ifndef __TIMER__
#define __TIMER__

#include <iostream>

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
};

#endif