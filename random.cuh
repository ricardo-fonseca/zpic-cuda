#ifndef __RANDOM__
#define __RANDOM__

#include "math.h"

namespace {

__device__
/**
 * @brief Returns a 32 bit pseudo random number using Marsaglia MWC algorithm
 * 
 * Follows George Marsaglia's post to Sci.Stat.Math on 1/12/99:
 * 
 * The  MWC generator concatenates two 16-bit multiply-
 * with-carry generators, x(n)=36969x(n-1)+carry,
 * y(n)=18000y(n-1)+carry  mod 2^16, has period about
 * 2^60 and seems to pass all tests of randomness. A
 * favorite stand-alone generator---faster than KISS,
 * which contains it. [sic]
 * 
 * See for example Numerical Recipes, 3rd edition, Section 7.1.7,
 * "When You Have Only 32-Bit Arithmetic"
 * 
 * @param state     Previous state of the of the PRNG, will be modified by
 *                  this call.
 * @return Random value in the range [0,2^32 - 1]
 */
unsigned int _rand_uint32( uint2 & state ) {
    state.x = 36969 * (state.x & 65535) + (state.x >> 16);
    state.y = 18000 * (state.y & 65535) + (state.y >> 16);
    return (state.x << 16) + state.y;  /* 32-bit result */
}


__device__
/**
 * @brief Returns a variate of the normal distribution (mean 0, stdev 1)
 * 
 * Uses the Box-Muller method for generating random deviates with a normal
 * (Gaussian) distribution:
 * 
 *  Box, G. E. P.; Muller, Mervin E. (1958). "A Note on the Generation of 
 *  Random Normal Deviates". The Annals of Mathematical Statistics. 29 (2):
 *  610–611. doi:10.1214/aoms/1177706645.
 * 
 * @param state     Previous state of the of the PRNG, will be modified by this
 *                  call
 * @param norm      Previous state of the of the normal random number generator,
 *                  will be modified by this call. The first time the routine
 *                  is called this value should be set to NAN.
 * 
 * @return Double precision random number following a normal distribution 
 */
double _rand_norm( uint2 & state, double & norm ) {
    double res;
    if ( std::isnan( norm ) ) {
        double v1, v2, rsq, fac;
        do {
            v1 = ( _rand_uint32( state ) + 0.5 ) / 2147483649.0 - 1.0;
            v2 = ( _rand_uint32( state ) + 0.5 ) / 2147483649.0 - 1.0;

            // check if they are inside the unit circle, and not (0,0)
            // otherwise try again
            rsq = v1*v1 + v2*v2;

        } while ( rsq == 0.0 || rsq >= 1.0 );

        // Use Box-Muller method to generate random deviates with
        // normal (gaussian) distribution
        fac = sqrt(-2.0 * log(rsq)/rsq);

        // store 1 value for future use
        norm = v1*fac;
        res  = v2*fac;
    } else {
        res  = norm;
        norm = NAN;
    }
    return res;
}

}

#endif