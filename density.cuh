#ifndef __DENSITY__
#define __DENSITY__

#include <iostream>

#include "particles.cuh"
#include "util.cuh"

namespace Density {


    class Profile {
        protected:
        float n0;
        public:
        __host__
        Profile(float const n0) : n0(abs(n0)) {};

        float get_n0() const { return n0; }

        virtual Profile * clone() const = 0;
        
        virtual void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const = 0;
    
        virtual void np_inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const = 0;
    };

    /**
     * @brief Zero density (no particles), used to disable injection
     * 
     */
    class None : public Profile {

        public:

        None( float const n0) : Profile( n0 ) { };

        None * clone() const override {
            return new None( n0 );
        };
        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override {
            // no injection
        };
        void np_inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override {
            // no injection
            device::zero( np, part -> ntiles.x * part -> ntiles.y );
        };
    };


    /**
     * @brief Uniform plasma density
     * 
     */
    class Uniform : public Profile {

        public:

        Uniform( float const n0 ) : Profile(n0) { };

        Uniform * clone() const override {
            return new Uniform(n0);
        };
        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief Step (Heavyside) plasma density
     * 
     * Uniform plasma density after a given position. Can be set in either x or y coordinates
     */
    class Step : public Profile {
        private:
        float pos;
        coord::cart dir;
        public:
        Step( coord::cart dir, float const n0, float const pos ) : Profile(n0), pos(pos), dir(dir) {};

        Step * clone() const override {
            return new Step( dir, n0, pos );
        };

        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief Slab plasma density
     * 
     * Uniform plasma density inside given 1D range. Can be set in either x or y coordinates
     * 
     */
    class Slab : public Profile {
        private:
        float begin, end;
        coord::cart dir;
        public:
        Slab( coord::cart dir, float const n0, float begin, float end ) : Profile(n0), begin(begin), end(end), dir(dir) {};
        
        Slab * clone() const override {
            return new Slab( dir, n0, begin, end );
        };

        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

    /**
     * @brief Sphere plasma density
     * 
     * Uniform plasma density centered about a given position
     * 
     */
    class Sphere : public Profile {
        private:
        float2 center;
        float radius;
        public:
        
        Sphere( float const n0, float2 center, float radius ) : Profile(n0), center(center), radius(radius) {};

        Sphere * clone() const override { 
            return new Sphere(n0, center, radius);
        };
        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
        void np_inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range, int * np ) const override;
    };

}

#endif