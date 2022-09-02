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
    };

    class Uniform : public Profile {

        public:

        Uniform( float const n0 ) : Profile(n0) { };

        Uniform * clone() const override {
            return new Uniform(n0);
        };
        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
    };

    class Step : public Profile {
        private:
        float pos;
        public:
        Step( float const n0, float const pos ) : Profile(n0), pos(pos) {};

        Step * clone() const override {
            return new Step(n0,pos);
        };

        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
    };

    class Slab : public Profile {
        private:
        float begin, end;
        public:
        Slab( float const n0, float begin, float end ) : Profile(n0), begin(begin), end(end) {};
        
        Slab * clone() const override {
            return new Slab(n0, begin, end);
        };

        void inject( Particles * part, uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const override;
    };

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
    };

}

#endif