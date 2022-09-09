#ifndef __UDIST__
#define __UDIST__

#include <iostream>
#include "particles.cuh"
#include "util.cuh"

namespace UDistribution {

    class Type {
        public:
        virtual Type * clone() const = 0;
        virtual void set( Particles & part, unsigned int seed ) const = 0;
    };
    
    class None : public Type {
        public:
        None * clone() const override { return new None(); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class Cold : public Type {
        public:
        float3 ufl;
        Cold( float3 const ufl ) : ufl(ufl) {};
        Cold * clone() const override { return new Cold(ufl); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class Thermal : public Type {
        public:
        float3 uth;
        float3 ufl;
        Thermal( float3 const uth, float3 const ufl ) : uth(uth), ufl(ufl) {};

        Thermal * clone() const override { return new Thermal(uth, ufl); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class ThermalCorr : public Type {
        protected:
        int npmin;
        
        public:
        float3 uth;
        float3 ufl;
        ThermalCorr( float3 const uth, float3 const ufl, int const npmin ) : uth(uth), ufl(ufl), npmin(npmin) {
            if ( npmin <= 1 ) {
                std::cout << "(*error*) invalid npmin parameter, must be > 1\n";
                exit(1);
            }
        };

        ThermalCorr * clone() const override { return new ThermalCorr(uth, ufl,npmin); };
        void set( Particles & part, unsigned int seed ) const override ;
    };
}

#endif