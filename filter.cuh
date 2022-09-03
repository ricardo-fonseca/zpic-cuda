#ifndef __FILTER__
#define __FILTER__

#include "vector_field.cuh"

namespace Filter {

class Digital {
    public:
    virtual Digital * clone() const = 0;
    virtual void apply( VectorField & fld ) = 0;
};

class None : public Digital {
    public:
    None * clone() const override { return new None(); };
    void apply( VectorField & fld ) { /* do nothing */ };
};

class Binomial : public Digital {
    protected:

    unsigned int order;
    
    public:

    Binomial( unsigned int order_ = 0 ) {
        order = ( order_ > 0 )? order_ : 1;
    };

    Binomial * clone() const override{ return new Binomial(order); };

    void apply( VectorField & fld ) {
        for( int i = 0; i < order; i++ )
            fld.kernel3_x( 0.25f, 0.5f, 0.25f);
    };
};

class Compensated : public Binomial {
    
    public:

    Compensated( unsigned int order_ = 0 ) : Binomial( order_ ) {};

    Compensated * clone() const override { return new Compensated(order); };

    void apply( VectorField & fld ) {
        for( int i = 0; i < order; i++ )
            fld.kernel3_x( 0.25f, 0.5f, 0.25f);
        
        // Calculate compensator values
        float a = -1.0f;
        float b = (4.0 + 2.0*order) / order;
        float norm = 2*a+b;

        fld.kernel3_x( a/norm, b/norm, a/norm);
    };
};


}


#endif