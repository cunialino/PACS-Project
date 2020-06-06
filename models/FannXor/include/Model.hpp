#ifndef _MODEL_HPP_
#define _MODEL_HPP_

#include <string>
#include <vector>
#include <iostream>


class Model{

  protected:

    unsigned NumP = 0;

  public:
    Model(void) {}
    Model(unsigned NP): NumP(NP) {}

    virtual void epoch(int lev)=0;

    virtual void set_weights(std::vector<double> &ws)=0;

    virtual std::vector<double> get_weights(void) =0;

    unsigned get_NumP(void) const{
        return NumP;
    }

    virtual void eval(void)=0;


    virtual ~Model() = default;

};

#endif
