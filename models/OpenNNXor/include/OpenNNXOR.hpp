#ifndef _OPENNNXOR_HPP_
#define _OPENNNXOR_HPP_ 

#include <memory>
#include <iostream>

#include "Model.hpp"
#include "opennn.h"
#include "training_strategy.h"

class OpenNNXor final: public Model{
    private:
        OpenNN::DataSet* data_set = new OpenNN::DataSet("/home/elia/build/xbraid/examples/PACS-Project/data/mydata.csv", ',', false);

        std::unique_ptr<OpenNN::TrainingStrategy> training_strategy;
        const double alpha;

    public:
        OpenNNXor(double); 

        void epoch(int lev) override;

        std::vector<double> get_weights(void) override;

        void set_weights(std::vector<double> &ws) override;

        void eval(void) override;

        ~OpenNNXor(){delete data_set;}

};


#endif
