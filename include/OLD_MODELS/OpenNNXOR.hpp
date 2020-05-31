#ifndef _OPENNNXOR_HPP_
#define _OPENNNXOR_HPP_ 

#include <memory>
#include <iostream>

#include "Model.hpp"
#include "opennn.h"

class OpenNNXor final: public Model{
    private:
        //std::unique_ptr<OpenNN::NeuralNetwork> net;
        //std::unique_ptr<OpenNN::DataSet> data_set = std::make_unique<OpenNN::DataSet>("/home/elia/build/xbraid/examples/PACS-Project/data/mydata.csv", ',', false);
        OpenNN::DataSet* data_set = new OpenNN::DataSet("/home/elia/build/xbraid/examples/PACS-Project/data/mydata.csv", ',', false);

        std::unique_ptr<OpenNN::TrainingStrategy> training_strategy;
        //std::unique_ptr<OpenNN::StochasticGradientDescent> opt_pointer;

    public:
        OpenNNXor(); 

        void epoch(int lev) override;

        std::vector<double> get_weights(void) const override;

        void set_weights(std::vector<double> &ws) override;

        void eval(void) override;

        ~OpenNNXor(){delete data_set;}

};


#endif
