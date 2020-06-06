#include "fann.h"
#include "fann_cpp.h"
#include "Model.hpp"
#include <vector>
#include <memory>

#include <ios>
#include <iostream>
#include <iomanip>

class FannXor final: public Model{
    private:
        const float learning_rate = 0.7f;
        const unsigned int num_layers = 3;
        const unsigned int num_input = 2;
        const unsigned int num_hidden = 3;
        const unsigned int num_output = 1;
        bool data_flag = false;
        FANN::neural_net net;
        FANN::training_data data;

    public:
        FannXor();

        void epoch(int lev) override;

        void set_weights(std::vector<double> &ws) override;

        std::vector<double> get_weights(void) override;

        void eval(void) override;

};















