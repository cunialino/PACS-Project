#include "OpenNNXOR.hpp"
#include <cmath>

OpenNNXor::OpenNNXor(double a_): alpha(a_)  {

  std::vector<size_t> net_struct({2, 4, 4, 1});
  OpenNN::NeuralNetwork* net = new OpenNN::NeuralNetwork(OpenNN::NeuralNetwork::Classification, net_struct);

  OpenNN::DataSet* data_set = new OpenNN::DataSet("data/xor.csv", ',', false);
  data_set->set_columns_uses({"Input", "Input", "Target"});
  data_set->split_instances_random();

  training_strategy = std::make_unique<OpenNN::TrainingStrategy>(net, data_set);
  training_strategy->set_optimization_method(OpenNN::TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
  training_strategy->set_loss_method(OpenNN::TrainingStrategy::CROSS_ENTROPY_ERROR);
  training_strategy->set_display(false);

  auto opt_pointer = training_strategy->get_stochastic_gradient_descent_pointer();
  opt_pointer->set_loss_goal(1.0e-3);
  opt_pointer->set_minimum_parameters_increment_norm(0.0);
  opt_pointer->set_maximum_epochs_number(1);
  opt_pointer->set_display(false);
  opt_pointer->set_initial_learning_rate(1);

  std::vector<OpenNN::Layer *> lays = net->get_trainable_layers_pointers();
  NumP = 0;
  for(unsigned i = 0; i < lays.size(); i++){
    auto l = (OpenNN::PerceptronLayer *) lays[i];
    NumP += l->get_parameters_number();
  }

  //OpenNN::ScalingLayer* scaling_layer_pointer = net->get_scaling_layer_pointer();
  //scaling_layer_pointer->set_scaling_methods("NoScaling");

  

}

void OpenNNXor::epoch(int lev){

    auto opt_pointer = training_strategy->get_stochastic_gradient_descent_pointer();
    opt_pointer->set_initial_learning_rate(min(8.0, std::pow(1.25, lev)*alpha));

    // Perform Training:
    training_strategy->perform_training();
}

std::vector<double> OpenNNXor::get_weights(void) {

    std::vector<double> retval(NumP);
    OpenNN::NeuralNetwork* net = training_strategy->get_neural_network_pointer();
    std::vector<OpenNN::Layer *> lays = net->get_trainable_layers_pointers();
    unsigned buffInd = 0;
    for(auto l: lays){
        OpenNN::Matrix<double> W = ((OpenNN::PerceptronLayer *)l)->get_synaptic_weights();
        std::vector<double> b = ((OpenNN::PerceptronLayer *)l)->get_biases();
        std::vector<double> pp = ((OpenNN::PerceptronLayer *)l)->get_parameters();
        for(auto p: pp){
            retval[buffInd++] = p;
        }
    }
    return retval;
}
void OpenNNXor::set_weights(std::vector<double> &ws){

    OpenNN::NeuralNetwork* net = training_strategy->get_neural_network_pointer();
    unsigned buffInd = 0;
    std::vector<OpenNN::Layer *> lays = net->get_trainable_layers_pointers();
    for(unsigned i = 0; i < lays.size(); i++){
        auto l = (OpenNN::PerceptronLayer *) lays[i];
        auto s = std::next(ws.cbegin(), buffInd);
        auto e = std::next(s, l->get_parameters_number());
        std::vector<double> params(s, e);
        l->set_parameters(params);
        buffInd += l->get_parameters_number();
    }
    return;
}
void OpenNNXor::eval(void){

    OpenNN::NeuralNetwork* net = training_strategy->get_neural_network_pointer();
    OpenNN::TestingAnalysis testing_analysis(net, data_set);

    const OpenNN::Matrix<size_t> confusion = testing_analysis.calculate_confusion();

    cout << "Confusion: " << endl;
    cout << confusion << endl;
    return;
}

extern "C"{
    Model* build(double alpha){
        return new OpenNNXor(alpha);
    }
}
