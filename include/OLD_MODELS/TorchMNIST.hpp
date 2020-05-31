#ifndef _TORCHMNIST_HPP_
#define _TORCHMNIST_HPP_

#include <torch/torch.h>
#include <string>
#include <vector>
#include "Model.hpp"


//To define this i used a demangler....
using SDL =torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>;

class TorchMNIST final: public Model{

  private:
    const int64_t batch_size;
    struct view : torch::nn::Module {
        view() { }
        torch::Tensor forward(torch::Tensor input) {
        return input.view({-1, 7*7*32});
    }
    }; 

    const std::string data_file;
    const bool cuda_available = torch::cuda::is_available();
    const std::vector<double> learning_rates;
    torch::Device device = (cuda_available ? torch::kCUDA : torch::kCPU) ;
    std::unique_ptr<SDL> train_loader;
    torch::nn::Sequential net;
  public:
    TorchMNIST(int64_t bs, std::vector<double> lr, std::string data_file_=std::string("/home/elia/build/pytorch-cpp/data/mnist/"));

    void epoch(int lev);

    std::vector<double> get_weights(void) const{
        torch::Tensor vals = torch::nn::utils::parameters_to_vector(net->parameters()).clone();
        std::vector<double> w(vals.sizes()[0]);
        for(int i = 0; i < w.size(); i++) {
          w[i] = vals[i].item<double>();

        }
        return w;
    }
    void set_weights(std::vector<double> &ws){
        torch::Tensor tmp = torch::from_blob(ws.data(), {static_cast<int64_t>(ws.size())}, torch::kDouble).clone();
        torch::nn::utils::vector_to_parameters(tmp.to(device), net->parameters());
        return;
    }

    void eval(void);
};



#endif
