#ifndef _TORCHXOR_HPP_
#define _TORCHXOR_HPP_

#include <torch/torch.h>
#include <string>
#include <vector>
#include "Model.hpp"
#include "CustomDataSet.h"


//To define this i used a demangler....
using SDL = std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >;

class TorchXor final: public Model{

  private:
    const int64_t batch_size;
    const std::string data_file;
    const bool cuda_available = torch::cuda::is_available();
    const double alpha;
    //torch::Device device = (cuda_available ? torch::kCUDA : torch::kCPU) ;
    SDL train_loader;
    torch::nn::Sequential net;
  public:
    TorchXor(int64_t bs, double alpha, std::string data_file_=std::string("data/xor.csv"));

    void epoch(int lev);

    std::vector<double> get_weights(void) override {
        torch::Tensor vals = torch::nn::utils::parameters_to_vector(net->parameters()).clone();
        std::vector<double> w(vals.sizes()[0]);
        for(int i = 0; i < w.size(); i++) {
          w[i] = vals[i].item<double>();
        }
        return w;
    }
    void set_weights(std::vector<double> &ws){
        torch::Tensor tmp = torch::from_blob(ws.data(), {static_cast<int64_t>(ws.size())}, torch::kDouble).clone();
        torch::nn::utils::vector_to_parameters(tmp, net->parameters());
        return;
    }

    void eval(void);
};



#endif
