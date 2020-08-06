#ifndef _TORCHXOR_HPP_
#define _TORCHXOR_HPP_

#include <torch/torch.h>
#include <string>
#include <vector>
#include "Model.hpp"
#include "CustomDataSet.h"


namespace {
  struct Dropout: torch::nn::Module{
      Dropout(){}
    torch::Tensor forward(torch::Tensor x){ 
        x = torch::dropout(x, 0.5, is_training());
        return x;
    }

  };

  struct Net : torch::nn::Module {
    Net() {
        seq = torch::nn::Sequential(torch::nn::Linear(torch::nn::LinearOptions(3, 4).bias(false)), torch::nn::Sigmoid(), torch::nn::Linear(torch::nn::LinearOptions(4, 4).bias(false)), torch::nn::Sigmoid(), torch::nn::Linear(torch::nn::LinearOptions(4, 1).bias(false)), torch::nn::Sigmoid());
        register_module("seq", seq);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = seq->forward(x);
        return x;
    }
    torch::nn::Sequential seq; 
  };
}

using SDL = std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >;

class TorchXor final: public Model{

  private:
    const int64_t batch_size;
    const std::string data_file;
    const bool cuda_available = torch::cuda::is_available();
    const double alpha;
    const double mult;
    torch::Device device = (cuda_available ? torch::kCUDA : torch::kCPU) ;
    SDL train_loader;
    //torch::nn::Sequential net;
    std::unique_ptr<Net> net = std::make_unique<Net>();

  public:
    TorchXor(int64_t bs, double alpha, double mult, std::string data_file_=std::string("data/PaperXor.csv"));

    void epoch(int lev) override;

    std::vector<double> get_weights(void) override {
        torch::Tensor vals = torch::nn::utils::parameters_to_vector(net->parameters()).clone();
        std::vector<double> w(vals.sizes()[0]);
        for(int i = 0; i < static_cast<int>(w.size()); i++) {
          w[i] = vals[i].item<double>();
        }
        return w;
    }
    void set_weights(std::vector<double> &ws) override {
        torch::Tensor tmp = torch::from_blob(ws.data(), {static_cast<int64_t>(ws.size())}, torch::kDouble).clone();
        torch::nn::utils::vector_to_parameters(tmp.to(device), net->parameters());
        return;
    }

    void eval(void);
};



#endif
