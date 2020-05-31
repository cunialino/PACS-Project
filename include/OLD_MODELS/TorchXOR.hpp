#ifndef _TORCHXOR_HPP_
#define _TORCHXOR_HPP_

#include "Model.hpp"


#include <torch/torch.h>
#include <string>
#include <vector>
#include "Model.hpp"


//To define this i used a demangler....
using SDL =

class TorchXOR final: public Model{

  private:
    const int64_t batch_size;
    const std::string data_file;
    const bool cuda_available = torch::cuda::is_available();
    const std::vector<double> learning_rates;
    torch::Device device = (cuda_available ? torch::kCUDA : torch::kCPU) ;
    std::unique_ptr<SDL> train_loader;

  public:
    TorchXOR(int64_t bs, std::vector<double> lr, std::string data_file_=std::string("/home/elia/build/xbraid/examples/Torch/data/mydata.csv"));

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




#endif

