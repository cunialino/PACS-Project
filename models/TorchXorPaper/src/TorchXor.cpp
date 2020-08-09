#include "TorchXor.hpp"
#include <iostream>
#include <cmath>

TorchXor::TorchXor(int64_t bs, double lr, double mlt, std::string data_file_): batch_size(bs),data_file(data_file_), alpha(lr), mult(mlt){

    //net = torch::nn::Sequential(torch::nn::Linear(2, 4), torch::nn::Sigmoid(), torch::nn::Linear(4, 4), torch::nn::Sigmoid(), torch::nn::Linear(4, 1), torch::nn::Sigmoid());
    net->to(device);
    net->to(torch::kDouble);
    NumP = torch::nn::utils::parameters_to_vector(net->parameters()).sizes()[0];
    auto dataset = CustomDataset(data_file).map(torch::data::transforms::Stack<>());

    std::vector<double> ws({0.5432, 0.3470, 0.4246, 0.2559, 0.2156, 0.4059, -0.5182, -0.2540, -0.4841, -0.2368, 0.3950, -0.4776, 0.2794, -0.1462, 0.0132, -0.2746, 0.2603, 0.0593, -0.3565, 0.2638, -0.3218, 0.0036, -0.0807, 0.4021, -0.0598, 0.1032, -0.1535, -0.1453, 0.1090, 0.3688, 0.0168, -0.0392});
    this->set_weights(ws);
    // Data loader
    auto num_train = dataset.size().value();
    auto samp = torch::data::samplers::RandomSampler(num_train, torch::kDouble);
    train_loader = torch::data::make_data_loader(dataset, samp, torch::data::DataLoaderOptions().batch_size(batch_size));

}


void TorchXor::epoch(int lev) {
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(std::pow(mult, lev)*alpha).momentum(0).weight_decay(0).dampening(0).nesterov(false));

    for (auto& batch : *train_loader) {
        optimizer.zero_grad();

        auto data = batch.data.to(device).to(torch::kDouble);
        auto target = batch.target.to(device);

        // Forward pass
        auto output = net->forward(data);

        // Calculate loss
        auto loss = torch::nn::functional::binary_cross_entropy(output, target);

        // Backward pass and optimize
        loss.backward();
        optimizer.step();
    }
    return;
}

void TorchXor::eval(void){

    auto test_dataset = CustomDataset(data_file).map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);

    net->eval();

    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device).to(torch::kDouble);
        auto target = batch.target.to(device);

        auto output = net->forward(data);

        auto loss = torch::nn::functional::binary_cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.round();
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / (double)num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

extern "C"{
    Model* build(double alpha, double max_alpha, double mult, int ntime, int max_levels){
        return new TorchXor(4, alpha, mult);
    }
}
