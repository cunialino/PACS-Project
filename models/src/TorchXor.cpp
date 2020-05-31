#include "TorchXor.hpp"
#include <iostream>
#include <cmath>

TorchXor::TorchXor(int64_t bs, double lr, std::string data_file_): batch_size(bs),data_file(data_file_), alpha(lr){

    net = torch::nn::Sequential(torch::nn::Linear(2, 4), torch::nn::ReLU(), torch::nn::Linear(4, 6), torch::nn::ReLU(), torch::nn::Linear(6, 1), torch::nn::Sigmoid());
    net->to(device);
    net->to(torch::kDouble);
    NumP = torch::nn::utils::parameters_to_vector(net->parameters()).sizes()[0];
    auto train_dataset = CustomDataset(data_file).map(torch::data::transforms::Stack<>());
    // Data loader
    train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);

}


void TorchXor::epoch(int lev) {
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(std::pow(1.25, lev)*alpha).momentum(std::pow(1.25, lev)*0.1));
    for (auto& batch : *train_loader) {
        // Transfer images and target labels to device
        auto data = batch.data.to(device).to(torch::kDouble);
        auto target = batch.target.to(device);

        // Forward pass
        auto output = net->forward(data);

        // Calculate loss
        auto loss = torch::nn::functional::binary_cross_entropy(output, target);

        // Calculate prediction
        auto prediction = output.round();

        // Backward pass and optimize
        optimizer.zero_grad();
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
    Model* build(double alpha){
        return new TorchXor(1, alpha);
    }
}