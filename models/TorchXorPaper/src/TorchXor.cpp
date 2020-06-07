#include "TorchXor.hpp"
#include <iostream>
#include <cmath>

TorchXor::TorchXor(int64_t bs, double lr, std::string data_file_): batch_size(bs),data_file(data_file_), alpha(lr){

    //net = torch::nn::Sequential(torch::nn::Linear(2, 4), torch::nn::Sigmoid(), torch::nn::Linear(4, 4), torch::nn::Sigmoid(), torch::nn::Linear(4, 1), torch::nn::Sigmoid());
    net->to(device);
    net->to(torch::kDouble);
    NumP = torch::nn::utils::parameters_to_vector(net->parameters()).sizes()[0];
    auto dataset = CustomDataset(data_file).map(torch::data::transforms::Stack<>());

    // Data loader
    auto num_train = std::round(0.8*dataset.size().value());
    num_train = dataset.size().value();
    auto num_test = dataset.size().value() - num_train;
    auto samp = torch::data::samplers::RandomSampler(num_train, torch::kDouble);
    std::cout << "Training - Testing: " << num_train << " - " << num_test << std::endl;
    train_loader = torch::data::make_data_loader(dataset, samp, torch::data::DataLoaderOptions().batch_size(batch_size));

}


void TorchXor::epoch(int lev) {
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(std::pow(1.25, lev)*alpha).momentum(0).weight_decay(0).dampening(0).nesterov(false));

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
    Model* build(double alpha){
        return new TorchXor(4, alpha);
    }
}
