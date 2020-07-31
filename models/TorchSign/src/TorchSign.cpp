#include "TorchSign.hpp"
#include <iostream>
#include <cmath>

TorchSign::TorchSign(double lr, double max_lr, double alpha_mult_, int ntime_, int max_levels_, std::string data_file_): data_file(data_file_), alpha(lr), max_alpha(max_lr), alpha_mult(alpha_mult_), ntime(ntime_), max_levels(max_levels_){

    net->to(device);
    net->to(torch::kDouble);
    NumP = torch::nn::utils::parameters_to_vector(net->parameters()).sizes()[0];
    auto dataset = CustomDataset(data_file).map(torch::data::transforms::Stack<>());
    //unsigned batch_size = std::floor(dataset.size().value()/max_levels);
    unsigned batch_size = dataset.size().value();
    // Data loader
    train_loader = torch::data::make_data_loader(dataset, batch_size);
    std::cout << "alpha: " << alpha << " | max alpha: " << max_alpha << " | mult: " << alpha_mult << std::endl;

}


void TorchSign::epoch(int lev) {
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(std::min(max_alpha, std::pow(alpha_mult, lev)*alpha))); //.momentum(0).weight_decay(0).dampening(0).nesterov(false));

    auto batch_it = train_loader->begin();

    auto batch = *batch_it;
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

    return;
}

void TorchSign::eval(void){

    auto test_dataset = CustomDataset(data_file).map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), 1);

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
        return new TorchSign(alpha, max_alpha, mult, ntime, max_levels);
    }
}
