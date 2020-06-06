#include "TorchMNIST.hpp"

TorchMNIST::TorchMNIST(int64_t bs, std::vector<double> lr, std::string data_file_): batch_size(bs),
                                                                                        data_file(data_file_), learning_rates(lr)
    {
        net = torch::nn::Sequential( torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)), torch::nn::BatchNorm2d(16), torch::nn::ReLU(),
                torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)), torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
                torch::nn::BatchNorm2d(32), torch::nn::ReLU(), torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)), view(), torch::nn::Linear(7*7*32, 10));
        net->to(device);
        net->to(torch::kDouble);
        NumP = torch::nn::utils::parameters_to_vector(net->parameters()).sizes()[0];
        auto train_dataset = torch::data::datasets::MNIST(data_file)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

        // Data loader
        train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);

    }


    void TorchMNIST::epoch(int lev) {
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(learning_rates[lev]));
        for (auto& batch : *train_loader) {
            // Transfer images and target labels to device
            auto data = batch.data.to(device).to(torch::kDouble);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = net->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        return;
    }

    void TorchMNIST::eval(void){

          auto test_dataset = torch::data::datasets::MNIST(data_file, torch::data::datasets::MNIST::Mode::kTest)
              .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
              .map(torch::data::transforms::Stack<>());

          // Number of samples in the testset
          auto num_test_samples = test_dataset.size().value();

          // Data loader
          auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
              std::move(test_dataset), batch_size);

          net->eval();

          torch::NoGradGuard no_grad;

          double running_loss = 0.0;
          size_t num_correct = 0;

          for (const auto& batch : *test_loader) {
              auto data = batch.data.to(device).to(torch::kDouble);
              auto target = batch.target.to(device);

              auto output = net->forward(data);

              auto loss = torch::nn::functional::cross_entropy(output, target);
              running_loss += loss.item<double>() * data.size(0);

              auto prediction = output.argmax(1);
              num_correct += prediction.eq(target).sum().item<int64_t>();
          }

          std::cout << "Testing finished!\n";

          auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
          auto test_sample_mean_loss = running_loss / num_test_samples;

          std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
            }
extern "C"{
    Model* build(void){
        return new TorchMNIST(300, {0.1, 0.2, 0.4, 0.8, 1.6});
    }
}
