#pragma once

#include <torch/torch.h>
#include <vector>

#include "csv.h"

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
      std::vector<std::vector<double>> datass;
    public:
        explicit CustomDataset( const std::string& file_names_csv) {

          io::CSVReader<4> in(file_names_csv);
          double x, y, z, lab;
          while(in.read_row(x, y, z, lab)){
            datass.push_back({x, y, z, lab});
          }

        };

        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {

          std::vector<double> coords({datass[index][0], datass[index][1], datass[index][2]});


          torch::Tensor img_tensor = torch::from_blob(coords.data(), {3}, torch::kDouble).clone();

          torch::Tensor label_tensor = torch::full({1}, datass[index][3], torch::kDouble);

          return {img_tensor, label_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {

            return datass.size();
        };
};
