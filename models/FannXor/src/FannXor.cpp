#include "FannXor.hpp"

FannXor::FannXor() {

    net.create_standard(4, 2, 5, 5, 1);

    net.set_learning_rate(learning_rate);

    net.set_activation_steepness_hidden(1.0);
    net.set_activation_steepness_output(1.0);
    
    net.set_activation_function_hidden(FANN::SIGMOID);
    net.set_activation_function_output(FANN::SIGMOID);

    if (data.read_train_from_file("my.data"))
    {
        net.init_weights(data);
        data_flag = true;
        NumP = net.get_total_connections();//net.get_num_layers()+get_total_connections()-1;
    }
    else{
        std::cout << "Data not found! " << std::endl;
        NumP = 0;
    }
}

void FannXor::epoch(int lev){

    if(data_flag)
        net.train_epoch(data);
    return;
}

std::vector<double> FannXor::get_weights(){

    //unsigned *b;
    FANN::connection *w = new FANN::connection[net.get_total_connections()];
    std::vector<double> retval(net.get_total_connections());
    int retvalInd = 0;
    //b = new unsigned[net.get_num_layers()];
    //net.get_bias_array(b);
    net.get_connection_array(w);
    for(int i = 0; i < net.get_total_connections(); i++){
        retval[i] = w[i].weight;
    }
    //for(i = 0; i < net.get_num_layers()){
    //    retval[retvalInd++] = b[i];
    //}
    //delete[] b;
    delete[] w;

    return retval;
}

void FannXor::set_weights(std::vector<double> &ws){

    //unsigned *b;
    FANN::connection *w = new FANN::connection[net.get_total_connections()];
    //b = new unsigned[net.get_num_layers()];
    //net.get_bias_array(b);
    net.get_connection_array(w);
    for(int i = 0; i < net.get_total_connections(); i++){
        w[i].weight = ws[i];
    }
    //for(i = 0; i < net.get_num_layers()){
    //    b[i] = retval[retvalInd++];
    //}
    net.set_weight_array(w, net.get_total_connections());
    //net.set_bias
    //delete[] b;
    delete[] w;

    return;

}

void FannXor::eval(){

    std::cout << std::endl << "Testing network." << std::endl;

    unsigned count = 0;
    for (unsigned int i = 0; i < data.length_train_data(); ++i)
    {
        // Run the network on the test data
        fann_type *calc_out = net.run(data.get_input()[i]);

        if((*calc_out < 0.5 and data.get_output()[i][0] == 0) or (*calc_out > 0.5 and data.get_output()[i][0] == 1))
            count ++;
    }

    std::cout << "Correctly predicted: " << count / (double) data.length_train_data() << std::endl; 
    std::cout << "MSE: " << net.get_MSE() << std::endl;
    std::cout << std::endl << "XOR test completed." << std::endl;
    return;

}
extern "C"{
    Model * build(){
        return new FannXor();
    }
}
