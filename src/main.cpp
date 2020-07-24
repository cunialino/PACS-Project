#include <iostream>
#include <fstream>
#include <string.h>
#include <memory>
#include <vector>
#include <ctime>
#include <cmath>


//Reading models directory to let you choose what to train
#include <string>
#include <unordered_map>

//Main library, used to perform MGRIT
#include <braid.hpp>
#include "NNBraidApp.hpp"

//Abstract class required for models 
#include "Model.hpp"
//Needed to read models at runtime
#include <dlfcn.h>


// --------------------------------------------------------------------------
// Main driver
// --------------------------------------------------------------------------

int main (int argc, char *argv[])
{

  // Initialize MPI
  MPI_Init(&argc, &argv);

  int           ntime, rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Define time domain: ntime intervals
  ntime  = 200;
  
  int           max_levels = 2;
  int           nrelax     = 1;
  double        tol        = 1.0e-09;
  int           cfactor    = 2;
  int           max_iter   = 10;
  int           fmg        = 0;
  int           print_level = 2;
  int           min_coarse = 10;
  int           skip       = 0;
  int           seq        = 0;
  double base_alpha = 0.1, max_alpha = 1, mult = 2;
  std::string model_path("models/libtorchxorpaper.so");


  int           arg_index;
  //Command line arguments: 
  arg_index = 1;
  while (arg_index < argc)
  {
    std::string arg = argv[arg_index];
    if ( arg.compare("-help") == 0 ){
        if ( rank == 0 ){
            std::cout << "-ntime    <value>     : set num time points" << std::endl;
            std::cout << "-ml       <value>     : set max levels" << std::endl;
            std::cout << "-tol      <value>     : set stopping tolerance" << std::endl;
            std::cout << "-cf       <value>     : set coarsening factor" << std::endl;
            std::cout << "-mi       <value>     : set max iterations" << std::endl;
            std::cout << "-fmg                  : use FMG cycling" << std::endl;
            std::cout << "-skip                 : skip down cycle" << std::endl;
            std::cout << "-ba       <value>     : value of alpha on finest level" << std::endl;
            std::cout << "-ma       <value>     : maxiumum value of alpha on every level" << std::endl;
            std::cout << "-mult     <value>     : multiplicative factor of alpha between levels" << std::endl;
            std::cout << "-minc     <value>     : minimum size of the coarse problem" << std::endl;
            std::cout << "-mod      <value>     : full path to the .so file containing your model" << std::endl;
            std::cout << "-pl       <value>     : print level of XBraid" << std::endl;

        }
        MPI_Finalize();
        return 0;
    } 
    else if(arg.compare("-seq") == 0){
        arg_index ++;
        seq = 1;
    }
    else if(arg.compare("-ba") == 0) {
        arg_index++;
        base_alpha = atof(argv[arg_index++]);
    }
    else if(arg.compare("-ma") == 0) {
        arg_index++;
        max_alpha = atof(argv[arg_index++]);
    }
    else if(arg.compare("-mult") == 0) {
        arg_index++;
        mult = atof(argv[arg_index++]);
    }
    else if (arg.compare("-ntime") == 0 ) {
        arg_index++;
        ntime = atoi(argv[arg_index++]);
    }
    else if (arg.compare("-mod") == 0) {
        arg_index++;
        model_path = argv[arg_index++];
    }
    else if (arg.compare("-pl") == 0 ){
       arg_index++;
       print_level = atoi(argv[arg_index++]);
    }
    else if (arg.compare("-skip") == 0 ){
       arg_index++;
       skip = 1;
    }
    else if (arg.compare("-ml") == 0 ) {
        arg_index++;
        max_levels = atoi(argv[arg_index++]);
    }
    else if (arg.compare("-minc") == 0 ) {
        arg_index++;
        min_coarse = atoi(argv[arg_index++]);
    }
    else if (arg.compare("-tol") == 0 ) {
        arg_index++;
        tol = atof(argv[arg_index++]);
    }
    else if (arg.compare("-cf") == 0 ) {
        arg_index++;
        cfactor = atoi(argv[arg_index++]);
    }
    else if (arg.compare("-mi") == 0 ) {
        arg_index++;
        max_iter = atoi(argv[arg_index++]);
    }
    else if (arg.compare("-fmg") == 0 ) {
        arg_index++;
        fmg = 1;
    }
    else {
        arg_index++;
    }
  }

  //Gathering avaiable models 
  std::unordered_map<unsigned, std::string> models_map;

  int actual_ml = std::min(static_cast<double>(max_levels), std::ceil((std::log(ntime)-std::log(min_coarse))/(std::log(cfactor))));
  if(rank == 0)
    std::cout << "There will be " << actual_ml << " levels at the end ;) " << std::endl;

  void* handle = dlopen(model_path.c_str(), RTLD_NOW);

  if(handle == nullptr){
      std::cerr << dlerror() << std::endl;
      MPI_Finalize();
      return 0;
  }
  Model* (*mkr)(double, double, double, int, int); 

  *(void**)(&mkr) = dlsym(handle, "build");

  if(mkr == nullptr){
      std::cerr << dlerror() << std::endl;
      MPI_Finalize();
    return 0;
  }
  std::unique_ptr<Model> model((*mkr)(base_alpha, max_alpha, mult, ntime, max_levels));

  // set up app structure
  NNBraidApp app(MPI_COMM_WORLD, rank, model, 0, ntime, ntime);
  if(seq && rank == 0){

    time_t tstart, tend; 
    tstart = time(0);

    BraidVector *u = new BraidVector(std::vector<double>(app.net->get_NumP()));
    app.Init(0, (braid_Vector*) (&u));
    std::cout << "Starting serial trainig" << std::endl;
    for(int i = 0; i < ntime; i++){
        app.set_weights(u->value);
        app.net->epoch(0);
        app.update_vector(u);
    }
    app.net->set_weights(u->value);
    tend = time(0);
    std::cout << "Serial training completed in " << difftime(tend, tstart) << " seconds(s)" << std::endl;
  }
  else if(seq == 0){
    BraidCore core(MPI_COMM_WORLD, &app);
    core.SetPrintLevel(print_level);
    core.SetMaxLevels(max_levels);
    core.SetMaxIter(max_iter);
    core.SetAbsTol(tol*std::sqrt(ntime));
    core.SetCFactor(-1, cfactor);
    core.SetNRelax(-1, nrelax);
    core.SetSkip(skip);
    if(fmg)
      core.SetFMG();
    core.SetMinCoarse(min_coarse);
    //Run Simulation
    core.Drive();
    app.net->set_weights(app.solWeights);
  }

  if(rank == 0){
    app.net->eval();
  }

  // Clean up
  MPI_Finalize();

  return (0);
}

