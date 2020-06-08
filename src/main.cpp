#include <iostream>
#include <fstream>
#include <string.h>
#include <memory>
#include <vector>
#include <ctime>


//Reading models directory to let you choose what to train
#include <string>
#include <filesystem>
#include <unordered_map>

//Main library, used to perform MGRIT
#include <braid.hpp>
#include "MyBraidApp.hpp"

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
  int           nrelax0    = -1;
  int           nrelaxc    = 1;
  double        tol        = 1.0e-09;
  int           cfactor    = 2;
  int           max_iter   = 10;
  int           fmg        = 0;
  int           res        = 0;
  int           sync       = 0;
  int           print_level = 2;
  int           min_coarse = 10;
  int           skip       = 0;
  int           seq        = 0;
  double base_alpha = 0.1, max_alpha = 1, mult = 2;


  int           arg_index;
  //Command line arguments: 
  arg_index = 1;
  while (arg_index < argc)
  {
    if ( strcmp(argv[arg_index], "-help") == 0 ){
        if ( rank == 0 ){
           printf("  -ntime <ntime>    : set num time points\n");
           printf("  -ml  <max_levels> : set max levels\n");
           printf("  -nu  <nrelax>     : set num F-C relaxations\n");
           printf("  -nu0 <nrelax>     : set num F-C relaxations on level 0\n");
           printf("  -nuc <nrelax>     : set num F-C relaxations on coarsest grid\n");
           printf("  -tol <tol>        : set stopping tolerance\n");
           printf("  -cf  <cfactor>    : set coarsening factor\n");
           printf("  -mi  <max_iter>   : set max iterations\n");
           printf("  -fmg              : use FMG cycling\n");
           printf("  -res              : use my residual\n");
           printf("  -sync             : enable calls to the sync function\n");
           printf("  -skip             : skip down cycle\n");
        }
        exit(1);
    } 
    else if(strcmp(argv[arg_index], "-seq") == 0){
        arg_index ++;
        seq = 1;
    }
    else if(strcmp(argv[arg_index], "-ba") == 0) {
        arg_index++;
        base_alpha = atof(argv[arg_index++]);
    }
    else if(strcmp(argv[arg_index], "-ma") == 0) {
        arg_index++;
        max_alpha = atof(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-ntime") == 0 ) {
        arg_index++;
        ntime = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-pl") == 0 ){
       arg_index++;
       print_level = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-skip") == 0 ){
       arg_index++;
       skip = 1;
    }
    else if ( strcmp(argv[arg_index], "-ml") == 0 ) {
        arg_index++;
        max_levels = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-minc") == 0 ) {
        arg_index++;
        min_coarse = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-nu") == 0 ) {
        arg_index++;
        nrelax = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-nu0") == 0 ) {
        arg_index++;
        nrelax0 = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-nuc") == 0 ) {
        arg_index++;
        nrelaxc = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-tol") == 0 ) {
        arg_index++;
        tol = atof(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-cf") == 0 ) {
        arg_index++;
        cfactor = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-mi") == 0 ) {
        arg_index++;
        max_iter = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-fmg") == 0 ) {
        arg_index++;
        fmg = 1;
    }
    else if ( strcmp(argv[arg_index], "-res") == 0 ) {
        arg_index++;
        res = 1;
    }
    else if( strcmp(argv[arg_index], "-sync") == 0 ) {
        arg_index++;
        sync = 1;
    }
    else {
        arg_index++;
    }
  }

  //Gathering avaiable models 
  std::unordered_map<unsigned, std::string> models_map;

  std::string suffix(".so");
  std::filesystem::path cwd = std::filesystem::current_path();

  unsigned indM = 0;

  if(rank == 0)
    std::cout << "Choose your models by typing its number: " << std::endl;
  for(auto &p: std::filesystem::directory_iterator("models")){
    std::string file = p.path().string();
    if(file.length() >= suffix.length() && (0 == file.compare (file.length() - suffix.length(), suffix.length(), suffix))){
        if(rank == 0)
            std::cout << indM << ": " << file << std::endl;
        std::string fullpath = cwd.string().append("/").append(file);
        models_map.insert(std::make_pair(indM++, fullpath));
    }
  }

  if(rank == 0)
    std::cin >> indM;

  MPI_Bcast(&indM, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  void* handle = dlopen(models_map[indM].c_str(), RTLD_NOW);

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
  MyBraidApp app(MPI_COMM_WORLD, rank, model, 0, ntime, ntime);
  if(seq && rank == 0){

    time_t tstart, tend; 
    tstart = time(0);

    BraidVector *u = new BraidVector(std::vector<double>(app.net->get_NumP()));
    app.Init(0, (braid_Vector*) (&u));
    std::cout << "Starting serial trainig" << std::endl;
    for(unsigned i = 0; i < ntime; i++){
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

