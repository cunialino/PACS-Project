#include "MyBraidApp.hpp"
#include <cassert>



// Braid App Constructor
MyBraidApp::MyBraidApp(MPI_Comm comm_t_, int rank_, std::unique_ptr<Model> &net_, double tstart_, double tstop_, int ntime_):
    BraidApp(comm_t_, tstart_, tstop_, ntime_), rank(rank_), net(std::move(net_)){ } 

//
void MyBraidApp::set_weights(std::vector<double> value){
  net->set_weights(value);
  return;
}

void MyBraidApp::update_vector(BraidVector * u){
  auto tmp = net->get_weights();
  for(int i = 0; i < net->get_NumP(); i++) {
    u->value[i] = tmp[i];

  }
  return;
}
int MyBraidApp::Step(braid_Vector    u_, braid_Vector    ustop_, braid_Vector    fstop_, BraidStepStatus &pstatus) {
   
  BraidVector *u = (BraidVector*) u_;
   
  int lev;
  double alpha;
  pstatus.GetLevel(&lev);
  //Set weights
  set_weights(u->value);
  //One step
  net->epoch(lev);
  //Update vector
  update_vector(u);
  // no refinement
  pstatus.SetRFactor(1);

 return 0;

}

int MyBraidApp::Init(double t, braid_Vector *u_ptr) {
   
  std::vector<double> vec(net->get_NumP(), 0);
  BraidVector *u = new BraidVector(vec);
  update_vector(u);
  *u_ptr = (braid_Vector) u;

  return 0;

}

int MyBraidApp::Clone(braid_Vector  u_, braid_Vector *v_ptr) {
   BraidVector *u = (BraidVector*) u_;
   BraidVector *v = new BraidVector(u->value); 
   *v_ptr = (braid_Vector) v;

   return 0;
}


int MyBraidApp::Free(braid_Vector u_)
{
   BraidVector *u = (BraidVector*) u_;
   delete u;
   return 0;
}
int MyBraidApp::Sum(double       alpha, braid_Vector x_, double       beta, braid_Vector y_) {
   BraidVector *x = (BraidVector*) x_;
   BraidVector *y = (BraidVector*) y_;
   assert(x->value.size() == y->value.size());
     for(unsigned i = 0; i < x->value.size(); i++){
       y->value[i] = alpha*(x->value[i]) + beta*(y->value[i]);
     }
   return 0;
}

int MyBraidApp::SpatialNorm(braid_Vector  u_, double       *norm_ptr) {
 double dot = 0;
 BraidVector *u = (BraidVector*) u_;
 for(unsigned  i = 0; i < net->get_NumP(); i++){
   dot += u->value[i]*u->value[i];
 }
 *norm_ptr = sqrt(dot)/(ntime);

  return 0;
}

int MyBraidApp::BufSize(int *size_ptr, BraidBufferStatus  &status){

   *size_ptr = (net->get_NumP())*sizeof(double);
   return 0;
}

int MyBraidApp::BufPack(braid_Vector u_, void *buffer, BraidBufferStatus  &status) {
   BraidVector *u = (BraidVector*) u_;
   double *dbuffer = (double *) buffer;
   
   for(unsigned i = 0; i < u->value.size(); i++){
     dbuffer[i] = u->value[i];
   }
   status.SetSize((net->get_NumP())*sizeof(double));

   return 0;
}

int MyBraidApp::BufUnpack(void *buffer, braid_Vector *u_ptr, BraidBufferStatus &status){
   double *dbuffer = (double *) buffer;
   
   std::vector<double> tmp(net->get_NumP());
   
   for(unsigned i = 0; i < net->get_NumP(); i++){
     tmp[i] = dbuffer[i];
   }

   BraidVector *u = new BraidVector(tmp); 
   *u_ptr = (braid_Vector) u;

   return 0;
}

int MyBraidApp::Access(braid_Vector u_, BraidAccessStatus &astatus) {
   BraidVector *u = (BraidVector*) u_;

   // Extract information from astatus
   int done, level, iter, index;
   double t;
   astatus.GetTILD(&t, &iter, &level, &done);
   astatus.GetTIndex(&index);

   if(level == 0 && done){
    int nprocs, solRank;
    MPI_Comm_size(comm_t, &nprocs);
    _braid_GetBlockDistProc(ntime, nprocs, ntime-1, 0, &solRank);

    solWeights.resize(net->get_NumP());

    if(rank != solRank && not accesed){
        accesed = true;
        MPI_Bcast(solWeights.data(), net->get_NumP(), MPI_DOUBLE, solRank, comm_t);
    }
    else if(t == tstop){

        MPI_Bcast(u->value.data(), net->get_NumP(), MPI_DOUBLE, solRank, comm_t);
        solWeights = u->value;
    }
   }
   return 0;
}



