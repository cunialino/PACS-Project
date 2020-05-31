#ifndef _MYBRAIDAPP_HPP_
#define _MTBRAIDAPP_HPP_

#include <vector>
#include <memory>
#include "braid.hpp"
#include "Model.hpp"

class BraidVector
{
public:
   std::vector<double> value;

   BraidVector(std::vector<double> value_) : value(value_) { }

   virtual ~BraidVector() {};

};


class MyBraidApp : public BraidApp
{
protected:
   // BraidApp defines tstart, tstop, ntime and comm_t
   bool accesed = false;

public:
  // Constructor
  MyBraidApp(MPI_Comm comm_t_, int rank_, std::unique_ptr<Model> &net, double tstart_ = 0, double tstop_ = 100, int ntime_ = 100);
   
  // We will need the MPI Rank
  int rank;
  std::vector<double> solWeights;

  // Net and number of connections (needed for the buffers)
  std::unique_ptr<Model> net;

  // Deconstructor
  virtual ~MyBraidApp() { net.~unique_ptr(); };

  // Define all the Braid Wrapper routines
  // Note: braid_Vector == BraidVector*
  void set_weights(std::vector<double>);

  void update_vector(BraidVector *);

  virtual int Step(braid_Vector    u_, braid_Vector    ustop_, braid_Vector    fstop_, BraidStepStatus &pstatus);

  virtual int Clone(braid_Vector  u_, braid_Vector *v_ptr);

  virtual int Init(double        t, braid_Vector *u_ptr);

  virtual int Free(braid_Vector u_);

  virtual int Sum(double       alpha, braid_Vector x_, double       beta, braid_Vector y_);

  virtual int SpatialNorm(braid_Vector  u_, double       *norm_ptr);

  virtual int BufSize(int *size_ptr, BraidBufferStatus  &status);

  virtual int BufPack(braid_Vector  u_, void         *buffer, BraidBufferStatus  &status);

  virtual int BufUnpack(void         *buffer, braid_Vector *u_ptr, BraidBufferStatus  &status);

  virtual int Access(braid_Vector       u_, BraidAccessStatus &astatus);

  // Not needed in this example
  virtual int Residual(braid_Vector     u_, braid_Vector     r_, BraidStepStatus &pstatus) { return 0; }

  // Not needed in this example
  virtual int Coarsen(braid_Vector   fu_, braid_Vector  *cu_ptr, BraidCoarsenRefStatus &status) { return 0; }

  // Not needed in this example
  virtual int Refine(braid_Vector   cu_, braid_Vector  *fu_ptr, BraidCoarsenRefStatus &status)  { return 0; }

};

#endif
