#ifndef HCML_H
#define HCML_H

#define BLOCKS 1000
#define THREADS 100

#define FPI 3.14159264F  // 円周率
#define RPI 0.318309886F // 円周率の逆数
#define CHANCE 0.1F
#define RECHANCE 10.0F
#define WEIGHT 0.00001F
#define STEPS 1000
#define COS90D 0.00001
#define NUM_LAYER 2
#define STRLEN 256

#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

typedef struct {
  float z_min;
  float z_max;
  float mua;
  float mus;
  float mutr;
  float g;
  float n;
} LayerStruct;

typedef struct {
  unsigned int num_photons;
  float Wth;
  float SurfaceSpecular;
  float dz;
  float dr;
  float da;
  unsigned int nz;
  unsigned int nr;
  unsigned int na;
} InputStruct;

typedef struct {
  float x;
  float y;
  float z;
  float dx;
  float dy;
  float dz;
  float weight;
  int layer;
  float sleft;
} PhotonStruct;

typedef struct {
  float *Rd_ra;
  float *A_rz;
  float *Tr_ra;
  unsigned int *thread_active;
  unsigned int *num_terminated_photons;
} DeviceOutStruct;

__constant__ InputStruct input;
__constant__ LayerStruct layerspecs[NUM_LAYER + 2];
__global__ void LaunchPhoton(PhotonStruct *p, DeviceOutStruct data,
                             curandStateMRG32k3a *state, unsigned int seed);
__global__ void MCML_EXE(PhotonStruct *devphoton, DeviceOutStruct data,
                         curandStateMRG32k3a *state);
__global__ int RFresnel(PhotonStruct *p, int nextlayer,
                        curandStateMRG32k3a *state);

float SpecularReflection(float n0, float n1, float n2, float mua, float mus);

void InputInfo();
void InputLayer();
void MemAllocation();
void MemCopyHtoD();
void MemCopyDtoH();
void ResultOutput();
void MemFree();

#endif
