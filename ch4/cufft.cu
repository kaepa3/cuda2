#include <corecrt_malloc.h>
#include <cuComplex.h>
#include <cuda_device_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <stdio.h>
#pragma comment(lib, "cufft.lib")

int main() {
#define NX 256
#define NY 256
  cufftHandle plan;
  cufftComplex *host;
  cufftComplex *dev;
  host = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * NY);
  for (int i = 0; i < NX; i++) {
    for (int j = 0; j < NX; j++) {
      if (j > NX - 50 && j < NX / 2 + 50 && i > NY / 2 - 50 &&
          i < NY / 2 + 50) {
        host[j + i * NX] = make_cuComplex(1.0f, 0.0f);
      } else {
        host[j + i * NX] = make_cuComplex(0.0f, 0.0f);
      }
    }
  }

  cudaMalloc((void **)&dev, sizeof(cufftComplex));
  cudaMemcpy(dev, host, sizeof(float) * NX * NY, cudaMemcpyHostToDevice);

  cufftPlan2d(&plan, NX, NY, CUFFT_C2C);
  cufftExecC2C(plan, dev, dev, CUFFT_FORWARD);
  cudaMemcpy(host, dev, sizeof(float) * NX * NY, cudaMemcpyDeviceToHost);
  cufftDestroy(plan);
  cudaFree(dev);
  free(host);
}
