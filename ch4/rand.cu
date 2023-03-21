#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>
#include <stdio.h>

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

int main() {
  int N = 1024;
  curandGenerator_t gen;
  float *p_d, *p_h;

  p_h = (float *)malloc(N * sizeof(float));
  cudaMalloc((void **)&p_d, N * sizeof(float));

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 9999ULL);
  curandGenerateUniform(gen, p_d, N);

  cudaMemcpy(p_h, p_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    printf("%.4f\n", p_h[i]);
  }

  curandDestroyGenerator(gen);
  cudaFree(p_d);
  free(p_h);
}
