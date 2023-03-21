#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>

#define N 256
__global__ void matrix_vector_multi_hpu_1_1(float *A, float *B, float *C) {

  for (int j = 0; j < N; j++) {
    A[j] = 0.0F;
    for (int i = 0; i < N; i++) {
      A[j] = A[j] + B[j * N + i] * C[i];
    }
  }
}

__global__ void matrix_vector_multi_hpu_1_2(float *A, float *B, float *C) {
  int N_start = threadIdx.x * 128;
  for (int j = N_start; j < N_start + 128; j++) {
    A[j] = 0.0F;
    for (int i = 0; i < N; i++) {
      A[j] = A[j] + B[j * N + i] * C[i];
    }
  }
}

__global__ void matrix_vector_multi_hpu_1_256(float *A, float *B, float *C) {
  A[threadIdx.x] = 0.0F;
  for (int i = 0; i < N; i++) {
    A[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x * N + i] * C[i];
  }
}

__global__ void matrix_vector_multi_hpu_1_256_sh(float *A, float *B, float *C) {
  __shared__ float tmp_c[N];

  tmp_c[threadIdx.x] = C[threadIdx.x];
  __syncthreads();

  A[threadIdx.x] = 0.0F;
  for (int i = 0; i < N; i++) {
    A[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x * N + i] * tmp_c[i];
  }
}

__global__ void matrix_vector_multi_hpu_2_128(float *A, float *B, float *C) {
  int j = blockIdx.x * 128 + threadIdx.x;
  A[j] = 0.0F;
  for (int i = 0; i < N; i++) {
    A[j] = A[j] + B[j * N + i] * C[i];
  }
}

__constant__ float C_c[N];
__global__ void matrix_vector_multi_hpu_1_256_const(float *A, float *B) {
  __shared__ float tmp_c[N];

  A[threadIdx.x] = 0.0F;
  for (int i = 0; i < N; i++) {
    A[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x * N + i] * C_c[i];
  }
}

void matrix_vector_multi_cpu(float *A, float *B, float *C) {

  for (int j = 0; j < N; j++) {
    A[j] = 0.0F;
    for (int i = 0; i < N; i++) {
      A[j] = A[j] + B[j * N + i] * C[i];
    }
  }
}

void check(float *A) {
  for (int j = 0; j < N; j++) {
    if (j != A[j]) {
      printf("A[%d]=%f \n", j, A[j]);
    }
  }
}

int main() {
  float A[N], B[N * N], C[N];
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      B[j * N + i] = ((float)j) / 256.0;
    }
  }

  for (int j = 0; j < N; j++) {
    C[j] = 1.0F;
  }

  matrix_vector_multi_cpu(A, B, C);

  check(A);
  memset(A, 0, N * sizeof(float));
  float *A_d, *B_d, *C_d = NULL;
  cudaMalloc((void **)&A_d, N * sizeof(float));
  cudaMalloc((void **)&B_d, N * N * sizeof(float));
  cudaMalloc((void **)&C_d, N * sizeof(float));

  dim3 blocks(1, 1, 1);
  dim3 thread(256, 1, 1);

  cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(C_c, C, N * sizeof(float), cudaMemcpyHostToDevice);

  // matrix_vector_multi_hpu_1_256_sh<<<blocks, thread>>>(A_d, B_d, C_d);
  matrix_vector_multi_hpu_1_256_sh<<<blocks, thread>>>(A_d, B_d, C_d);

  cudaMemcpy(A, A_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  check(A);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}
