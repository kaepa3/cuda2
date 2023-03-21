#include <stdio.h>

#define N 256

void matrix_vector_multi_cpu(float *A, float *B, float *C) {

  for (int j = 0; j < N; j++) {
    A[j] = 0.0F;
    for (int i = 0; i < N; i++) {
      A[j] = A[j] + B[j * N + i] * C[i];
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

  for (int j = 0; j < N; j++) {
    printf("A[%d]=%f \n", j, A[j]);
  }
  return 0;
}
