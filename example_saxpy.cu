#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
  int N = 1<<20; //1 million
  float alpha = 1.0; //scalar value for saxpy
  cublasHandle_t handle;
  float *x, *y, *d_x, *d_y;

  // Allocate host memory
  x = (float*)malloc(N * sizeof(x[0]));
  y = (float*)malloc(N * sizeof(y[0]));

  // initialize x and y vectors on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Allocate device memory
  cudaMalloc((void**)&d_x, N*sizeof(x[0]));
  cudaMalloc((void**)&d_y, N*sizeof(y[0]));

  // Start cuBLAS code
  cublasCreate(&handle);

  // Copy host vectors to device
  cublasSetVector(N, sizeof(x[0]), x, 1, d_x, 1);
  cublasSetVector(N, sizeof(y[0]), y, 1, d_y, 1);

  // y = alpha*x + y
  cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);

  // Copy device vector back to host
  cublasGetVector(N, sizeof(d_y[0]), d_y, 1, y, 1);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  // End of cuBLAS code
  cublasDestroy(handle);
}