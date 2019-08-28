#include <iostream>

// cuBLAS includes
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel function to initialize arrays in GPU memory
__global__
void init(int n, float *x, float *y)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

int main(void)
{
  int N = 1<<20; // 1 Million
  float *x, *y;
  float alpha = 1.0; //scalar value for saxpy
  cublasHandle_t handle;

  // Allocate Unified Memory accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  init<<<numBlocks, blockSize>>>(N, x, y);

  // Start cuBLAS code
  cublasCreate(&handle);

  // y = alpha*x + y
  cublasSaxpy(handle, N, &alpha, x, 1, y, 1);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  // End of cuBLAS code
  cublasDestroy(handle);

  return 0;
}
