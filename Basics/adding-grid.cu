// First, I just have to turn our add function into a function that the GPU can run, called a kernel in CUDA. To do this, all I have to do is add the specifier __global__ to the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.

#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays

// Find the number of CUDA cores per SM
 // Your RTX 3050 has 16 SMs, each with 128 CUDA cores, giving you 2048 CUDA cores total.

__global__ void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1000000;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Optimal configuration
  int blockSize = 1024; // Max threads per block
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);

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
  
  return 0;
}