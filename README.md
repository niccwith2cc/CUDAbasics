# CUDAbasics

This repository contains introductory materials and examples to help you get started with CUDA programming. This is adapted from this
[technical blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) and from the [official](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) CUDA guide

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Starting Simple](#starting-simple)

## Introduction
CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to leverage the power of NVIDIA GPUs for general-purpose computing.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- An NVIDIA GPU with CUDA capability
- NVIDIA CUDA Toolkit installed
- Basic knowledge of C/C++ programming

## Installation
To install the CUDA Toolkit, follow these steps:
1. Download the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
2. Follow the installation instructions for your operating system.

## Starting Simple

Let's write a program that adds the elements of two arrays with a million elements each. check ``` adding.cu ```.

Now I want to get this computation running (in parallel) on the many cores of a GPU. It’s actually pretty easy to take the first steps.

### Adding Function
First, I just have to turn our add function into a function that the GPU can run, called a kernel in CUDA. To do this, all I have to do is add the specifier ```__global__``` to the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code. These ```__global__``` functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.

### Memory Allocation on GPU

To compute on the GPU, I need to allocate memory accessible by the GPU. Unified Memory in CUDA makes this easy by providing a single memory space accessible by all GPUs and CPUs in your system. To allocate data in unified memory, call ```cudaMallocManaged()```, which returns a pointer that you can access from host (CPU) code or device (GPU) code. To free the data, just pass the pointer to ```cudaFree()```.

I just need to replace the calls to ```new``` in the code above with calls to ```cudaMallocManaged()```, and replace calls to ```delete[]``` with calls to ```cudaFree```.

Finally, I need to launch the ```add()``` kernel, which invokes it on the GPU. CUDA kernel launches are specified using the triple angle bracket syntax ```<<< >>>```. I just have to add it to the call to ```add``` before the parameter list, this line launches one GPU thread to run ```add()```.

Just one more thing: I need the CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread). To do this I just call ```cudaDeviceSynchronize()``` before doing the final error checking on the CPU.

### Compiling and Running

CUDA files have the file extension ```.cu``` and compile it with nvcc, the CUDA C++ compiler. You can also profile it using ```nvprof```

``` 
> nvcc add.cu -o add_cuda 
> nsys profile -o profile_report --force-overwrite true ./your_cuda_program
> nsys-ui
```

## Adding More GPU Threads

Now that you’ve run a kernel with one thread that does some computation, how do you make it parallel? The key is in CUDA’s ```<<<1, 1>>>``` syntax. This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU. There are two parameters here, but let’s start by changing the second one: the number of threads in a thread block. CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose ```<<<1, 256>>> ```

BUT

If I run the code with only this change, it will do the computation once per thread, rather than spreading the computation across the parallel threads. To do it properly, I need to modify the kernel. CUDA C++ provides keywords that let kernels get the indices of the running threads. Specifically, ```threadIdx.x``` contains the index of the current thread within its block, and ```blockDim.x``` contains the number of threads in the block. I’ll just modify the loop to stride through the array with parallel threads. You can check the implementation in ```adding-block.cu```.

```
__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
```

## Exploiting SM

CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. Each SM can run multiple concurrent thread blocks. As an example, a Tesla P100 GPU based on the Pascal GPU Architecture has 56 SMs, each capable of supporting up to 2048 active threads. To take full advantage of all these threads, I should launch the kernel with multiple thread blocks.

By now you may have guessed that the first parameter of the execution configuration specifies the number of thread blocks. Together, the blocks of parallel threads make up what is known as the grid. Since I have ```N``` elements to process, and 256 threads per block, I just need to calculate the number of blocks to get at least N threads. I simply divide ```N``` by the block size (being careful to round up in case ```N``` is not a multiple of ```blockSize```).
```
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```

I also need to update the kernel code to take into account the entire grid of thread blocks. CUDA provides ```gridDim.x```, which contains the number of blocks in the grid, and ```blockIdx.x```, which contains the index of the current thread block in the grid. The idea is that each thread gets its index by computing the offset to the beginning of its block (the block index times the block size: ```blockIdx.x * blockDim.x```) and adding the thread’s index within the block (```threadIdx.x```). The code ```blockIdx.x * blockDim.x + threadIdx.x``` is idiomatic CUDA. You can check ```adding-grid.cu``` for the implementation

```
__global__ void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```

Happy coding!