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



Happy coding!