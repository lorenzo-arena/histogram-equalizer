#ifndef __UTILS_H__
#define __UTILS_H__

#ifdef __NVCC__
__device__ unsigned char max_uc(unsigned char a, unsigned char b)
#else
unsigned char max_uc(unsigned char a, unsigned char b)
#endif
{
    return (a > b) ? a : b;
}

#ifdef __NVCC__
__device__ float max_f(float a, float b)
#else
float max_f(float a, float b)
#endif
{
    return (a > b) ? a : b;
}

#ifdef __NVCC__
__device__ unsigned char min_uc(unsigned char a, unsigned char b)
#else
unsigned char min_uc(unsigned char a, unsigned char b)
#endif
{
    return (a < b) ? a : b;
}

#ifdef __NVCC__
__device__ float min_f(float a, float b)
#else
float min_f(float a, float b)
#endif
{
    return (a < b) ? a : b;
}

#endif
