#include "equalizer.cuh"

extern "C" {
    #include "hsl.h"
}

#define N_BINS 500

__global__ void compute_histogram(const char *image,
                                  unsigned int *bins,
                                  unsigned int num_elements)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned int bins_s[];
    for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x)
    {
        bins_s[binIdx] = 0;
    }

    __syncthreads();

    for (unsigned int i = tid; i < num_elements; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&(bins_s[(unsigned int)input[i]]), 1);
    }

    __syncthreads();

    for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x)
    {
        atomicAdd(&(bins[binIdx]), bins_s[binIdx]);
    }
}

__global__ void convert_hsl(const char *rgb_image,
    hsl_image_t *hsl_image)
{

}

void equalize(uint8_t *input, unsigned int width, unsigned int height, uint8_t **output)
{
    cudaError_t err = cudaSuccess;

    uint8_t *d_rgb_image = NULL;
    uint8_t *d_output_image = NULL;
    unsigned int *d_histogram = NULL;

    // Allocate memory for the image on the device
    cudaMalloc((void**)&d_rgb_image, 3 * width * height);
    cudaMemcpy(d_rgb_image, rgb_image, 3 * width * height, cudaMemcpyHostToDevice);

    // Allocate memory for the output
    *output_image = (uint8_t *)calloc(3 * width * height, sizeof(uint8_t));
    cudaMalloc((void**)&d_output_image, 3 * width * height);

    cudaMalloc((void**)&d_histogram, N_BINS * sizeof(unsigned int));

    // TODO : here the kernel must be started
    compute_histogram<<<1024, 1024, N_BINS * sizeof(unsigned int)>>>(d_rgb_image, (int)(3 * width * height));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch setToZero kernel (error code %s)!\n", cudaGetErrorString(err));
    }

    // Copy the result back from the device
    cudaMemcpy(output_image, d_output_image, 3 * width * height, cudaMemcpyDeviceToHost);

    cudaFree(d_rgb_image);
    cudaFree(d_output_image);
    cudaFree(d_histogram);
}