#include "equalizer.cuh"

#include "hsl.cuh"

#define N_BINS 500

__global__ void compute_histogram(const char *image,
                                  unsigned int *bins,
                                  unsigned int num_elements)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned int bins_s[];
    for (unsigned int binIdx = threadIdx.x; binIdx < N_BINS; binIdx += blockDim.x)
    {
        bins_s[binIdx] = 0;
    }

    __syncthreads();

    for (unsigned int i = tid; i < num_elements; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&(bins_s[(unsigned int)image[i]]), 1);
    }

    __syncthreads();

    for (unsigned int binIdx = threadIdx.x; binIdx < N_BINS; binIdx += blockDim.x)
    {
        atomicAdd(&(bins[binIdx]), bins_s[binIdx]);
    }
}

__global__ void convert_rgb_to_hsl(const uint8_t *rgb_image,
                                   hsl_image_t hsl_image,
                                   unsigned int num_elements)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_elements)
    {
        const uint8_t *pixel_offset = rgb_image + (tid * 3);

        rgb_pixel_t rgb_pixel = {
            .r = pixel_offset[0],
            .g = pixel_offset[1],
            .b = pixel_offset[2]
        };

        hsl_pixel_t hsl_pixel = { .h = 0, .s = 0, .l = 0 };

        rgb_to_hsl(rgb_pixel, &hsl_pixel);

        hsl_image.h[tid] = hsl_pixel.h;
        hsl_image.s[tid] = hsl_pixel.s;
        hsl_image.l[tid] = hsl_pixel.l;
    }
}

__global__ void convert_hsl_to_rgb(const hsl_image_t hsl_image,
                                   uint8_t *rgb_image,
                                   unsigned int num_elements)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_elements)
    {
        uint8_t *pixel_offset = rgb_image + (tid * 3);

        rgb_pixel_t rgb_pixel = { .r = 0, .g = 0, .b = 0 };

        hsl_pixel_t hsl_pixel = {
            .h = hsl_image.h[tid],
            .s = hsl_image.s[tid],
            .l = hsl_image.l[tid]
        };

        hsl_to_rgb(hsl_pixel, &rgb_pixel);

        pixel_offset[0] = rgb_pixel.r;
        pixel_offset[1] = rgb_pixel.g;
        pixel_offset[2] = rgb_pixel.b;
    }
}

void equalize(uint8_t *input, unsigned int width, unsigned int height, uint8_t **output)
{
    cudaError_t err = cudaSuccess;

    uint8_t *d_rgb_image = NULL;
    uint8_t *d_output_image = NULL;
    unsigned int *d_histogram = NULL;

    hsl_image_t d_hsl_image = {
        .h = NULL,
        .s = NULL,
        .l = NULL
    };

    // Allocate memory for the image on the device
    cudaMalloc((void**)&d_rgb_image, 3 * width * height);
    cudaMemcpy(d_rgb_image, input, 3 * width * height, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(d_hsl_image.h), width * height);
    cudaMalloc((void**)&(d_hsl_image.s), width * height);
    cudaMalloc((void**)&(d_hsl_image.l), width * height);

    // Allocate memory for the output
    *output = (uint8_t *)calloc(3 * width * height, sizeof(uint8_t));
    cudaMalloc((void**)&d_output_image, 3 * width * height);

    cudaMalloc((void**)&d_histogram, N_BINS * sizeof(unsigned int));

    // TODO : here the kernel must be started
    int threadsPerBlock = 512;
    int blocksPerGrid = ((width * height) + threadsPerBlock - 1) / threadsPerBlock;
    convert_rgb_to_hsl<<<blocksPerGrid, threadsPerBlock>>>(d_rgb_image, d_hsl_image, width * height);
    //compute_histogram<<<1024, threadsPerBlock, N_BINS * sizeof(unsigned int)>>>(d_rgb_image, (int)(3 * width * height));
    convert_hsl_to_rgb<<<blocksPerGrid, threadsPerBlock>>>(d_hsl_image, d_output_image, width * height);

    err = cudaGetLastError();

    // Copy the result back from the device
    cudaMemcpy(*output, d_output_image, 3 * width * height, cudaMemcpyDeviceToHost);

    cudaFree(d_rgb_image);
    cudaFree(d_output_image);
    cudaFree(d_histogram);
    cudaFree(d_hsl_image.h);
    cudaFree(d_hsl_image.s);
    cudaFree(d_hsl_image.l);
}