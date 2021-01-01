#include "equalizer.cuh"

#include "error_checker.cuh"
#include "hsl.cuh"

extern "C" {
    #include <stdio.h>
    #include "cexception/lib/CException.h"
    #include "log.h"
    #include "errors.h"
}

#define N_BINS 500

__global__ void compute_histogram(const float *image,
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
        atomicAdd(&(bins_s[(unsigned int)__float2int_rn(image[i] * (N_BINS - 1))]), 1);
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
        const uint8_t *pixel_offset = &rgb_image[tid * 3];

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

int equalize(uint8_t *input, unsigned int width, unsigned int height, uint8_t **output)
{
    CEXCEPTION_T e = NO_ERROR;
    cudaError_t err = cudaSuccess;

    uint8_t *d_rgb_image = NULL;
    uint8_t *d_output_image = NULL;
    unsigned int *d_histogram = NULL;

    hsl_image_t d_hsl_image = {
        .h = NULL,
        .s = NULL,
        .l = NULL
    };

    Try {
        // Allocate memory for the image on the device
        gpuErrorCheck( cudaMalloc((void**)&d_rgb_image, 3 * width * height * sizeof(uint8_t)) );
        gpuErrorCheck( cudaMemcpy(d_rgb_image, input, 3 * width * height, cudaMemcpyHostToDevice) );

        gpuErrorCheck( cudaMalloc((void**)&(d_hsl_image.h), width * height * sizeof(int)) );
        gpuErrorCheck( cudaMalloc((void**)&(d_hsl_image.s), width * height * sizeof(float)) );
        gpuErrorCheck( cudaMalloc((void**)&(d_hsl_image.l), width * height * sizeof(float)) );

        // Allocate memory for the output
        *output = (uint8_t *)calloc(3 * width * height, sizeof(uint8_t));

        if(NULL == (*output))
        {
            Throw(UNALLOCATED_MEMORY);
        }

        gpuErrorCheck( cudaMalloc((void**)&d_output_image, 3 * width * height * sizeof(uint8_t)) );

        gpuErrorCheck( cudaMalloc((void**)&d_histogram, N_BINS * sizeof(unsigned int)) );

        int threadsPerBlock = 512;
        int blocksPerGrid = ((width * height) + threadsPerBlock - 1) / threadsPerBlock;

        // **************************************
        // STEP 1 - convert every pixel from RGB to HSL
        convert_rgb_to_hsl<<<blocksPerGrid, threadsPerBlock>>>(d_rgb_image, d_hsl_image, width * height);

        // **************************************
        // STEP 2 - compute the histogram of the luminance for each pixel
        blocksPerGrid = 30;
        compute_histogram<<<blocksPerGrid, BLOCK_SIZE, N_BINS * sizeof(unsigned int)>>>(d_hsl_image.l, d_histogram, (width * height));

        // **************************************
        // STEP 3 - compute the cumulative distribution function

        // **************************************
        // STEP 4 - compute the normalized cumulative distribution function

        // **************************************
        // STEP 5 - apply the normalized CDF to the luminance for each pixel

        // **************************************
        // STEP 6 - convert each HSL pixel back to RGB
        blocksPerGrid = ((width * height) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        convert_hsl_to_rgb<<<blocksPerGrid, BLOCK_SIZE>>>(d_hsl_image, d_output_image, width * height);

        // Copy the result back from the device
        gpuErrorCheck( cudaMemcpy(*output, d_output_image, 3 * width * height, cudaMemcpyDeviceToHost) );
    } Catch(e) {
        log_error("Caught exception %d while equalizing image!", e);
    }

    cudaFree(d_rgb_image);
    cudaFree(d_output_image);
    cudaFree(d_histogram);
    cudaFree(d_hsl_image.h);
    cudaFree(d_hsl_image.s);
    cudaFree(d_hsl_image.l);

    return e;
}