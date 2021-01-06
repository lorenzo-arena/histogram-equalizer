#include "equalizer.cuh"

#include "error_checker.cuh"
#include "hsl.cuh"

extern "C" {
    #include <stdio.h>
    #include "cexception/lib/CException.h"
    #include "log.h"
    #include "errors.h"
    #include "arguments.h"
    #include "defines.h"
}

#define BLOCK_SIZE 512

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
        const rgb_pixel_t rgb_pixel = *(rgb_pixel_t *)(&rgb_image[tid * 4]);

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
        uint8_t *pixel_offset = &rgb_image[tid * 4];

        rgb_pixel_t rgb_pixel = { .r = 0, .g = 0, .b = 0, .a = 0xFF };

        hsl_pixel_t hsl_pixel = {
            .h = hsl_image.h[tid],
            .s = hsl_image.s[tid],
            .l = hsl_image.l[tid]
        };

        hsl_to_rgb(hsl_pixel, &rgb_pixel);

        pixel_offset[0] = rgb_pixel.r;
        pixel_offset[1] = rgb_pixel.g;
        pixel_offset[2] = rgb_pixel.b;
        pixel_offset[3] = rgb_pixel.a;
    }
}

__global__ void compute_cdf(unsigned int *input, unsigned int *output, int input_size)
{
    __shared__ unsigned int sh_out[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input_size)
    {
        sh_out[threadIdx.x] = input[tid];
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();

        if(threadIdx.x >= stride)
        {
            sh_out[threadIdx.x] += sh_out[threadIdx.x - stride];
        }
    }

    __syncthreads();

    if (tid < input_size)
    {
        output[tid] = sh_out[threadIdx.x];
    }
}

__global__ void compute_normalized_cdf(unsigned int *cdf, float *cdf_norm, int cdf_size, int norm_factor)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < cdf_size)
    {
        cdf_norm[tid] = ((float)(cdf[tid] - cdf[0]) / (norm_factor - cdf[0])) * (cdf_size - 1);
    }
}

__global__ void apply_normalized_cdf(const float *cdf_norm, const hsl_image_t hsl_image, int cdf_size, int image_size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < image_size)
    {
        hsl_image.l[tid] = cdf_norm[(unsigned int)__float2int_rn(hsl_image.l[tid] * (cdf_size - 1))] / (cdf_size - 1);
    }
}

int equalize(uint8_t *input, unsigned int width, unsigned int height, uint8_t **output)
{
    CEXCEPTION_T e = NO_ERROR;

    int blocksPerGrid = 0;

    uint8_t *d_rgb_image = NULL;
    uint8_t *d_output_image = NULL;
    unsigned int *d_histogram = NULL;
    unsigned int *d_cdf = NULL;
    float *d_cdf_norm = NULL;

    hsl_image_t d_hsl_image = {
        .h = NULL,
        .s = NULL,
        .l = NULL
    };

    Try {
        // Allocate memory for the image on the device
        gpuErrorCheck( cudaMalloc((void**)&d_rgb_image, 4 * width * height * sizeof(uint8_t)) );
        gpuErrorCheck( cudaMemcpy(d_rgb_image, input, 4 * width * height, cudaMemcpyHostToDevice) );

        gpuErrorCheck( cudaMalloc((void**)&(d_hsl_image.h), width * height * sizeof(int)) );
        gpuErrorCheck( cudaMalloc((void**)&(d_hsl_image.s), width * height * sizeof(float)) );
        gpuErrorCheck( cudaMalloc((void**)&(d_hsl_image.l), width * height * sizeof(float)) );

        // Allocate memory for the output
        *output = (uint8_t *)calloc(4 * width * height, sizeof(uint8_t));

        check_pointer(*output);

        gpuErrorCheck( cudaMalloc((void**)&d_output_image, 4 * width * height * sizeof(uint8_t)) );

        gpuErrorCheck( cudaMalloc((void**)&d_histogram, N_BINS * sizeof(unsigned int)) );
        gpuErrorCheck( cudaMalloc((void**)&d_cdf, N_BINS * sizeof(unsigned int)) );
        gpuErrorCheck( cudaMalloc((void**)&d_cdf_norm, N_BINS * sizeof(float)) );

        // **************************************
        // STEP 1 - convert every pixel from RGB to HSL
        blocksPerGrid = ((width * height) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        convert_rgb_to_hsl<<<blocksPerGrid, BLOCK_SIZE>>>(d_rgb_image, d_hsl_image, (width * height));

        // **************************************
        // STEP 2 - compute the histogram of the luminance for each pixel
        blocksPerGrid = 30;
        compute_histogram<<<blocksPerGrid, BLOCK_SIZE, N_BINS * sizeof(unsigned int)>>>(d_hsl_image.l, d_histogram, (width * height));

        // **************************************
        // STEP 3 - compute the cumulative distribution function by applying the parallelized
        // version of the scan algorithm
        blocksPerGrid = (N_BINS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_cdf<<<blocksPerGrid, BLOCK_SIZE>>>(d_histogram, d_cdf, N_BINS);

        // **************************************
        // STEP 4 - compute the normalized cumulative distribution function
        blocksPerGrid = (N_BINS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_normalized_cdf<<<blocksPerGrid, BLOCK_SIZE>>>(d_cdf, d_cdf_norm, N_BINS, (width * height));

        // **************************************
        // STEP 5 - apply the normalized CDF to the luminance for each pixel
        blocksPerGrid = ((width * height) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_normalized_cdf<<<blocksPerGrid, BLOCK_SIZE>>>(d_cdf_norm, d_hsl_image, N_BINS, (width * height));

        // **************************************
        // STEP 6 - convert each HSL pixel back to RGB
        blocksPerGrid = ((width * height) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        convert_hsl_to_rgb<<<blocksPerGrid, BLOCK_SIZE>>>(d_hsl_image, d_output_image, width * height);

        // Copy the result back from the device
        gpuErrorCheck( cudaMemcpy(*output, d_output_image, 4 * width * height, cudaMemcpyDeviceToHost) );

        if(arguments.log_histogram)
        {
            unsigned int *h_histogram = NULL;
            unsigned int *h_cdf = NULL;
            float *h_cdf_norm = NULL;

            h_histogram = (unsigned int *)calloc(N_BINS, sizeof(unsigned int));
            h_cdf = (unsigned int *)calloc(N_BINS, sizeof(unsigned int));
            h_cdf_norm = (float *)calloc(N_BINS, sizeof(float));

            check_pointer(h_histogram);
            check_pointer(h_cdf);
            check_pointer(h_cdf_norm);

            gpuErrorCheck( cudaMemcpy(h_histogram, d_histogram, N_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
            gpuErrorCheck( cudaMemcpy(h_cdf, d_cdf, N_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
            gpuErrorCheck( cudaMemcpy(h_cdf_norm, d_cdf_norm, N_BINS * sizeof(float), cudaMemcpyDeviceToHost) );

            log_info("Printing histogram..");
            for(int bin = 0; bin < N_BINS; bin++)
            {
                log_info("%d:%d", bin, h_histogram[bin]);
            }

            log_info("Printing cdf..");
            for(int bin = 0; bin < N_BINS; bin++)
            {
                log_info("%d:%d", bin, h_cdf[bin]);
            }

            log_info("Printing normalized cdf..");
            for(int bin = 0; bin < N_BINS; bin++)
            {
                log_info("%d:%g", bin, h_cdf_norm[bin]);
            }

            free(h_histogram);
            free(h_cdf);
            free(h_cdf_norm);
        }
    } Catch(e) {
        log_error("Caught exception %d while equalizing image!", e);
    }

    cudaFree(d_rgb_image);
    cudaFree(d_output_image);
    cudaFree(d_histogram);
    cudaFree(d_cdf);
    cudaFree(d_cdf_norm);
    cudaFree(d_hsl_image.h);
    cudaFree(d_hsl_image.s);
    cudaFree(d_hsl_image.l);

    return e;
}