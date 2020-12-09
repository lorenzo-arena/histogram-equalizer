#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "hsl.h"
#include "log.h"
#include "stopwatch.h"

#define N_BINS 500

void histogram_calc(unsigned int *hist, float *lum, size_t img_size)
{
    assert(hist != NULL);
    assert(lum != NULL);

    for(unsigned int index = 0; index < img_size; index++)
    {
        hist[(int)round(lum[index] * N_BINS)]++;
    }
}

void cdf_calc(unsigned int *cdf, unsigned int *buf, size_t buf_size)
{
    assert(cdf != NULL);
    assert(buf != NULL);

    cdf[0] = buf[0];
    for(unsigned int index = 1; index < buf_size; index++)
    {
        cdf[index] = cdf[index - 1] + buf[index];
    }
}

int main(int argc, char **argv)
{
    int width, height, bpp;

    stopwatch_start();

    if(argc <= 1)
    {
        log_error("Usage: %s <image-path>\n", argv[0]);
        return 1;
    }

    uint8_t* rgb_image = stbi_load(argv[1], &width, &height, &bpp, STBI_rgb);

    log_info("Width %d", width);
    log_info("Height %d", height);

    // Image BPP will be 4 but the reading is forced to be RGB only
    log_info("BPP %d", bpp);

    hsl_image_t hsl_image = {
        .h = calloc(width * height, sizeof(int)),
        .s = calloc(width * height, sizeof(float)),
        .l = calloc(width * height, sizeof(float))
    };

    assert(hsl_image.h != NULL);
    assert(hsl_image.s != NULL);
    assert(hsl_image.l != NULL);

    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            unsigned char* pixel_offset = rgb_image + (((width * x) + y) * STBI_rgb);
            
            rgb_pixel_t rgb_pixel = {
                .r = pixel_offset[0],
                .g = pixel_offset[1],
                .b = pixel_offset[2]
            };

            hsl_pixel_t hsl_pixel = { .h = 0, .s = 0, .l = 0 };

            rgb_to_hsl(rgb_pixel, &hsl_pixel);

            //log_info("Calculating HSL for x %d, y %d", x, y);
            hsl_image.h[(width * x) + y] = hsl_pixel.h;
            hsl_image.s[(width * x) + y] = hsl_pixel.s;
            hsl_image.l[(width * x) + y] = hsl_pixel.l;
        }
    }

    // Calculate the histogram by multiplying the luminance by 1000
    unsigned int *histogram = calloc(N_BINS, sizeof(unsigned int));
    log_info("Starting histogram calculation..");
    histogram_calc(histogram, hsl_image.l, width * height);

    // Important: since it can be that a lot of pixel are black or white,
    // all calculations on the histogram must exclude the extremes

    // Calculate the cdf
    unsigned int *cdf = calloc(N_BINS - 1, sizeof(unsigned int));
    log_info("Starting cdf calculation..");
    cdf_calc(cdf, &histogram[1], N_BINS - 1);

    // Normalize the cdf so that it can be used as luminance
    float *cdf_norm = calloc(N_BINS - 1, sizeof(float));
    log_info("Starting normalized cdf calculation..");
    for(int bin = 0; bin < (N_BINS - 2); bin++)
    {
        float temp = (float)(cdf[bin] - cdf[0]);
        float temp2 = (float)(cdf[N_BINS - 2] - cdf[0]);
        float temp3 = (float)(N_BINS - 1);
        cdf_norm[bin] = (float)(temp) / temp2;
        //cdf_norm[bin] = (unsigned int)round(((float)(cdf[bin] - cdf[0]) / (float)(cdf[N_BINS - 2] - cdf[0])) * (float)(N_BINS - 1));
    }

    // Apply the normalized cdf to the luminance
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            hsl_image.l[(width * x) + y] = cdf_norm[(int)round(hsl_image.l[(width * x) + y] * N_BINS)];
        }
    }

    // Convert back to rgb and save the image
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            unsigned char* pixel_offset = rgb_image + (((width * x) + y) * STBI_rgb);

            rgb_pixel_t rgb_pixel = { .r = 0, .g = 0, .b = 0 };

            hsl_pixel_t hsl_pixel = {
                .h = hsl_image.h[(width * x) + y],
                .s = hsl_image.s[(width * x) + y],
                .l = hsl_image.l[(width * x) + y]
            };

            hsl_to_rgb(hsl_pixel, &rgb_pixel);

            pixel_offset[0] = rgb_pixel.r;
            pixel_offset[1] = rgb_pixel.g;
            pixel_offset[2] = rgb_pixel.b;
        }
    }

    stbi_write_jpg("test.jpg", width, height, STBI_rgb, rgb_image, 100);

    // TODO: add an option for print
    log_info("Printing histogram..");
    for(int bin = 0; bin < N_BINS; bin++)
    {
        log_info("%d:%d", bin, histogram[bin]);
    }

    log_info("Printing cdf..");
    for(int bin = 0; bin < (N_BINS - 1); bin++)
    {
        log_info("%d:%d", bin, cdf[bin]);
    }

    log_info("Printing normalized cdf..");
    for(int bin = 0; bin < (N_BINS - 1); bin++)
    {
        log_info("%d:%g", bin, cdf_norm[bin]);
    }

    // TODO: add an option for plot
    FILE *gnuplot = popen("gnuplot -persistent", "w");
    fprintf(gnuplot, "plot '-'\n");
    for (int bin = 1; bin < N_BINS; bin++)
    {
        fprintf(gnuplot, "%d %d\n", bin, histogram[bin]);
    }
    fprintf(gnuplot, "e\n");
    fprintf(gnuplot,"set xrange[1:%d]\n", N_BINS);
    fflush(gnuplot);

    // Clean up buffers
    stbi_image_free(rgb_image);

    free(hsl_image.h);
    free(hsl_image.s);
    free(hsl_image.l);
    free(histogram);
    free(cdf);
    free(cdf_norm);

    stopwatch_stop();

    struct timespec elapsed = stopwatch_get_elapsed();

    log_info("Elapsed time: %ld.%09ld",
        elapsed.tv_sec,
        elapsed.tv_nsec);

    return 0;
}