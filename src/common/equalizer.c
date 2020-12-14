#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

#include "hsl.h"
#include "log.h"
#include "arguments.h"

#include "equalizer.h"

#define N_BINS 500

void histogram_calc(unsigned int *hist, float *lum, size_t img_size)
{
    assert(hist != NULL);
    assert(lum != NULL);

    #pragma omp parallel for
    for(unsigned int index = 0; index < img_size; index++)
    {
        // The luminance should be between 0 and 1; multiply by
        // N_BINS - 1 so that it can't cause overflow
        #pragma omp atomic
        hist[(int)roundf(lum[index] * (N_BINS - 1))]++;
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

void equalize(uint8_t *input, unsigned int width, unsigned int height, uint8_t **output)
{
    hsl_image_t hsl_image = {
        .h = calloc(width * height, sizeof(int)),
        .s = calloc(width * height, sizeof(float)),
        .l = calloc(width * height, sizeof(float))
    };

    assert(hsl_image.h != NULL);
    assert(hsl_image.s != NULL);
    assert(hsl_image.l != NULL);
    assert(input != NULL);
    assert(output != NULL);

    #pragma omp parallel for collapse(2)
    for(unsigned int x = 0; x < height; x++)
    {
        for(unsigned int y = 0; y < width; y++)
        {
            uint8_t *pixel_offset = input + (((width * x) + y) * 3);
            
            rgb_pixel_t rgb_pixel = {
                .r = pixel_offset[0],
                .g = pixel_offset[1],
                .b = pixel_offset[2]
            };

            hsl_pixel_t hsl_pixel = { .h = 0, .s = 0, .l = 0 };

            rgb_to_hsl(rgb_pixel, &hsl_pixel);

            hsl_image.h[(width * x) + y] = hsl_pixel.h;
            hsl_image.s[(width * x) + y] = hsl_pixel.s;
            hsl_image.l[(width * x) + y] = hsl_pixel.l;
        }
    }

    // Calculate the histogram by multiplying the luminance by N_BINS - 1
    unsigned int *histogram = calloc(N_BINS, sizeof(unsigned int));
    log_info("Starting histogram calculation..");
    histogram_calc(histogram, hsl_image.l, width * height);

    // Calculate the cdf
    unsigned int *cdf = calloc(N_BINS, sizeof(unsigned int));
    log_info("Starting cdf calculation..");
    cdf_calc(cdf, histogram, N_BINS);

    // Normalize the cdf so that it can be used as luminance
    float *cdf_norm = calloc(N_BINS, sizeof(float));
    log_info("Starting normalized cdf calculation..");
    #pragma omp parallel for
    for(unsigned int bin = 0; bin < N_BINS; bin++)
    {
        cdf_norm[bin] = (float)(cdf[bin] - cdf[0]) / ((width * height) - cdf[0]) * (N_BINS - 1);
    }

    // Apply the normalized cdf to the luminance
    #pragma omp parallel for collapse(2)
    for(unsigned int x = 0; x < height; x++)
    {
        for(unsigned int y = 0; y < width; y++)
        {
            // Multiply by N_BINS - 1 to prevent overflow
            hsl_image.l[(width * x) + y] = cdf_norm[(unsigned int)roundf(hsl_image.l[(width * x) + y] * (N_BINS - 1))] / (N_BINS - 1);
        }
    }

    // Convert back to rgb and save the image
    *output = calloc(width * height * 3, sizeof(uint8_t));

    #pragma omp parallel for collapse(2)
    for(unsigned int x = 0; x < height; x++)
    {
        for(unsigned int y = 0; y < width; y++)
        {
            uint8_t *pixel_offset = (*output) + (((width * x) + y) * 3);

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

    if(arguments.log_histogram)
    {
        log_info("Printing histogram..");
        for(int bin = 0; bin < N_BINS; bin++)
        {
            log_info("%d:%d", bin, histogram[bin]);
        }

        log_info("Printing cdf..");
        for(int bin = 0; bin < N_BINS; bin++)
        {
            log_info("%d:%d", bin, cdf[bin]);
        }

        log_info("Printing normalized cdf..");
        for(int bin = 0; bin < N_BINS; bin++)
        {
            log_info("%d:%g", bin, cdf_norm[bin]);
        }
    }

    if(arguments.plot)
    {
        // Compute the post processed image histogram
        unsigned int *pp_histogram = calloc(N_BINS, sizeof(unsigned int));
        log_info("Starting post processed histogram calculation..");
        histogram_calc(pp_histogram, hsl_image.l, width * height);

        FILE *gnuplot = popen("gnuplot -persistent", "w");
        fprintf(gnuplot, "set style line 1 lc rgb '#0025ad' lt 1 lw 0.75\n");
        fprintf(gnuplot, "set style line 2 lc rgb '#ad2500' lt 1 lw 0.75\n");
        fprintf(gnuplot, "plot '-' with lines ls 1 title 'Image histogram',\\\n");
        fprintf(gnuplot, "'-' with lines ls 2 title 'Post-processed image histogram'\n");
        for (int bin = 0; bin < N_BINS; bin++)
        {
            fprintf(gnuplot, "%d %d\n", bin, histogram[bin]);
        }
        fprintf(gnuplot, "e\n");
        for (int bin = 0; bin < N_BINS; bin++)
        {
            fprintf(gnuplot, "%d %d\n", bin, pp_histogram[bin]);
        }
        fprintf(gnuplot, "e\n");
        fprintf(gnuplot, "set xrange[0:%d]\n", N_BINS - 1);
        fflush(gnuplot);

        free(pp_histogram);
    }

    free(hsl_image.h);
    free(hsl_image.s);
    free(hsl_image.l);
    free(histogram);
    free(cdf);
    free(cdf_norm);

    return;
}