#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "hsl.h"
#include "log.h"
#include "arguments.h"

#include "equalizer.h"
#include "cexception/lib/CException.h"
#include "errors.h"

#define N_BINS 500

void histogram_calc(unsigned int *hist, float *lum, size_t img_size)
{
    #pragma omp single
    {
        if(NULL == hist)
        {
            Throw(UNALLOCATED_MEMORY);
        }
        if(NULL == lum)
        {
            Throw(UNALLOCATED_MEMORY);
        }
    }

    #pragma omp for
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
    if(NULL == cdf)
    {
        Throw(UNALLOCATED_MEMORY);
    }
    if(NULL == buf)
    {
        Throw(UNALLOCATED_MEMORY);
    }

    cdf[0] = buf[0];
    for(unsigned int index = 1; index < buf_size; index++)
    {
        cdf[index] = cdf[index - 1] + buf[index];
    }
}

int equalize(uint8_t *input, unsigned int width, unsigned int height, uint8_t **output)
{
    CEXCEPTION_T e = NO_ERROR;

    hsl_image_t hsl_image = {
        .h = NULL,
        .s = NULL,
        .l = NULL
    };

    unsigned int *histogram = NULL;
    unsigned int *cdf = NULL;
    float *cdf_norm = NULL;

    Try {
        if(NULL == input)
        {
            Throw(UNALLOCATED_MEMORY);
        }

        if(NULL == output)
        {
            Throw(UNALLOCATED_MEMORY);
        }

        hsl_image.h = calloc(width * height, sizeof(int));
        hsl_image.s = calloc(width * height, sizeof(float));
        hsl_image.l = calloc(width * height, sizeof(float));

        if(NULL == hsl_image.h)
        {
            Throw(UNALLOCATED_MEMORY);
        }

        if(NULL == hsl_image.s)
        {
            Throw(UNALLOCATED_MEMORY);
        }

        if(NULL == hsl_image.l)
        {
            Throw(UNALLOCATED_MEMORY);
        }

        #pragma omp parallel \
            shared(input, hsl_image, histogram, cdf, cdf_norm, output, height, width)
        {
            // **************************************
            // STEP 1 - convert every pixel from RGB to HSL
            #pragma omp for collapse(2)
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

            // **************************************
            // STEP 2 - compute the histogram of the luminance for each pixel
            #pragma omp single
            {
                // Calculate the histogram by multiplying the luminance by N_BINS - 1
                histogram = calloc(N_BINS, sizeof(unsigned int));

                if(NULL == histogram)
                {
                    Throw(UNALLOCATED_MEMORY);
                }

                log_info("Starting histogram calculation..");
            }

            histogram_calc(histogram, hsl_image.l, width * height);

            // **************************************
            // STEP 3 - compute the cumulative distribution function
            #pragma omp single
            {
                // Calculate the cdf
                cdf = calloc(N_BINS, sizeof(unsigned int));

                if(NULL == cdf)
                {
                    Throw(UNALLOCATED_MEMORY);
                }

                log_info("Starting cdf calculation..");
                cdf_calc(cdf, histogram, N_BINS);

                // Normalize the cdf so that it can be used as luminance
                cdf_norm = calloc(N_BINS, sizeof(float));

                if(NULL == cdf_norm)
                {
                    Throw(UNALLOCATED_MEMORY);
                }

                log_info("Starting normalized cdf calculation..");
            }

            // **************************************
            // STEP 4 - compute the normalized cumulative distribution function
            #pragma omp for
            for(unsigned int bin = 0; bin < N_BINS; bin++)
            {
                cdf_norm[bin] = (float)(cdf[bin] - cdf[0]) / ((width * height) - cdf[0]) * (N_BINS - 1);
            }

            // **************************************
            // STEP 5 - apply the normalized CDF to the luminance for each pixel
            #pragma omp for collapse(2)
            for(unsigned int x = 0; x < height; x++)
            {
                for(unsigned int y = 0; y < width; y++)
                {
                    // Multiply by N_BINS - 1 to prevent overflow
                    hsl_image.l[(width * x) + y] = cdf_norm[(unsigned int)roundf(hsl_image.l[(width * x) + y] * (N_BINS - 1))] / (N_BINS - 1);
                }
            }

            #pragma omp single
            {
                // Convert back to rgb and save the image
                *output = calloc(width * height * 3, sizeof(uint8_t));
                
                if(NULL == (*output))
                {
                    Throw(UNALLOCATED_MEMORY);
                }
            }

            // **************************************
            // STEP 6 - convert each HSL pixel back to RGB
            #pragma omp for collapse(2)
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
            if(NULL == pp_histogram)
            {
                Throw(UNALLOCATED_MEMORY);
            }

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
    } Catch(e) {
        log_error("Catched error %d in %s %d", e, __FILE__, __LINE__);
    }

    if(NULL != hsl_image.h)
    {
        free(hsl_image.h);
    }

    if(NULL != hsl_image.s)
    {
        free(hsl_image.s);
    }

    if(NULL != hsl_image.l)
    {
        free(hsl_image.l);
    }

    if(NULL != histogram)
    {
        free(histogram);
    }

    if(NULL != cdf)
    {
        free(cdf);
    }

    if(NULL != cdf_norm)
    {
        free(cdf_norm);
    }

    return e;
}
