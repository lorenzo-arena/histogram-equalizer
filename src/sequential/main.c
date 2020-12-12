#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <argp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "hsl.h"
#include "log.h"
#include "stopwatch.h"

#define N_BINS 500

const char *argp_program_version =
" 1.0";

struct arguments
{
    char *args[2];                /* image and output */
    bool stopwatch;
    bool plot;
    bool log_histogram;
};

void set_default_arguments(struct arguments *arguments)
{
    arguments->args[0] = "";
    arguments->args[1] = "";
    arguments->stopwatch = false;
    arguments->plot = false;
    arguments->log_histogram = false;
}

static struct argp_option options[] =
{
    {"stopwatch", 's', 0, 0, "Enable stopwatch usage", 0},
    {"plot", 'p', 0, 0, "Enable histogram plot", 0},
    {"log_histogram", 'l', 0, 0, "Enable histogram log", 0},
    {0}
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
        case 's':
        {
            arguments->stopwatch = true;
            break;
        }
        case 'p':
        {
            arguments->plot = true;
            break;
        }
        case 'l':
        {
            arguments->log_histogram = true;
            break;
        }
        case ARGP_KEY_ARG:
        {
            if (state->arg_num >= 2)
            {
                argp_usage(state);
            }
            arguments->args[state->arg_num] = arg;
            break;
        }
        case ARGP_KEY_END:
        {
            if (state->arg_num < 2)
            {
                argp_usage(state);
            }
            break;
        }
        default:
        return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

static char args_doc[] = "image output";

static char doc[] =
"histogram-equalizer-sequential -- Used to equalize the histogram of an image.";

static struct argp argp = {options, parse_opt, args_doc, doc, NULL, NULL, NULL};


void histogram_calc(unsigned int *hist, float *lum, size_t img_size)
{
    assert(hist != NULL);
    assert(lum != NULL);

    for(unsigned int index = 0; index < img_size; index++)
    {
        // The luminance should be between 0 and 1; multiply by
	// N_BINS - 1 so that it can't cause overflow
        hist[(int)floorf(lum[index] * (N_BINS - 1))]++;
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
    struct arguments arguments;

    set_default_arguments(&arguments);

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if(arguments.stopwatch)
    {
        stopwatch_start();
    }

    uint8_t* rgb_image = stbi_load(arguments.args[0], &width, &height, &bpp, STBI_rgb);

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

    for(int x = 0; x < height; x++)
    {
        for(int y = 0; y < width; y++)
        {
            unsigned char* pixel_offset = rgb_image + (((width * x) + y) * STBI_rgb);
            
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
    for(int bin = 0; bin < N_BINS; bin++)
    {
        cdf_norm[bin] = (float)(cdf[bin]) / (width * height);
    }

    // Apply the normalized cdf to the luminance
    for(int x = 0; x < height; x++)
    {
        for(int y = 0; y < width; y++)
        {
	    // Multiply by N_BINS - 1 to prevent overflow
            hsl_image.l[(width * x) + y] = cdf_norm[(int)floorf(hsl_image.l[(width * x) + y] * (N_BINS - 1))];
        }
    }

    // Convert back to rgb and save the image
    for(int x = 0; x < height; x++)
    {
        for(int y = 0; y < width; y++)
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

    stbi_write_jpg(arguments.args[1], width, height, STBI_rgb, rgb_image, 100);

    if(arguments.log_histogram)
    {
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
    }

    if(arguments.plot)
    {
        FILE *gnuplot = popen("gnuplot -persistent", "w");
        fprintf(gnuplot, "plot '-'\n");
        for (int bin = 1; bin < N_BINS; bin++)
        {
            fprintf(gnuplot, "%d %d\n", bin, histogram[bin]);
        }
        fprintf(gnuplot, "e\n");
        fprintf(gnuplot,"set xrange[1:%d]\n", N_BINS);
        fflush(gnuplot);
    }

    // Clean up buffers
    stbi_image_free(rgb_image);

    free(hsl_image.h);
    free(hsl_image.s);
    free(hsl_image.l);
    free(histogram);
    free(cdf);
    free(cdf_norm);

    if(arguments.stopwatch)
    {
        stopwatch_stop();

        struct timespec elapsed = stopwatch_get_elapsed();

        log_info("Elapsed time: %ld.%09ld",
            elapsed.tv_sec,
            elapsed.tv_nsec);
    }

    return 0;
}
