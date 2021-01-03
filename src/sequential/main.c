#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "log.h"
#include "equalizer.h"
#include "stopwatch.h"
#include "arguments.h"
#include "cexception/lib/CException.h"
#include "errors.h"

struct arguments arguments;

const char *argp_program_version =
" 1.0";

#ifdef _OPENMP
    const char doc[] =
        "histogram-equalizer-openmp -- Used to equalize the histogram of an image.";
#else
    const char doc[] =
        "histogram-equalizer-sequential -- Used to equalize the histogram of an image.";
#endif

int main(int argc, char **argv)
{
    int width, height, bpp;
    uint8_t* rgb_image = NULL;
    uint8_t *output_image = NULL;
    CEXCEPTION_T e;

    Try {
        set_default_arguments(&arguments);

        argp_parse(&argp, argc, argv, 0, 0, &arguments);

        rgb_image = stbi_load(arguments.args[0], &width, &height, &bpp, STBI_rgb);

        if(NULL == rgb_image)
        {
            log_error("Couldn't read image %s", arguments.args[0]);
            Throw(UNALLOCATED_MEMORY);
        }

        // Image BPP will be 4 but the reading is forced to be RGB only
        log_info("BPP %d", bpp);
        log_info("Width %d", width);
        log_info("Height %d", height);

#ifdef _OPENMP
        log_info("Starting processing with %d threads", arguments.threads);
#endif

        if(arguments.stopwatch)
        {
            stopwatch_start();
        }

        int eq_res = equalize(rgb_image, width, height, &output_image);

        if(NO_ERROR != eq_res)
        {
            log_error("Error while equalizing image!");
            Throw(eq_res);
        }

        if(NULL == output_image)
        {
            log_error("Error while equalizing image!");
            Throw(UNALLOCATED_MEMORY);
        }

        if(arguments.stopwatch)
        {
            stopwatch_stop();

            struct timespec elapsed = stopwatch_get_elapsed();

            log_info("Elapsed time: %ld.%09ld",
                elapsed.tv_sec,
                elapsed.tv_nsec);
        }

        log_info("Writing result in %s..", arguments.args[1]);
        stbi_write_jpg(arguments.args[1], width, height, STBI_rgb, output_image, 100);
    } Catch(e) {
        log_error("Catched error %d!", e);
    }

    // Clean up buffers
    if(NULL != rgb_image)
    {
        stbi_image_free(rgb_image);
    }

    if(NULL != output_image)
    {
        free(output_image);
    }

    return 0;
}
