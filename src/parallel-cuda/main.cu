#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
    
    #include "log.h"
    #include "stopwatch.h"
    #include "arguments.h"
}

#include "equalizer.cuh"

struct arguments arguments;

const char *argp_program_version =
" 1.0";

const char doc[] =
"histogram-equalizer-cudaa -- Used to equalize the histogram of an image.";

int main(int argc, char **argv)
{
    int width, height, bpp;

    uint8_t *rgb_image = NULL;
    uint8_t *output_image = NULL;

    set_default_arguments(&arguments);

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    rgb_image = stbi_load(arguments.args[0], &width, &height, &bpp, STBI_rgb);

    assert(rgb_image != NULL);

    // Image BPP will be 4 but the reading is forced to be RGB only
    log_info("BPP %d", bpp);
    log_info("Width %d", width);
    log_info("Height %d", height);

    if(arguments.stopwatch)
    {
        stopwatch_start();
    }

    equalize(rgb_image, width, height, &output_image);

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

    // Clean up buffers
    if(rgb_image != NULL)
    {
        stbi_image_free(rgb_image);
    }

    if(output_image != NULL)
    {
        free(output_image);
    }

    return 0;
}
