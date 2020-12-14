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

struct arguments arguments;

const char *argp_program_version =
" 1.0";

const char doc[] =
"histogram-equalizer-sequential -- Used to equalize the histogram of an image.";

int main(int argc, char **argv)
{
    int width, height, bpp;

    set_default_arguments(&arguments);

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if(arguments.stopwatch)
    {
        stopwatch_start();
    }

    uint8_t* rgb_image = stbi_load(arguments.args[0], &width, &height, &bpp, STBI_rgb);

    // Image BPP will be 4 but the reading is forced to be RGB only
    log_info("BPP %d", bpp);
    log_info("Width %d", width);
    log_info("Height %d", height);

    uint8_t *output_image = NULL;

    equalize(rgb_image, width, height, &output_image);

    if(NULL == output_image)
    {
        log_error("Error while equalizing image!");

        // Clean up buffers
        stbi_image_free(rgb_image);
        free(output_image);

        return 1;
    }

    stbi_write_jpg(arguments.args[1], width, height, STBI_rgb, output_image, 100);

    // Clean up buffers
    stbi_image_free(rgb_image);
    free(output_image);

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
