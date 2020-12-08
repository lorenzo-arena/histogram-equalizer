#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "hsl.h"
#include "log.h"
#include "stopwatch.h"

void histogram_calc(int *hyst, float *lum, size_t img_size)
{
    assert(hyst != NULL);
    assert(lum != NULL);


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
            unsigned char* pixel_offset = rgb_image + ((x + (height * y)) * STBI_rgb);
            
            rgb_pixel_t rgb_pixel = {
                .r = pixel_offset[0],
                .g = pixel_offset[1],
                .b = pixel_offset[2]
            };

            hsl_pixel_t hsl_pixel = { .h = 0, .s = 0, .l = 0 };

            rgb_to_hsl(rgb_pixel, &hsl_pixel);

            hsl_image.h[x + (height * y)] = hsl_pixel.h;
            hsl_image.s[x + (height * y)] = hsl_pixel.s;
            hsl_image.l[x + (height * y)] = hsl_pixel.l;
        }
    }

    // Calculate the histogram

    stbi_image_free(rgb_image);

    stopwatch_stop();

    struct timespec elapsed = stopwatch_get_elapsed();

    log_info("Elapsed time: %ld.%09ld",
        elapsed.tv_sec,
        elapsed.tv_nsec);

    return 0;
}