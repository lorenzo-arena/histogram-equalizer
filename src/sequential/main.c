#include <stdint.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "hsl.h"

int main(int argc, char **argv) {
    int width, height, bpp;

    if(argc <= 1)
    {
        printf("Usage: %s <image-path>\n", argv[0]);
        return 1;
    }

    uint8_t* rgb_image = stbi_load(argv[1], &width, &height, &bpp, STBI_rgb);

    printf("Width %d\n", width);
    printf("Height %d\n", height);

    // Image BPP will be 4 but the reading is forced to be RGB only
    printf("BPP %d\n", bpp);

    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            unsigned char* pixel_offset = rgb_image + (x + height * y) * STBI_rgb;
            printf("Pixel %d:%d: %d %d %d\n", x, y, pixel_offset[0], pixel_offset[1], pixel_offset[2]);
        }
    }

    stbi_image_free(rgb_image);

    return 0;
}