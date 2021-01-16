#ifndef __EQUALIZER_H__
#define __EQUALIZER_H__

extern "C" {
    #include <stdint.h>
}

#include "hsl.cuh"

int equalize(rgb_pixel_t *input, unsigned int width, unsigned int height, uint8_t **output);

#endif
