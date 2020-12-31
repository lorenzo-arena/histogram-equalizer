#ifndef __EQUALIZER_H__
#define __EQUALIZER_H__

extern "C" {
    #include <stdint.h>
}

int equalize(uint8_t *input, unsigned int width, unsigned int height, uint8_t **output);

#endif
