#ifndef __HSL_H__
#define __HSL_H__

typedef struct {
    int *h;
    float *s;
    float *l;
} hsl_image;

typedef struct {
    int h;
    float s;
    float l;
} hsl_pixel;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char dummy;
} rgb_pixel;

void rgb_to_hsl(rgb_pixel rgb, hsl_pixel* hsl);

int rgb_to_h(rgb_pixel rgb);

float rgb_to_s(rgb_pixel rgb);

float rgb_to_l(rgb_pixel rgb);

void hsl_to_rgb(hsl_pixel hsl, rgb_pixel *rgb);

#endif
