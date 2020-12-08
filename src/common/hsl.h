#ifndef __HSL_H__
#define __HSL_H__

typedef struct {
    int *h;
    float *s;
    float *l;
} hsl_image_t;

typedef struct {
    int h;
    float s;
    float l;
} hsl_pixel_t;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char dummy;
} rgb_pixel_t;

void rgb_to_hsl(rgb_pixel_t rgb, hsl_pixel_t* hsl);

int rgb_to_h(rgb_pixel_t rgb);

float rgb_to_s(rgb_pixel_t rgb);

float rgb_to_l(rgb_pixel_t rgb);

void hsl_to_rgb(hsl_pixel_t hsl, rgb_pixel_t *rgb);

#endif
