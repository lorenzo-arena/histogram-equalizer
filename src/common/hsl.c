#include "hsl.h"
#include "utils.h"

#include <math.h>

#define LUMINANCE_THRESHOLD (0.5)

#ifdef __NVCC__
__device__ void rgb_to_hsl(rgb_pixel_t rgb, hsl_pixel_t* hsl)
#else
void rgb_to_hsl(rgb_pixel_t rgb, hsl_pixel_t* hsl)
#endif
{
    float scaled_r = (float)(rgb.r) / 255;
    float scaled_g = (float)(rgb.g) / 255;
    float scaled_b = (float)(rgb.b) / 255;

    // 1. Calculate the min and max and mean to get the luminance
    float min = min_f(min_f(scaled_r, scaled_g), scaled_b);
    float max = max_f(max_f(scaled_r, scaled_g), scaled_b);

    float chroma = max - min;

    hsl->l = (min + max) / 2;

    // 2. If min and max are equal, we have a shade of gray and hue and saturation are 0;
    // otherwise they must be calculated
    if(min == max)
    {
        hsl->s = 0;
        hsl->h = 0;
    }
    else
    {
        if(max == scaled_r)
        {
            hsl->h = 60 * ((scaled_g - scaled_b) / chroma);
        }
        else if(max == scaled_g)
        {
            hsl->h = 60 * (2 + ((scaled_b - scaled_r) / chroma));
        }
        else
        {
            hsl->h = 60 * (4 + ((scaled_r - scaled_g) / chroma));
        }

        if(hsl->h < 0)
        {
            hsl->h += 360;
        }
        else if(hsl->h > 360)
        {
            hsl->h -= 360;
        }

#ifdef __NVCC__
        hsl->s = chroma / (1 - fabs((2 * max) - chroma -1));
#else
        hsl->s = chroma / (1 - fabsf((2 * max) - chroma -1));
#endif
    }
}

#ifdef __NVCC__
__device__ int rgb_to_h(rgb_pixel_t rgb)
#else
int rgb_to_h(rgb_pixel_t rgb)
#endif
{
    float scaled_r = (float)(rgb.r) / 255;
    float scaled_g = (float)(rgb.g) / 255;
    float scaled_b = (float)(rgb.b) / 255;

    // 1. Calculate the min and max and mean to get the luminance
    float min = min_f(min_f(scaled_r, scaled_g), scaled_b);
    float max = max_f(max_f(scaled_r, scaled_g), scaled_b);

    float chroma = max - min;

    // 2. If min and max are equal, we have a shade of gray and hue and saturation are 0;
    // otherwise they must be calculated
    if(min == max)
    {
        return 0;
    }
    else
    {
        float hue = 0.0;
        if(max == scaled_r)
        {
            hue = 60 * ((scaled_g - scaled_b) / chroma);
        }
        else if(max == scaled_g)
        {
            hue = 60 * (2 + ((scaled_b - scaled_r) / chroma));
        }
        else
        {
            hue = 60 * (4 + ((scaled_r - scaled_g) / chroma));
        }

        if(hue < 0)
        {
            hue += 360;
        }
        else if(hue > 360)
        {
            hue -= 360;
        }

        return (int)hue;
    }
}

#ifdef __NVCC__
__device__ float rgb_to_s(rgb_pixel_t rgb)
#else
float rgb_to_s(rgb_pixel_t rgb)
#endif
{
    float scaled_r = (float)(rgb.r) / 255;
    float scaled_g = (float)(rgb.g) / 255;
    float scaled_b = (float)(rgb.b) / 255;

    // 1. Calculate the min and max and mean to get the luminance
    float min = min_f(min_f(scaled_r, scaled_g), scaled_b);
    float max = max_f(max_f(scaled_r, scaled_g), scaled_b);

    float chroma = max - min;

    // 2. If min and max are equal, we have a shade of gray and hue and saturation are 0;
    // otherwise they must be calculated
    if(min == max)
    {
        return 0;
    }
    else
    {
        return chroma / (1 - fabs((2 * max) - chroma -1));
    }
}

#ifdef __NVCC__
__device__ float rgb_to_l(rgb_pixel_t rgb)
#else
float rgb_to_l(rgb_pixel_t rgb)
#endif
{
    float scaled_r = (float)(rgb.r) / 255;
    float scaled_g = (float)(rgb.g) / 255;
    float scaled_b = (float)(rgb.b) / 255;

    // 1. Calculate the min and max and mean to get the luminance
    float min = min_f(min_f(scaled_r, scaled_g), scaled_b);
    float max = max_f(max_f(scaled_r, scaled_g), scaled_b);

    return (min + max) / 2;
}

#ifdef __NVCC__
__device__ void hsl_to_rgb(hsl_pixel_t hsl, rgb_pixel_t *rgb)
#else
void hsl_to_rgb(hsl_pixel_t hsl, rgb_pixel_t *rgb)
#endif
{
    // 1. If saturation is zero we have a shade of gray
    if(hsl.s == 0)
    {
        rgb->r = rgb->g = rgb->b = (hsl.l * 255);
    }
    else
    {
#ifdef __NVCC__
        float chroma = (1 - fabsf((2 * hsl.l) - 1)) * hsl.s;
#else
        float chroma = (1 - fabsf((2 * hsl.l) - 1)) * hsl.s;
#endif

        double hue_iptr = 0.0;
        float hue = (modf((double)(hsl.h) / 360, &hue_iptr) * 360) / 60;

#ifdef __NVCC__
        float x = chroma * (1 - fabsf(fmodf(hue, 2.0) - 1));
#else
        float x = chroma * (1 - fabsf(fmodf(hue, 2.0) - 1));
#endif

        float red = 0.0;
        float green = 0.0;
        float blue = 0.0;

        if((0 <= hue) && (hue <= 1))
        {
            red = chroma;
            green = x;
            blue = 0;
        }
        else if((1 <= hue) && (hue <= 2))
        {
            red = x;
            green = chroma;
            blue = 0;
        }
        else if((2 <= hue) && (hue <= 3))
        {
            red = 0;
            green = chroma;
            blue = x;
        }
        else if((3 <= hue) && (hue <= 4))
        {
            red = 0;
            green = x;
            blue = chroma;
        }
        else if((4 <= hue) && (hue <= 5))
        {
            red = x;
            green = 0;
            blue = chroma;
        }
        else if((5 <= hue) && (hue <= 6))
        {
            red = chroma;
            green = 0;
            blue = x;
        }

        float m = hsl.l - (chroma / 2);

        rgb->r = round((red + m) * 255);
        rgb->g = round((green + m) * 255);
        rgb->b = round((blue + m) * 255);
    }
}
