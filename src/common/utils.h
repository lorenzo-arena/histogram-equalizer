#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>

unsigned char max_uc(unsigned char a, unsigned char b)
{
    return (a > b) ? a : b;
}

float max_f(float a, float b)
{
    return (a > b) ? a : b;
}

unsigned char min_uc(unsigned char a, unsigned char b)
{
    return (a < b) ? a : b;
}

float min_f(float a, float b)
{
    return (a < b) ? a : b;
}

#endif
