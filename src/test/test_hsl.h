#ifndef __TEST_HSL_H__
#define __TEST_HSL_H__

#include "Unity/src/unity.h"

#include "../common/hsl.h"
#include <string.h>

void test_rgb_to_hsl(void)
{
    TEST_MESSAGE("Starting RGB to HSL test..");

    // Test for red
    hsl_pixel hsl;
    rgb_pixel rgb_red = {
        .r = 255,
        .g = 0,
        .b = 0
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_red, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for red..");
    TEST_ASSERT_EQUAL_INT32(0, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(1.0, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.5,hsl.l);

    rgb_pixel rgb_green = {
        .r = 0,
        .g = 255,
        .b = 0
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_green, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for green..");
    TEST_ASSERT_EQUAL_INT32(120, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(1.0, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.5, hsl.l);

    rgb_pixel rgb_blue = {
        .r = 0,
        .g = 0,
        .b = 255
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_blue, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for blue..");
    TEST_ASSERT_EQUAL_INT32(240, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(1.0, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.5, hsl.l);

    rgb_pixel rgb_aqua = {
        .r = 0,
        .g = 255,
        .b = 255
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_aqua, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for aqua..");
    TEST_ASSERT_EQUAL_INT32(180, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(1.0, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.5, hsl.l);

    rgb_pixel rgb_turquoise = {
        .r = 64,
        .g = 224,
        .b = 208
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_turquoise, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for turquoise..");
    TEST_ASSERT_EQUAL_INT32(174, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(0.7207207, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.5647059, hsl.l);

    rgb_pixel rgb_moccasin = {
        .r = 255,
        .g = 228,
        .b = 181
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_moccasin, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for moccasin..");
    TEST_ASSERT_EQUAL_INT32(38, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(1.0, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.854902, hsl.l);

    rgb_pixel rgb_blueviolet = {
        .r = 138,
        .g = 43,
        .b = 226
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_blueviolet, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for blueviolet..");
    TEST_ASSERT_EQUAL_INT32(271, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(0.7593361, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.527451, hsl.l);

    rgb_pixel rgb_mintcream = {
        .r = 245,
        .g = 255,
        .b = 250
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_mintcream, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for mintcream..");
    TEST_ASSERT_EQUAL_INT32(150, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(1.0, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.9803922, hsl.l);

    rgb_pixel rgb_darkslategray = {
        .r = 47,
        .g = 79,
        .b = 79
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_darkslategray, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for darkslategray..");
    TEST_ASSERT_EQUAL_INT32(180, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(0.2539682, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.2470588, hsl.l);

    rgb_pixel rgb_darkgray = {
        .r = 169,
        .g = 169,
        .b = 169
    };
    memset(&hsl, 0x00, sizeof(hsl_pixel));
    rgb_to_hsl(rgb_darkgray, &hsl);

    TEST_MESSAGE("Testing RGB to HSL for darkgray..");
    TEST_ASSERT_EQUAL_INT32(0, hsl.h);
    TEST_ASSERT_EQUAL_FLOAT(0.0, hsl.s);
    TEST_ASSERT_EQUAL_FLOAT(0.6627451, hsl.l);
}

void test_hsl_to_rgb(void)
{
    TEST_MESSAGE("Starting HSL to RGB test..");

    // Test for red
    rgb_pixel rgb;
    hsl_pixel hsl_red = {
        .h = 0,
        .s = 1.0,
        .l = 0.5
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_red, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for red..");
    TEST_ASSERT_EQUAL_UINT8(255, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(0, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(0, rgb.b);

    hsl_pixel hsl_green = {
        .h = 120,
        .s = 1.0,
        .l = 0.5
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_green, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for green..");
    TEST_ASSERT_EQUAL_UINT8(0, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(255, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(0, rgb.b);

    hsl_pixel hsl_blue = {
        .h = 240,
        .s = 1.0,
        .l = 0.5
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_blue, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for blue..");
    TEST_ASSERT_EQUAL_UINT8(0, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(0, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(255, rgb.b);

    hsl_pixel hsl_aqua = {
        .h = 180,
        .s = 1.0,
        .l = 0.5
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_aqua, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for aqua..");
    TEST_ASSERT_EQUAL_UINT8(0, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(255, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(255, rgb.b);

    hsl_pixel hsl_turquoise = {
        .h = 174,
        .s = 0.7207207,
        .l = 0.5647059
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_turquoise, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for turquoise..");
    TEST_ASSERT_EQUAL_UINT8(64, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(224, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(208, rgb.b);

    hsl_pixel hsl_moccasin = {
        .h = 38,
        .s = 1.0,
        .l = 0.854902
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_moccasin, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for moccasin..");
    TEST_ASSERT_EQUAL_UINT8(255, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(228, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(181, rgb.b);

    hsl_pixel hsl_blueviolet = {
        .h = 271,
        .s = 0.7593361,
        .l = 0.527451
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_blueviolet, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for blueviolet..");
    TEST_ASSERT_EQUAL_UINT8(138, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(43, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(226, rgb.b);

    hsl_pixel hsl_mintcream = {
        .h = 150,
        .s = 1.0,
        .l = 0.9803922
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_mintcream, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for mintcream..");
    TEST_ASSERT_EQUAL_UINT8(245, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(255, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(250, rgb.b);

    hsl_pixel hsl_darkslategray = {
        .h = 180,
        .s = 0.2539682,
        .l = 0.2470588
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_darkslategray, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for darkslategray..");
    TEST_ASSERT_EQUAL_UINT8(47, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(79, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(79, rgb.b);

    hsl_pixel hsl_darkgray = {
        .h = 0,
        .s = 0.0,
        .l = 0.6627451
    };
    memset(&rgb, 0x00, sizeof(rgb_pixel));
    hsl_to_rgb(hsl_darkgray, &rgb);

    TEST_MESSAGE("Testing HSL to RGB for darkgray..");
    TEST_ASSERT_EQUAL_UINT8(169, rgb.r);
    TEST_ASSERT_EQUAL_UINT8(169, rgb.g);
    TEST_ASSERT_EQUAL_UINT8(169, rgb.b);
}

void test_rgb_to_h(void)
{
    TEST_MESSAGE("Starting RGB to H test..");

    // Test for red
    int hue = 0;
    rgb_pixel rgb_red = {
        .r = 255,
        .g = 0,
        .b = 0
    };
    hue = 0;
    hue = rgb_to_h(rgb_red);

    TEST_MESSAGE("Testing RGB to H for red..");
    TEST_ASSERT_EQUAL_INT32(0, hue);

    rgb_pixel rgb_green = {
        .r = 0,
        .g = 255,
        .b = 0
    };
    hue = 0;
    hue = rgb_to_h(rgb_green);

    TEST_MESSAGE("Testing RGB to H for green..");
    TEST_ASSERT_EQUAL_INT32(120, hue);

    rgb_pixel rgb_blue = {
        .r = 0,
        .g = 0,
        .b = 255
    };
    hue = 0;
    hue = rgb_to_h(rgb_blue);

    TEST_MESSAGE("Testing RGB to H for blue..");
    TEST_ASSERT_EQUAL_INT32(240, hue);

    rgb_pixel rgb_aqua = {
        .r = 0,
        .g = 255,
        .b = 255
    };
    hue = 0;
    hue = rgb_to_h(rgb_aqua);

    TEST_MESSAGE("Testing RGB to H for aqua..");
    TEST_ASSERT_EQUAL_INT32(180, hue);

    rgb_pixel rgb_turquoise = {
        .r = 64,
        .g = 224,
        .b = 208
    };
    hue = 0;
    hue = rgb_to_h(rgb_turquoise);

    TEST_MESSAGE("Testing RGB to H for turquoise..");
    TEST_ASSERT_EQUAL_INT32(174, hue);

    rgb_pixel rgb_moccasin = {
        .r = 255,
        .g = 228,
        .b = 181
    };
    hue = 0;
    hue = rgb_to_h(rgb_moccasin);

    TEST_MESSAGE("Testing RGB to H for moccasin..");
    TEST_ASSERT_EQUAL_INT32(38, hue);

    rgb_pixel rgb_blueviolet = {
        .r = 138,
        .g = 43,
        .b = 226
    };
    hue = 0;
    hue = rgb_to_h(rgb_blueviolet);

    TEST_MESSAGE("Testing RGB to H for blueviolet..");
    TEST_ASSERT_EQUAL_INT32(271, hue);

    rgb_pixel rgb_mintcream = {
        .r = 245,
        .g = 255,
        .b = 250
    };
    hue = 0;
    hue = rgb_to_h(rgb_mintcream);

    TEST_MESSAGE("Testing RGB to H for mintcream..");
    TEST_ASSERT_EQUAL_INT32(150, hue);

    rgb_pixel rgb_darkslategray = {
        .r = 47,
        .g = 79,
        .b = 79
    };
    hue = 0;
    hue = rgb_to_h(rgb_darkslategray);

    TEST_MESSAGE("Testing RGB to H for darkslategray..");
    TEST_ASSERT_EQUAL_INT32(180, hue);

    rgb_pixel rgb_darkgray = {
        .r = 169,
        .g = 169,
        .b = 169
    };
    hue = 0;
    hue = rgb_to_h(rgb_darkgray);

    TEST_MESSAGE("Testing RGB to H for darkgray..");
    TEST_ASSERT_EQUAL_INT32(0, hue);
}

#endif