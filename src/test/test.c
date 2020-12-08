#include "Unity/src/unity.h"

// Files to test
#include "test_hsl.h"

void setUp(void) {
    // Needed for UNITY_BEGIN
}

void tearDown(void) {
    // Needed for UNITY_END
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_rgb_to_hsl);
    RUN_TEST(test_hsl_to_rgb);
    RUN_TEST(test_rgb_to_h);
    return UNITY_END();
}