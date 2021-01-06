#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <time.h>

typedef struct {
    struct timespec start;
    struct timespec end;
    struct timespec delta;
} stopwatch_t;

void stopwatch_start(stopwatch_t *stopwatch);
void stopwatch_stop(stopwatch_t *stopwatch);
struct timespec stopwatch_get_elapsed(stopwatch_t *stopwatch);

#endif
