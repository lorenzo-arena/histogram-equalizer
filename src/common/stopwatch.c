#include "stopwatch.h"

#define NS_PER_SECOND 1000000000

void stopwatch_start(stopwatch_t *stopwatch)
{
    stopwatch->start.tv_sec = 0;
    stopwatch->start.tv_nsec = 0;
    stopwatch->end.tv_sec = 0;
    stopwatch->end.tv_nsec = 0;
    stopwatch->delta.tv_sec = 0;
    stopwatch->delta.tv_nsec = 0;

    clock_gettime(CLOCK_MONOTONIC, &(stopwatch->start));
}

void stopwatch_stop(stopwatch_t *stopwatch)
{
    struct timespec *start = &(stopwatch->start);
    struct timespec *end = &(stopwatch->end);
    struct timespec *delta = &(stopwatch->delta);

    clock_gettime(CLOCK_MONOTONIC, end);

    delta->tv_nsec = end->tv_nsec - start->tv_nsec;
    delta->tv_sec  = end->tv_sec - start->tv_sec;
    if (delta->tv_sec > 0 && delta->tv_nsec < 0)
    {
        delta->tv_nsec += NS_PER_SECOND;
        delta->tv_sec--;
    }
    else if (delta->tv_sec < 0 && delta->tv_nsec > 0)
    {
        delta->tv_nsec -= NS_PER_SECOND;
        delta->tv_sec++;
    }
}

struct timespec stopwatch_get_elapsed(stopwatch_t *stopwatch)
{
    return stopwatch->delta;
}
