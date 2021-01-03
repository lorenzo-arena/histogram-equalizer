#ifndef __ARGUMENTS_H__
#define __ARGUMENTS_H__

#include <argp.h>
#include <stdbool.h>

struct arguments
{
    char *args[2];                /* image and output */
    bool stopwatch;
    bool plot;
    bool log_histogram;
#ifdef _OPENMP
    int threads;
#endif
};

extern struct arguments arguments;

extern const char *argp_program_version;

extern const char doc[];

extern struct argp argp;

void set_default_arguments(struct arguments *arguments);
error_t parse_opt(int key, char *arg, struct argp_state *state);

#endif
