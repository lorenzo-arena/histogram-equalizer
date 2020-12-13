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
};

extern struct arguments arguments;

extern const char *argp_program_version;

extern const char doc[];

void set_default_arguments(struct arguments *arguments);
error_t parse_opt(int key, char *arg, struct argp_state *state);

static char args_doc[] = "image output";

static struct argp_option options[] =
{
    {"stopwatch", 's', 0, 0, "Enable stopwatch usage", 0},
    {"plot", 'p', 0, 0, "Enable histogram plot", 0},
    {"log_histogram", 'l', 0, 0, "Enable histogram log", 0},
    {0}
};

static struct argp argp = {options, parse_opt, args_doc, doc, NULL, NULL, NULL};

#endif
