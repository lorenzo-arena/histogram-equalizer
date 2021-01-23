#include "arguments.h"
#include "log.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <sys/sysinfo.h>
#endif

static char args_doc[] = "image output";

static struct argp_option options[] =
{
    {"stopwatch", 's', 0, 0, "Enable stopwatch usage", 0},
    {"plot", 'p', 0, 0, "Enable histogram plot", 0},
    {"log_histogram", 'l', 0, 0, "Enable histogram log", 0},
#ifdef _OPENMP
    {"threads", 't', "", 0, "Number of threads to use, from 1", 0},
#endif
    {0}
};

struct argp argp = {options, parse_opt, args_doc, doc, NULL, NULL, NULL};

void set_default_arguments(struct arguments *arguments)
{
    arguments->args[0] = "";
    arguments->args[1] = "";
    arguments->stopwatch = false;
    arguments->plot = false;
    arguments->log_histogram = false;
#ifdef _OPENMP
    arguments->threads = get_nprocs_conf();
#endif
}

error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
        case 's':
        {
            arguments->stopwatch = true;
            break;
        }
        case 'p':
        {
            arguments->plot = true;
            break;
        }
        case 'l':
        {
            arguments->log_histogram = true;
            break;
        }
#ifdef _OPENMP
        case 't':
        {
            char *next;
            arguments->threads = strtol(arg, &next, 10);

            if(*next != '\0')
            {
                log_error("Invalid threads number %s", arg);
                argp_usage(state);
            }
            else if(arguments->threads < 1)
            {
                log_error("Invalid threads number: %d", arguments->threads);
                argp_usage(state);
            }
            break;
        }
#endif
        case ARGP_KEY_ARG:
        {
            if (state->arg_num >= 2)
            {
                argp_usage(state);
            }
            arguments->args[state->arg_num] = arg;
            break;
        }
        case ARGP_KEY_END:
        {
            if (state->arg_num < 2)
            {
                argp_usage(state);
            }
            break;
        }
        default:
        return ARGP_ERR_UNKNOWN;
    }

    return 0;
}