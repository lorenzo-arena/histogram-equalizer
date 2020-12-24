#include "arguments.h"

static char args_doc[] = "image output";

static struct argp_option options[] =
{
    {"stopwatch", 's', 0, 0, "Enable stopwatch usage", 0},
    {"plot", 'p', 0, 0, "Enable histogram plot", 0},
    {"log_histogram", 'l', 0, 0, "Enable histogram log", 0},
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