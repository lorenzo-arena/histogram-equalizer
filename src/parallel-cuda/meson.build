project('histogram-equalizer-cuda', 'cuda',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('../common')

sources = ['main.cu',
           'equalizer.cu',
           '../common/hsl.cu',
           '../common/log.c',
           '../common/stopwatch.c',
           '../common/arguments.c',
           '../common/cexception/lib/CException.c']

# add_global_arguments('-DTRACE=1', language : 'c')

# To use decoupled __device__ functions
add_global_arguments('-dc', language : 'cuda')

executable('histogram-equalizer-cuda',
           [sources],
           install : true,
           include_directories : incdir,
           link_args : '-lm')
