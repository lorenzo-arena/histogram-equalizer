project('histogram-equalizer-cuda', 'c',
  'cuda',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('../common')

sources = ['main.cu',
           'equalizer.cu',
           '../common/hsl.c',
           '../common/log.c',
           '../common/stopwatch.c',
           '../common/arguments.c']

# add_global_arguments('-DTRACE=1', language : 'c')

executable('histogram-equalizer-cuda',
           [sources],
           install : true,
           include_directories : incdir,
           link_args : '-lm')
