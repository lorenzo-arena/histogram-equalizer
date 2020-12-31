project('histogram-equalizer-sequential', 'c',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('../common')

sources = ['main.c',
           '../common/hsl.c',
           '../common/log.c',
           '../common/stopwatch.c',
           '../common/equalizer.c',
           '../common/arguments.c',
           '../common/cexception/lib/CException.c']

# add_global_arguments('-DTRACE=1', language : 'c')

executable('histogram-equalizer-sequential',
           [sources],
           install : true,
           include_directories : incdir,
           link_args : '-lm')
