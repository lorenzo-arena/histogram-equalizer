project('histogram-equalizer-openmp', 'c',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('../common')

sources = ['main.c',
           '../common/hsl.c',
           '../common/log.c',
           '../common/stopwatch.c',
           '../common/equalizer.c',
           '../common/arguments.c']

# add_global_arguments('-DTRACE=1', language : 'c')

cc = meson.get_compiler('c')
libgomp = cc.find_library('gomp')

executable('histogram-equalizer-openmp',
           [sources],
           install : true,
           include_directories : incdir,
           c_args: '-fopenmp',
           dependencies : libgomp,
           link_args : '-lm')
