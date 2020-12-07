project('hystogram-equalizer-test', 'c',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('../common', 'Unity/src')

sources = ['test.c',
           'Unity/src/unity.c',
           '../common/hsl.c']

# add_global_arguments('-DTRACE=1', language : 'c')

executable('hystogram-equalizer-test',
           [sources],
           install : true,
           include_directories : incdir,
           link_args : '-lm')
