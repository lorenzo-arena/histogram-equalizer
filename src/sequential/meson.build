project('hystogram-equalizer-sequential', 'c',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('../common')

sources = ['main.c']

# add_global_arguments('-DTRACE=1', language : 'c')

executable('hystogram-equalizer-sequential',
           [sources],
           install : true,
           include_directories : incdir,
           link_args : '-lm')
