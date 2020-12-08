## Histogram equalizer project

This project was made for the "parallel computing" course at the Computer Engineering University of Florence.

### Goal

The main goal is to write two version of the same program which must compute the histogram of an image an perform an histogram equalization.

### Requirements

This project can be built using [meson](https://mesonbuild.com/) and ninja; the development was made under Ubuntu 20.10.

To execute a build you need to navigate to the folder "sequential" or "parallel" and use the following commands:

```bash
$ meson builddir
$ cd builddir
$ ninja
```

The project uses [STB Single file libraries](https://github.com/nothings/stb) for reading and writing images.

GNUPlot is necessary to plot the histogram.