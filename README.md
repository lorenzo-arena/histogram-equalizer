## Histogram equalizer

This project was made for the "parallel computing" course at the Computer Engineering University of Florence.

### Goal

The main goal is to write two version of the same program which must compute the histogram of an image an perform an histogram equalization.

### Requirements

This project can be built using [meson](https://mesonbuild.com/) and ninja; the development was made under Ubuntu 20.10.

To build the program you need to navigate to the folder "*sequential*" or "*parallel*" and use the following commands:

```bash
$ meson builddir
$ cd builddir
$ ninja
```

The project uses [STB Single file libraries](https://github.com/nothings/stb) for reading and writing images.

You can run the program by specifying an image path:

```bash
# To run sequential version
$ cd src/sequential/builddir
$ ./histogram-equalizer-sequential <input_file_path> <output_file_path> [options]
```

### Options

For the sequential project, the following options are available:

- `-p` to plot the image histogram and the post-processed image histogram (requires [gnuplot](http://www.gnuplot.info/) to be installed on the system)
- `-s` to print the time elapsed for the execution of the program
- `-l` to log the histogram and cdf values on stdout

### Tests

Some tests are written using the [Unity](http://www.throwtheswitch.org/unity) C framework.

There are also some script under *utils* like *verifier.sh* which can be used to build multiple project versions, run them and compare the outputs.

### Result

The following images show the result after the equalization process:

| Before processing                                | After processing                                        |
| ------------------------------------------------ | ------------------------------------------------------- |
| ![low_contast_pic](./assets/pic_low_contrast.jpg) | ![better_contast_pic](./assets/pic_better_contrast.jpg) |

