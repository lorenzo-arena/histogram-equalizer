#!/bin/sh

OUTPUT_FILE="output.jpg"
INPUT_FILEPATH="../../../assets/pic_low_contrast.jpg"
SEQUENTIAL_PATH="src/sequential"
OPENMP_PATH="src/parallel-openmp"

check_exit_code () {
    if [ $? -ne 0 ]; then
        echo $1
        exit 1
    fi
}

check_file_differ () {
    DIFF_OUTPUT=$(diff $1 $2)
    if [ -z "$DIFF_OUTPUT" ]; then
        echo "Files $1 $2 are equal!"
    else
        echo "Files $1 $2 are different!"
        exit 1
    fi
}

echo "Building sequential project.."
cd $SEQUENTIAL_PATH
meson builddir > /dev/null

cd builddir/
ninja
check_exit_code "Sequential project build failed!"

echo "Running sequential project.."
./histogram-equalizer-sequential $INPUT_FILEPATH $OUTPUT_FILE
check_exit_code "Sequential project run failed!"

# Go back to root folder
cd ../../../

echo "Building OpenMP project.."
cd $OPENMP_PATH
meson builddir > /dev/null

cd builddir/
ninja
check_exit_code "OpenMP project build failed!"

echo "Running OpenMP project.."
./histogram-equalizer-openmp $INPUT_FILEPATH $OUTPUT_FILE
check_exit_code "OpenMP project run failed!"

# Go back to root folder
cd ../../../

check_file_differ $SEQUENTIAL_PATH/builddir/$OUTPUT_FILE $OPENMP_PATH/builddir/$OUTPUT_FILE



