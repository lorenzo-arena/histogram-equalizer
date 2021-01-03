#!/bin/sh

OUTPUT_FILE="output.jpg"
INPUT_FILEPATH="../../../assets/pic_low_contrast.jpg"
SEQUENTIAL_PATH="src/sequential"
OPENMP_PATH="src/parallel-openmp"

RUN_CUDA=0
CUDA_PATH="src/parallel-cuda"

while [ "$#" -gt 0 ]
do
    case $1 in
        --cuda )
            RUN_CUDA=1
        ;;
	*)
            echo "Unknown parameter passed: $1, discarding it"
        ;;
    esac
    shift
done

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
ninja > /dev/null
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
ninja > /dev/null
check_exit_code "OpenMP project build failed!"

echo "Running OpenMP project.."
./histogram-equalizer-openmp $INPUT_FILEPATH $OUTPUT_FILE
check_exit_code "OpenMP project run failed!"

# Go back to root folder
cd ../../../

check_file_differ $SEQUENTIAL_PATH/builddir/$OUTPUT_FILE $OPENMP_PATH/builddir/$OUTPUT_FILE

if [ $RUN_CUDA -eq 1 ]; then
	echo "Building CUDA project.."
	cd $CUDA_PATH
	meson builddir > /dev/null

	cd builddir/
	ninja > /dev/null
	check_exit_code "CUDA project build failed!"

	echo "Running CUDA project.."
	./histogram-equalizer-cuda $INPUT_FILEPATH $OUTPUT_FILE
	check_exit_code "CUDA project run failed!"

	# Go back to root folder
	cd ../../../

	check_file_differ $SEQUENTIAL_PATH/builddir/$OUTPUT_FILE $CUDA_PATH/builddir/$OUTPUT_FILE
fi

