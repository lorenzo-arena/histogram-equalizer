#!/bin/bash

declare -a TIMES
SUM="0"
REPETITIONS=0

RUN_SEQUENTIAL=0
RUN_OPENMP=0
RUN_CUDA=0

SEQUENTIAL_PATH="src/sequential"
OPENMP_PATH="src/parallel-openmp"
CUDA_PATH="src/parallel-cuda"

SEQUENTIAL_BIN="histogram-equalizer-sequential"
OPENMP_BIN="histogram-equalizer-openmp"
CUDA_BIN="histogram-equalizer-cuda"

INPUT_PATH="../../../assets/pic_low_contrast.jpg"
OUTPUT_PATH="output.jpg"

VERBOSE=0

CHECK_TOTAL=0
CHECK_STRING="Elapsed time:"

BIG_IMAGE=0

THREADS=0
USE_THREADS=0

usage () {
    echo "Usage:"
    echo "utils/runner.sh -r|--repetitions [repetitions] (-s|--sequential)|(-o|--openmp)|(-c|--cuda) [-v|--verbose] [-l|--total] [-b|--big] [-t|--threads]"
    exit 1
}

check_exit_code () {
    if [ $? -ne 0 ]; then
        echo $1
        exit 1
    fi
}

while [ "$#" -gt 0 ]
do
    case $1 in
        -r | --repetitions )
            REPETITIONS=$2
            shift
        ;;
        -s | --sequential )
            RUN_SEQUENTIAL=1
        ;;
        -o | --openmp )
            RUN_OPENMP=1
        ;;
        -c | --cuda )
            RUN_CUDA=1
        ;;
        -v | --verbose )
            VERBOSE=1
        ;;
	-l | --total )
            CHECK_TOTAL=1
        ;;
        -b | --big)
            BIG_IMAGE=1
        ;;
        -t | --threads)
            USE_THREADS=1
            THREADS=$2
            shift
        ;;
        *)
            echo "Unknown parameter passed: $1, discarding it"
        ;;
    esac
    shift
done

if [ $REPETITIONS -eq 0 ]; then
    usage
fi

if [ $CHECK_TOTAL -eq 1 ]; then
	CHECK_STRING="Total elapsed time:"
fi

if [ $BIG_IMAGE -eq 1 ]; then
        INPUT_PATH="../../../assets/pic_low_contrast_big.jpg"
fi


if [ $RUN_SEQUENTIAL -eq 1 ]; then
    cd $SEQUENTIAL_PATH
    meson builddir > /dev/null

    cd builddir/
    ninja > /dev/null
    check_exit_code "Sequential project build failed!"

    for i in $(seq 1 $REPETITIONS);
    do
        if [ $VERBOSE -eq 1 ]; then
            echo "Running iteration $i.."
        fi

	TIME=$(./$SEQUENTIAL_BIN $INPUT_PATH $OUTPUT_PATH -s | grep "$CHECK_STRING" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
	check_exit_code "Sequential project run failed!"
        TIMES+=($TIME)
    done

    for i in $(seq 1 $REPETITIONS);
    do
        SUM=$(python -c "print $SUM+${TIMES[$(python -c "print $i - 1")]}")
    done

    MEAN=$(python -c "print $SUM / $REPETITIONS")

    echo "$MEAN"
elif [ $RUN_OPENMP -eq 1 ]; then
    cd $OPENMP_PATH
    meson builddir > /dev/null

    cd builddir/
    ninja > /dev/null
    check_exit_code "OpenMP project build failed!"

    for i in $(seq 1 $REPETITIONS);
    do
        if [ $VERBOSE -eq 1 ]; then
            echo "Running iteration $i.."
        fi

        THREADS_OPT=""

        if [ $USE_THREADS -eq 1 ]; then
            THREADS_OPT="-t $THREADS"
        fi

        TIME=$(./$OPENMP_BIN $INPUT_PATH $OUTPUT_PATH -s $THREADS_OPT | grep "$CHECK_STRING" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
        check_exit_code "OpenMP project run failed!"
        TIMES+=($TIME)
    done

    for i in $(seq 1 $REPETITIONS);
    do
        SUM=$(python -c "print $SUM+${TIMES[$(python -c "print $i - 1")]}")
    done

    MEAN=$(python -c "print $SUM / $REPETITIONS")

    echo "$MEAN"
elif [ $RUN_CUDA -eq 1 ]; then
    cd $CUDA_PATH
    meson builddir > /dev/null

    cd builddir/
    ninja > /dev/null
    check_exit_code "CUDA project build failed!"

    for i in $(seq 1 $REPETITIONS);
    do
        if [ $VERBOSE -eq 1 ]; then
            echo "Running iteration $i.."
        fi

        TIME=$(./$CUDA_BIN $INPUT_PATH $OUTPUT_PATH -s | grep "$CHECK_STRING" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
        check_exit_code "CUDA project run failed!"
        TIMES+=($TIME)
    done

    for i in $(seq 1 $REPETITIONS);
    do
        SUM=$(python -c "print $SUM+${TIMES[$(python -c "print $i - 1")]}")
    done

    MEAN=$(python -c "print $SUM / $REPETITIONS")

    echo "$MEAN"
else
    usage
fi






