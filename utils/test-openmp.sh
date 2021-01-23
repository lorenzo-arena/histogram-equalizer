#!/bin/bash

echo "Testing sequential solution.."
SEQ_TIME=$(utils/runner.sh -r 30 -s)

for i in $(seq 1 32);
do
	echo "Testing OpenMP solution with $i threads.."
	printf "$i;" >> test-omp.csv
	OMP_TIME=$(utils/runner.sh -r 30 -o -t $i)
	SPEEDUP=$(python -c "print $SEQ_TIME / $OMP_TIME")
	printf "$SPEEDUP;\n" >> test-omp.csv
done

