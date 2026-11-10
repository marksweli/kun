#!/bin/bash
# Four initial temperatures, two at a time (2-core machine).
PY=${PY:-/opt/fenicsx/bin/python}
export OMP_NUM_THREADS=1
mkdir -p results
run() { $PY thermal_shock.py --theta0 $1 --outdir results/theta0_$1C > results/log_theta0_$1C.txt 2>&1; }
( run 600; run 400 ) &
( run 300; run 500 ) &
wait
