#!/bin/bash
# 3D TBC run matrix (Sec. 3.4.4): cases 1/2/3 x w = 50/100 um, Gi = 0.001 N/mm, hTC = 100 um,
# half model, h = 10 um, T = 10 ms. Two runs in parallel on a 2-core machine.
set -e
cd "$(dirname "$0")"
PY=${PY:-/opt/fenicsx/bin/python}
export OMP_NUM_THREADS=1
T=${T:-0.01}
run() { c=$1; w=$2; name=case${c}_w${w}_Gi0.001_hTC100; [ -f results/$name/run_info.json ] && return 0
        $PY tbc_3d.py --case $c --w $w --T $T --outdir results/$name > results/log_$name.txt 2>&1; }
mkdir -p results
run 3 100 & run 3 50 ; wait
run 1 100 & run 1 50 ; wait
run 2 100 & run 2 50 ; wait
