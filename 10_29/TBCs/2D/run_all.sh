#!/bin/bash
# 2D TBC run matrix (Sec. 3.4.1-3.4.3). Two runs in parallel on a 2-core machine.
set -e
cd "$(dirname "$0")"
PY=${PY:-/opt/fenicsx/bin/python}
export OMP_NUM_THREADS=1
run() { name=$1; shift; [ -f results/$name/run_info.json ] && return 0
        $PY tbc_2d.py "$@" --outdir results/$name > results/log_$name.txt 2>&1; }
mkdir -p results
run quench_Gi0.001_hTC100 --mode quench --Gi 0.001 &
run heat_Gi0.001_hTC100   --mode heat   --Gi 0.001 ; wait
run quench_Gi0.002_hTC100 --mode quench --Gi 0.002 &
run quench_Gi0.004_hTC100 --mode quench --Gi 0.004 ; wait
run quench_Gi0.008_hTC100 --mode quench --Gi 0.008 &
run quench_Gi0.001_hTC300 --mode quench --hTC 300 ; wait
run quench_Gi0.001_hTC500 --mode quench --hTC 500
# roller-bottom comparison (results/roller/)
# $PY tbc_2d.py --mode quench --bottom roller --outdir results/roller/quench_Gi0.001_hTC100
