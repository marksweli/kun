#!/bin/bash
# Queue A: fine mesh (long); Queue B: stress levels + coarser mesh study. Run with 2 cores.
PY=${PY:-/opt/fenicsx/bin/python}
( $PY crack_branching.py --h 0.125 --dt 10 --outdir results/mesh_h0.125 > results/log_mesh_h0.125.txt 2>&1 ) &
(
for s in 2 4 6 12; do
  $PY crack_branching.py --sigma $s --h 0.25 --dt 20 --outdir results/sigma${s}MPa > results/log_sigma${s}MPa.txt 2>&1
done
for hh in 0.5 0.25; do
  $PY crack_branching.py --h $hh --dt 10 --outdir results/mesh_h$hh > results/log_mesh_h$hh.txt 2>&1
done
) &
wait
