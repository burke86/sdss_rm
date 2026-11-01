#!/bin/sh
# Run rm_lag_profile.py over a job list, N at a time; finished fits are skipped, so it can be restarted.
#
# usage: sh run_lag_profiles.sh <jobs.txt> <out_dir> [n_parallel=3] [extra flags for rm_lag_profile.py ...]
# e.g.   sh run_lag_profiles.sh jobs.txt lag_runs/run1 3
#        sh run_lag_profiles.sh jobs.txt lag_runs/run1_noslow 3 --no-slow
set -e
JOBS=$1; OUT=$2; NPAR=${3:-3}
shift 2; [ $# -gt 0 ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/out" "$OUT/logs"
echo "start $(date '+%I:%M %p'): $(wc -l < "$JOBS") jobs, $NPAR at a time, flags: $*"
export HERE OUT FLAGS="$*"
xargs -P "$NPAR" -L 1 sh -c '
  rmid=$0; case=$1
  log="$OUT/logs/${rmid}_${case}.log"
  if ls "$OUT"/out/rm$(printf %03d "$rmid")_"$case"_*.json >/dev/null 2>&1; then exit 0; fi
  python "$HERE/rm_lag_profile.py" "$rmid" "$case" "$OUT/out" $FLAGS > "$log" 2>&1 \
    && grep -h DONE "$log" || echo "FAIL $rmid $case (see $log)"
' < "$JOBS"
echo "ALL DONE $(date '+%I:%M %p'): $(ls "$OUT"/out/*.json 2>/dev/null | wc -l) results in $OUT/out"
