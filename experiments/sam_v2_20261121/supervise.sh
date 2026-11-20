#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
host=$(hostname -s)
case "$host" in
 morgoth) python=/opt/venv/jdusza_itr_prompt_v2/bin/python; mode=master;;
 glaurung) python=/data/dream/cs_courses/jdusza_itr/sam_review_20261114/venv/bin/python; mode=worker;;
 rivendell) python=/opt/venv/jdusza_venv/bin/python; mode=worker;;
 *) exit 2;;
esac
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec 9>supervisor.lock
flock -n 9 || exit 0
trap 'printf "%s\n" "$?" > supervisor.exit' EXIT
for attempt in 1 2 3; do
  printf 'Supervisor attempt %s at %s\n' "$attempt" "$(date --iso-8601=seconds)"
  set +e
  "$python" -u code/overnight_sam_v2.py "$mode" --work "$PWD" --host-config "$PWD/$host.json"
  result=$?
  set -e
  if [[ "$result" -eq 0 ]]; then exit 0; fi
  printf 'Supervisor recorded exit %s; retrying after 30 seconds.\n' "$result"
  sleep 30
done
exit "$result"
