#!/bin/bash
# Diagnostic capture for the intermittent macOS pytest-xdist shutdown hang.
#
# Run this by hand over SSH once a `[actions ssh]` job has hung (pytest printed
# its "N passed in Ms" summary but the step keeps spinning). It captures, for
# the controller + every worker + every descendant/orphan python process:
#   - the process tree and per-worker liveness
#   - Python+thread stacks (py-spy), TWICE ~30s apart so a true deadlock
#     (frozen stacks) is distinguishable from a slow-shutdown livelock (moving)
#   - native stacks (sample) as backup
#   - the full pipe FD table, and a best-effort test of the inherited-FD
#     hypothesis: does a NON-worker process hold a worker's execnet pipe?
#
# sudo is passwordless on GitHub-hosted macOS runners. Everything lands in
# ${OUT:-./hang-capture} and is tarred to /tmp/hang-capture.tgz for scp/upload.
set -u
OUT="${OUT:-$(pwd)/hang-capture}"
mkdir -p "$OUT"
python -m pip install --quiet py-spy 2>/dev/null || true

PYRE='Contents/MacOS/Python'   # matches every framework python proc on the runner
allpy() { pgrep -f "$PYRE" | sort -un; }

CTRL=$(pgrep -f 'pytest.*loadscope' | head -1)
if [ -z "$CTRL" ]; then
  echo "No controller (pytest --dist loadscope) found; is the run over?" | tee "$OUT/NO_CONTROLLER.txt"
  exit 1
fi
WORKERS=$(pgrep -P "$CTRL" | sort -un)
echo "controller=$CTRL"
echo "workers=$(echo $WORKERS | tr '\n' ' ')"

snapshot() {  # $1 = pass label
  local pass="$1" d="$OUT/pass_$1"
  mkdir -p "$d"
  ps -axo pid,ppid,pgid,stat,etime,command | grep -F "$PYRE" | grep -v grep > "$d/ps.txt"
  {
    echo "controller=$CTRL"
    for w in $WORKERS; do
      printf "worker %s: " "$w"; ps -o stat=,etime= -p "$w" 2>/dev/null || echo "GONE"
    done
  } > "$d/worker_liveness.txt"
  for pid in $(allpy); do
    sudo py-spy dump --pid "$pid" > "$d/pyspy_$pid.txt" 2>&1 || true
    sample "$pid" 1 -file "$d/sample_$pid.txt" >/dev/null 2>&1 || true
  done
  sudo lsof -p "$(allpy | tr '\n' ',' | sed 's/,$//')" 2>/dev/null > "$d/lsof_full.txt"
  grep -iE 'PIPE|FIFO' "$d/lsof_full.txt" > "$d/lsof_pipes.txt"
  echo "  pass $pass captured -> $d"
}

echo "=== snapshot 1 ==="; snapshot 1
echo "=== waiting 30s to compare stacks ==="; sleep 30
echo "=== snapshot 2 ==="; snapshot 2

# --- best-effort inherited-FD hypothesis test (uses pass_1) ---
PIPES="$OUT/pass_1/lsof_pipes.txt"
{
  echo "### Inherited-FD hypothesis test"
  echo "# The shared stderr pipe is inherited by ALL children and is NOT the"
  echo "# culprit. We look for a NON-worker, NON-controller process holding a"
  echo "# worker's execnet pipe (worker fd3/fd4). Anything printed below is the"
  echo "# smoking gun; empty => hypothesis refuted, look at worker stacks."
  echo
  # lsof PIPE columns: COMMAND PID USER FD TYPE DEVICE(=pipe id, $6) SIZE NAME(->peer)
  STDERR_PIPE=$(awk -v c="$CTRL" '$2==c && $4=="2" {print $6}' "$PIPES" | head -1)
  echo "# (shared stderr pipe, ignored: ${STDERR_PIPE:-none})"
  for w in $WORKERS; do
    for pipe in $(awk -v w="$w" '$2==w && ($4=="3"||$4=="4") {print $6}' "$PIPES"); do
      [ "$pipe" = "$STDERR_PIPE" ] && continue
      # who else holds this pipe id (as primary id or as ->peer)?
      grep -F "$pipe" "$PIPES" | awk -v w="$w" -v c="$CTRL" '$2!=w && $2!=c' \
        | sed "s/^/worker $w pipe $pipe also held by: /"
    done
  done
  echo
  echo "### Orphaned python processes (PPID 1) at hang time:"
  ps -axo pid,ppid,command | awk '$2==1' | grep -F "$PYRE" || echo "(none)"
} | tee "$OUT/ANALYSIS.txt"

tar czf /tmp/hang-capture.tgz -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null
echo
echo "Done. Review $OUT/ANALYSIS.txt and $OUT/pass_{1,2}/."
echo "Tarball: /tmp/hang-capture.tgz  (scp it off, or it's the upload-artifact path)"
echo "Key files to paste back: pass_1/pyspy_<worker>.txt, pass_2/pyspy_<worker>.txt,"
echo "  pass_1/lsof_pipes.txt, ANALYSIS.txt"
