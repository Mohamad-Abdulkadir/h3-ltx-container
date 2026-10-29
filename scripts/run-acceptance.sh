#!/usr/bin/env bash
# Run only after setup and API startup on a dedicated GPU pod. No provisioning/publishing.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]] || ! command -v nvidia-smi >/dev/null; then
  printf 'Run the acceptance plan on the GPU pod only.\n' >&2; exit 2
fi
H3LTX_ACCEPTANCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$H3LTX_ACCEPTANCE_ROOT"
: "${BENCHMARK_RUN_ID:?Set a unique BENCHMARK_RUN_ID}"
for h3ltx_asset in first.png last.png clip.mp4 face-clip.mp4; do
  test -f "benchmark-assets/$h3ltx_asset" || { printf 'Missing benchmark asset: %s\n' "$h3ltx_asset" >&2; exit 2; }
done
# This command runs on the pod, never the operator's Mac. Serial transfers fit the 550 GB plan.
/opt/venvs/runtime/bin/h3ltx models sync --parallel 1 \
  --groups h3-core,h3-upscale,seedvr2,ltx-core,ltx-iclora,h3-ab-full,h3-turbo-extra,face-refine
mkdir -p benchmark-results
h3ltx_acceptance_status=0
/opt/venvs/runtime/bin/python scripts/matrix.py examples/acceptance-matrix.yaml \
  --run-id "$BENCHMARK_RUN_ID" --preload --continue-on-error --pull benchmark-results/runs \
  --stats-output benchmark-results/stats.json || h3ltx_acceptance_status=1
/opt/venvs/runtime/bin/python scripts/matrix.py examples/native-matrix.yaml \
  --run-id "$BENCHMARK_RUN_ID-native" --preload --continue-on-error --pull benchmark-results/runs \
  --stats-output benchmark-results/stats.json || h3ltx_acceptance_status=1
/opt/venvs/runtime/bin/python scripts/report.py --runs benchmark-results/runs \
  --stats benchmark-results/stats.json --output docs/BENCHMARKS.md
printf 'Acceptance complete. Review outputs and docs/BENCHMARKS.md before committing or publishing.\n'
exit "$h3ltx_acceptance_status"
