#!/bin/bash
# The weights lane: the tests that need the real BioEmu2 checkpoint
# (bioemu2-preview) or a GPU, which GitHub's CPU runners cannot run.
#
# Run it on the machine and in the environment of a campaign, before the
# campaign: on the workstation GPU, or inside a one-GPU job on JUPITER. It
# doubles as a check of the production install: it fails when the
# checkpoint is not in the Hugging Face cache (nothing is downloaded), when
# no GPU is visible, or when a weights test fails.
#
# Usage (from anywhere; the tests are those of this checkout):
#
#   devtools/run_weights_tests.sh [--python /path/to/python] [extra pytest arguments]
#
# Fetch the checkpoint once beforehand, on a machine with internet access and
# with the same HF_HOME as the jobs:
#
#   python -c "from bioemu2.checkpoints import get_checkpoint_files as g; print(g('bioemu2-preview'))"
#
# Exit status: that of pytest (5 when no test is marked bioemu2_weights or
# gpu), or 2 when the environment is not ready.

set -euo pipefail

PYTHON=python
if [[ "${1:-}" == "--python" ]]; then
    PYTHON="$2"
    shift 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# never download: a missing checkpoint fails at once
export HF_HUB_OFFLINE=1

echo "weights lane of ${ROOT}"
if ! "${PYTHON}" - <<'EOF'
import importlib.metadata
import sys

import torch

print(f'python {sys.version.split()[0]} ({sys.executable})')
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
if not torch.cuda.is_available():
    sys.exit('no GPU is visible (torch.cuda.is_available() is False)')
for index in range(torch.cuda.device_count()):
    print(f'GPU {index}: {torch.cuda.get_device_name(index)}')
for name in ('aimmd-lab', 'bioemu2', 'cuequivariance-torch'):
    try:
        print(f'{name} {importlib.metadata.version(name)}')
    except importlib.metadata.PackageNotFoundError:
        print(f'{name} not installed')
from bioemu2.checkpoints import get_checkpoint_files
try:
    files = get_checkpoint_files('bioemu2-preview')
except Exception as exc:                                   # noqa: BLE001
    sys.exit(f'the bioemu2-preview checkpoint is not in the Hugging Face cache '
             f'({type(exc).__name__}: {exc}); fetch it once with internet access')
print(f'checkpoint: {files}')
EOF
then
    echo "weights lane: the environment is not ready (see above)" >&2
    exit 2
fi

cd "${ROOT}"
set +e
"${PYTHON}" -m pytest -v --runweights -m "bioemu2_weights or gpu" "$@" tests/
status=$?
set -e
if [[ ${status} -eq 5 ]]; then
    echo "weights lane: no test is marked bioemu2_weights or gpu" >&2
fi
exit ${status}
