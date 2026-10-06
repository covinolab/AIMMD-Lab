"""``python -m aimmd.network.nodetables``: prefill, repack and verify the
node-table series of AIMMD runs (see `aimmd.network.nodetables._cli`)."""

import os
import sys

if __name__ == '__main__':
    # The tools featurize on CPUs only. Hide the GPUs before the params file
    # is imported (it may move its network to a GPU), in this process and in
    # the worker processes, which inherit the environment.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from aimmd.network.nodetables._cli import main
    sys.exit(main())
