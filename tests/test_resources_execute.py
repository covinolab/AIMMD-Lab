import io
import os
import time

import numpy as np

from aimmd.execute.processes import ProcessExecutor
from aimmd.execute.threads import ThreadExecutor
from aimmd.execute.utils import execute_command, target_wrapper
from aimmd.resources.binding import bind_resources
from aimmd.resources.cpu import get_available_cpus
from aimmd.resources.gpu import get_available_gpus


def test_cpu_gpu_helpers_and_binding(monkeypatch):
    """Resource discovery and binding should honor the visible resource set."""

    monkeypatch.setattr("aimmd.resources.cpu.os.sched_getaffinity", lambda _: {1, 3})
    monkeypatch.setattr("aimmd.resources.gpu.get_num_gpus", lambda: 4)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert get_available_cpus() == [1, 3]
    assert get_available_gpus() == [0, 1, 2, 3]

    affinity_calls = []

    class DummyProcess:
        def cpu_affinity(self, cpus):
            affinity_calls.append(cpus)

    monkeypatch.setattr("aimmd.resources.binding.get_available_cpus", lambda: [0, 1, 2, 3])
    monkeypatch.setattr("aimmd.resources.binding.get_available_gpus", lambda: [0, 1])
    monkeypatch.setattr("aimmd.resources.binding.psutil.Process", lambda: DummyProcess())

    bind_resources(localid=1, cpus_per_task=2, gpus_per_task=1)
    # Worker 1 with 2 CPUs per task should be assigned the second contiguous CPU
    # slice, while one GPU per task selects the second visible GPU id.
    assert affinity_calls == [[2, 3]]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_execute_command_and_target_wrapper():
    """Exercise cooperative command execution and task wrapping."""

    log = io.StringIO()
    code = execute_command("python -c \"print('ok')\"", log_file=log)
    assert code == 0
    assert "ok" in log.getvalue()

    log = io.StringIO()
    # A true stop condition should terminate the child process cleanly and
    # report that path through the log output.
    code = execute_command("python -c \"import time; time.sleep(5)\"", stop_condition=lambda: True, log_file=log)
    assert code == 0
    assert "StopCondition" in log.getvalue()

    calls = []
    target_wrapper(lambda x: calls.append(x), "demo", 5)
    assert calls == [5]


def test_thread_and_process_executors():
    """The two executor backends should run small tasks through the common API."""

    seen = []

    def add_value(x):
        seen.append(x)

    threads = ThreadExecutor()
    threads.add(add_value, 7)
    threads.run(parallel=False)
    assert seen == [7]

    # For the process executor we only assert the lifecycle surface here rather
    # than any cross-process side effect, keeping the test deterministic.
    processes = ProcessExecutor()
    processes.add(time.sleep, 0.01)
    processes.run()
    time.sleep(0.1)
    assert processes.alive.shape == (1,)
    processes.clear()
