"""
aimmd.execute.utils
==================

Utility functions for robust execution and task wrapping.

This module provides:

- `execute_command`: run a shell command via `subprocess.Popen` while:
  - streaming stdout in (near) real time,
  - periodically checking a Python-side stop condition,
  - enforcing a walltime limit,
  - handling graceful termination and forced kill,
  - optionally raising on non-zero exit codes.

- `target_wrapper`: wrap arbitrary Python callables to:
  - print a consistent start/exit banner including PID and thread ID,
  - surface error information in logs,
  - standardize process/thread task output for debugging.

Design notes
------------
- `execute_command` is designed for long-running external commands
  (e.g., GROMACS) where you want continuous logging and cooperative stopping.
- Output streaming is implemented with `select.select` on the subprocess stdout
  file descriptor (POSIX-friendly approach).
- Process termination targets the *process group* via `os.killpg`, enabled by
  creating the subprocess in a new session (`preexec_fn=os.setsid`).
"""

# external
import os
import sys
import time
import select
import signal
import threading
import subprocess
import multiprocessing
from math import inf
import traceback

# aimmd imports
from ..core.utils import now


def execute_command(command, stop_condition=lambda: False,
                    walltime=inf, termination_timeout=60.,
                    raise_if_failure=True, log_file='stdout'):
    """
    Execute a shell command while streaming output and polling a stop condition.

    The command is executed in a dedicated subprocess (via `subprocess.Popen`)
    and its stdout/stderr stream is captured and forwarded in near real time
    to either:
    - `sys.stdout` (default), or
    - a file-like object supporting `.write(...)`.

    The loop repeatedly:
    - reads and prints available output (non-blocking),
    - checks whether the process has exited,
    - evaluates `stop_condition()`,
    - enforces a maximum walltime.

    Parameters
    ----------
    command : str
        Command string executed with `shell=True`.
    stop_condition : callable, optional
        Zero-argument callable returning bool.
        If True, the subprocess is terminated and the function returns 0.
    walltime : float, optional
        Maximum allowed runtime in seconds. If exceeded, the subprocess is
        terminated and the function returns 0.
        Default is `math.inf` (no walltime).
    termination_timeout : float, optional
        Seconds to wait after sending SIGTERM before escalating to SIGKILL.
    raise_if_failure : bool, optional
        If True, a non-zero subprocess exit code raises RuntimeError.
        If False, the exit code is returned.
    log_file : {'stdout'} or file-like, optional
        Where to write logs/output.
        - If 'stdout', output is written to `sys.stdout`.
        - If a file-like object, `.write(text)` is used.

    Returns
    -------
    int
        - The subprocess exit code if it completes normally.
        - 0 if stopped by `stop_condition`, walltime, or KeyboardInterrupt.
        - 1 if an exception occurs and `raise_if_failure` is False.

    Raises
    ------
    RuntimeError
        If the subprocess exits with non-zero status and `raise_if_failure` is True.
        Also raised if an internal exception occurs and `raise_if_failure` is True.

    Notes
    -----
    - The subprocess is launched in a new process group (`os.setsid`), allowing
      group termination via `os.killpg`.
    - Output is read via non-blocking polling (`select.select`).
    - On stop/interrupt/exception, remaining buffered output is read and printed.
    """
    # Normalize logging target
    if log_file == 'stdout':
        log_file = sys.stdout

    # Start time for walltime enforcement
    t0 = time.time()

    # Inherit environment, but enforce unbuffered Python for child processes
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Allow GROMACS (or other tools) to override OpenMP threads if requested
    if ' -ntomp ' in command:
        env.pop("OMP_NUM_THREADS", None)

    # Launch subprocess:
    # - shell=True to accept a command string
    # - stdout=PIPE and stderr redirected to stdout for unified streaming
    # - preexec_fn=os.setsid to create a new process group
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=env,
        preexec_fn=os.setsid
    )

    if log_file:
        print(f'Executing "{command}" {now()}', file=log_file)

    def print_output(whole_text=False):
        """
        Stream available output from the subprocess.

        Parameters
        ----------
        whole_text : bool, optional
            If True, read all remaining output at once.
            If False, read line-by-line when available.

        Notes
        -----
        Uses `select.select` with a short timeout to avoid blocking.
        """
        try:
            if select.select([process.stdout.fileno()], [], [], 0.1)[0]:
                text = process.stdout.readline() if not whole_text else process.stdout.read()
                if log_file == sys.stdout:
                    print(text, end="", flush=True)
                elif log_file:
                    log_file.write(text)
        except (OSError, ValueError):
            # Can happen if fd is closed while polling
            pass

    def terminate_process_and_print_remaining_output():
        """
        Terminate the subprocess process group and flush remaining output.

        Termination sequence:
        1) SIGTERM the process group
        2) wait up to `termination_timeout`
        3) if still alive, SIGKILL the process group
        4) wait until process exits
        5) read remaining stdout and close handle
        """
        # First attempt: graceful termination
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception as exception:
            warning_message = (f'Warning: sending SIGTERM to {command!r} '
                               f'resulted in exception {exception}')
            if log_file:
                print(warning_message, file=log_file)

        # Wait for completion and flush output
        try:
            process.wait(timeout=termination_timeout)
            print_output(whole_text=True)
            return
        except subprocess.TimeoutExpired:
            pass

        # Escalation: force kill
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception as exception:
            warning_message = (f'Warning: sending SIGKILL to {command!r} '
                               f'resulted in exception {exception}')
            if log_file:
                print(warning_message, file=log_file)

        process.wait()
        print_output(whole_text=True)

        # Close stdout pipe safely
        try:
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
        except Exception:
            pass

    try:
        while True:
            # 1) Stream output continuously
            print_output()

            # 2) Check if the process exited
            exit_code = process.poll()
            if exit_code is not None:
                if exit_code and raise_if_failure:
                    error_message = (f'{command!r} failed '
                                     f'with exit code {exit_code}')
                    # If logging to file, also write an explicit marker
                    if log_file and log_file != sys.stdout:
                        print(f'RuntimeError: {error_message}', file=log_file)
                    raise RuntimeError(error_message)
                return exit_code

            # 3) Cooperative stop request or walltime
            if stop_condition() or time.time() - t0 > walltime:
                if log_file:
                    print(f'[StopCondition] Terminating "{command}" ({time.ctime()})',
                          file=log_file)
                terminate_process_and_print_remaining_output()
                return 0

    except KeyboardInterrupt:
        # User interrupt: terminate subprocess and exit cleanly
        if log_file:
            print(f'[KeyboardInterrupt] Terminating "{command}" ({time.ctime()})',
                  file=log_file)
        terminate_process_and_print_remaining_output()
        return 0

    except Exception as exception:
        # Any exception in the control loop: terminate and optionally raise
        if log_file:
            print(f'[Exception] Terminating "{command}" ({time.ctime()})',
                  file=log_file)
        terminate_process_and_print_remaining_output()

        # print traceback, even if we're not raising failures, for DEBUG
        traceback_message = (f'Exception in execute_command control loop: '
                             f'{traceback.format_exc()}')
        if log_file and log_file != sys.stdout:
            print(traceback_message, file=log_file)

        if raise_if_failure:
            # report failure with the original exception message if available
            error_message = f'{command!r} failed with exit code 1'
            if str(exception):
                error_message += f' and exception: {exception}'
            
            if log_file and log_file != sys.stdout:
                print(f'RuntimeError: {error_message}', file=log_file)
            raise RuntimeError(error_message)
        return 1

    finally:
        # Ensure subprocess group is not leaked if still running
        if process.poll() is None:
            terminate_process_and_print_remaining_output()


def target_wrapper(target, name, *args, **kwargs):
    """
    Wrap a callable to provide standardized logging and error reporting.

    This wrapper is used by :class:`aimmd.execute.base.TaskExecutor` when
    launching targets in threads/processes (especially important under
    multiprocessing "spawn", where child initialization can be opaque).

    Parameters
    ----------
    target : callable
        The function to call.
    name : str
        Human-readable identifier for log messages.
    *args
        Positional arguments forwarded to `target`.
    **kwargs
        Keyword arguments forwarded to `target`.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Re-raises any exception raised by `target` after recording a message.

    Notes
    -----
    - Prints PID and thread ID to help debugging concurrency issues.
    - If an exception occurs, the exception message is included in the final line.
    """
    pid = os.getpid()
    tid = threading.get_ident()
    error_message = ''
    try:
        print(f"[PID {pid}, TID {tid}: {name}] starting {now()}")
        if args:
            print(f'...   args: {args}')
        if kwargs:
            print(f'... kwargs: {kwargs}')
        target(*args, **kwargs)
    except Exception as exception:
        error_message = str(exception)
        raise exception
    finally:
        if not error_message:
            print(f"[PID {pid}, TID {tid}: {name}] exited correctly")
        else:
            print(f"[PID {pid}, TID {tid}: {name}] exited with error"
                  f" ({error_message})")
