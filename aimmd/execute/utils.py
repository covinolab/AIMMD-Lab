"""
...
"""

# external
import os
import sys
import time
import select
import signal
import threading
import traceback
import subprocess
import multiprocessing
from math import inf

# aimmd imports
from ..core.utils import now

# utils
def execute_command(command, stop_condition=lambda : False,
                    walltime=inf, termination_timeout=60.,
                    raise_if_failure=True, log_file='stdout'):
    """On bash shell spawned by process.
    File: print to file."""
    if log_file == 'stdout':
        log_file = sys.stdout

    # time and environment
    t0 = time.time()
    env = os.environ.copy()  # inherit resources of parent process
    env["PYTHONUNBUFFERED"] = "1"
    if ' -ntomp ' in command:  # let gromacs override num threads
        env.pop("OMP_NUM_THREADS", None)
    
    # start subprocess
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=env,
        preexec_fn=os.setsid)

    if log_file:
        print(f'Executing "{command}" {now()}', file=log_file)

    # procedure: print the output in real time
    def print_output(whole_text=False):
        try:
            if select.select([process.stdout.fileno()], [], [], 0.1)[0]:
                text = process.stdout.readline() if not whole_text else process.stdout.read()
                if log_file == sys.stdout:
                    print(text, end="", flush=True)
                elif log_file:
                    log_file.write(text)
        except (OSError, ValueError):
            pass

    # procedure: handle termination gracefully
    def terminate_process_and_print_remaining_output():
        try:
            os.killpg(process.pid, signal.SIGTERM)  # Use SIGTERM only
        except Exception as exception:
            warning_message = (f'Warning: sending SIGTERM to {command!r} '
                               f'resulted in exception {exception}')
            if log_file:
                print(warning_message, file=log_file)

        try:
            process.wait(timeout=termination_timeout)
            print_output(whole_text=True)
            return
        except subprocess.TimeoutExpired:
            pass

        # force kill if still running
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception as exception:
            warning_message = (f'Warning: sending SIGKILL to {command!r} '
                               f'resulted in exception {exception}')
            if log_file:
                print(warning_message, file=log_file)

        process.wait()
        print_output(whole_text=True)

        try:
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
        except Exception:
            pass

    # execution cycle: run, check stop condition, and print output if needed
    try:
        while True:
            # print the output in real time
            print_output()

            # process terminated
            exit_code = process.poll()
            if exit_code is not None:
                if exit_code and raise_if_failure:
                    error_message = (f'{command!r} failed '
                                     f'with exit code {exit_code}')
                    if log_file and log_file != sys.stdout:
                        print(f'RuntimeError: {error_message}', file=log_file)
                    raise RuntimeError(error_message)
                return exit_code

            # stop condition or walltime reached
            if stop_condition() or time.time() - t0 > walltime:
                if log_file:
                    print(f'[StopCondition] Terminating "{command}" ({time.ctime()})',
                          file=log_file)
                terminate_process_and_print_remaining_output()
                return 0

    # keyboard
    except KeyboardInterrupt:
        if log_file:
            print(f'[KeyboardInterrupt] Terminating "{command}" ({time.ctime()})',
                  file=log_file)
        terminate_process_and_print_remaining_output()
        return 0

    except Exception as exception:
        if log_file:
            print(f'[Exception] Terminating "{command}" ({time.ctime()})',
                  file=log_file)
        terminate_process_and_print_remaining_output()
        if raise_if_failure:
            error_message = f'{command!r} failed with exit code 1'
            if log_file and log_file != sys.stdout:
                print(f'RuntimeError: {error_message}', file=log_file)
            raise RuntimeError(error_message)
        return 1

    # ensure cleanup
    finally:
        if process.poll() is None:
            terminate_process_and_print_remaining_output()


def target_wrapper(target, name, *args, **kwargs):
    """Wrapper that signals parent after Python is fully initialized.
    Used by TaskExecutor to coordinate multiple processes.
    Necessary with multiprocessing spawn.
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
        #traceback.print_exc()
        raise exception
    finally:
        if not error_message:
            print(f"[PID {pid}, TID {tid}: {name}] exited correctly")
        else:
            print(f"[PID {pid}, TID {tid}: {name}] exited with error"
                  f" ({error_message})")
