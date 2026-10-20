"""Per-attempt ProgramBench timeout; copied into evaluator containers only.

Each setup/call/teardown sequence shares 60 seconds. A rerun starts a fresh
sequence. Timeout failures remain ordinary pytest reports, so xdist workers and
the remainder of the test queue survive. After an exhausted attempt, teardown
has at most one second to clean fixture state; it cannot turn that failure into
a passing result. This is separate from the test's execution allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest
from _pytest.fixtures import FixtureDef

ATTEMPT_SECONDS = 60.0
CLEANUP_SECONDS = 1.0


def pytest_addoption(parser):
    parser.addoption("--ac-attempt-timeout", type=float, default=ATTEMPT_SECONDS,
                     help="ProgramBench seconds per attempt, including fixtures")


def pytest_configure(config):
    value = config.getoption("--ac-attempt-timeout")
    if not 0 < value <= ATTEMPT_SECONDS or not hasattr(signal, "setitimer"):
        raise pytest.UsageError("ProgramBench attempt timeout requires POSIX signals and 0 < timeout <= 60")


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_timeout_set_timer(item, settings):
    # The original outer protocol includes all reruns and cannot bound them.
    return True


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_timeout_cancel_timer(item):
    # In particular, pytest_exception_interact must not cancel our deadline.
    return True


def _descendants():
    """Only this pytest worker's children, never sibling workers or their work."""
    parents = {}
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                parents[int(entry.name)] = int(stat[1])
            except (OSError, ValueError, IndexError):
                continue  # Process exited between enumeration and reading.
    else:  # Local POSIX synthetic tests; deployment always uses /proc.
        result = subprocess.run(["ps", "-axo", "pid=,ppid="], check=True,
                                capture_output=True, text=True, timeout=1)
        parents = {int(pid): int(ppid) for pid, ppid in
                   (line.split() for line in result.stdout.splitlines())}
    descendants = []
    frontier = [os.getpid()]
    while frontier:
        frontier = [pid for pid, parent in parents.items() if parent in frontier]
        descendants.extend(frontier)
    return list(reversed(descendants))


def _kill_children():
    for pid in _descendants():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _guard_finalizers(item, deadline):
    """A caught timeout cannot let the next finalizer hang without a timer.

    FixtureDef.finish itself must still run: it clears cached fixture state even
    when its callbacks fail. Only the actual callbacks are skipped once cleanup
    is exhausted. These are the pytest 9 SetupState/FixtureDef finalizer lists.
    """
    collections = [row[0] for row in item.session._setupstate.stack.values()]
    collections += [fixture._finalizers for fixture in item._request._fixture_defs.values()
                    if isinstance(fixture, FixtureDef)]
    for finalizers in collections:
        for index, callback in enumerate(finalizers):
            original = getattr(callback, "_ac_original_finalizer", callback)
            target = original.func if isinstance(original, partial) else original
            if isinstance(getattr(target, "__self__", None), FixtureDef) and target.__name__ == "finish":
                continue

            def bounded(original=original):
                if item._ac_teardown_active and time.monotonic() >= deadline:
                    item._ac_attempt_timed_out = True
                    pytest.fail("ProgramBench cleanup deadline exhausted", pytrace=False)
                return original()

            bounded._ac_original_finalizer = original
            finalizers[index] = bounded


@contextmanager
def _phase(item, phase):
    now = time.monotonic()
    if phase == "setup":
        item._ac_attempt_deadline = now + item.config.getoption("--ac-attempt-timeout")
        item._ac_attempt_timed_out = False
    remaining = item._ac_attempt_deadline - now
    if remaining <= 0 and phase != "teardown":
        item._ac_attempt_timed_out = True
        _kill_children()
        pytest.fail("ProgramBench attempt exceeded its timeout", pytrace=False)
    # A timeout in setup/call still needs bounded fixture cleanup before rerun.
    allowance = max(remaining, CLEANUP_SECONDS) if phase == "teardown" and remaining <= 0 else remaining
    if phase == "teardown":
        item._ac_teardown_active = True
        _guard_finalizers(item, now + allowance)
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        item._ac_attempt_timed_out = True
        _kill_children()
        pytest.fail("ProgramBench attempt exceeded its timeout", pytrace=False)

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, allowance)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if phase == "teardown":
            item._ac_teardown_active = False
            if item._ac_attempt_timed_out:
                _kill_children()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    with _phase(item, "setup"):
        yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    with _phase(item, "call"):
        yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    with _phase(item, "teardown"):
        yield


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "setup":
        item._ac_failed_report = None
    if report.when != "teardown" and report.failed:
        item._ac_failed_report = report
    prior = getattr(item, "_ac_failed_report", None)
    if report.when == "teardown" and report.failed and prior is not None and item._ac_attempt_timed_out:
        # JUnit otherwise emits two testcase elements with the same name for a
        # failed call plus failed teardown, inflating the official denominator.
        # Retain both diagnostics in the already-failed testcase, never a pass.
        prior.longrepr = f"{prior.longrepr}\n\nProgramBench teardown diagnostics:\n{report.longrepr}"
        prior.sections.extend(report.sections)
        report.outcome = "passed"
        report.longrepr = None
