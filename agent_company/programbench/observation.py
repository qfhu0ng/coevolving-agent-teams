"""Opt-in, non-injected ProgramBench observations. No provider or evaluator access.

File scope matches source export: regular files below delivery root, excluding any
.git and root executable/reference; worker worktrees are outside that root. Git
uncommitted_paths follows porcelain status (untracked files included, ignored files
excluded). staging_without_commits means delivery has regular files and nonignored
uncommitted paths but no commits after the initial observed HEAD; it is not a quality
or progress judgment. Source bodies, path names and commit IDs are never emitted.

Container memory counters are cgroup-v2 readings. Shutdown never waits for a
sample. The last successful inference reading is only a lower bound on the final
peak; final rows after Docker stop/start have a distinct scope as counters reset. Polling, Git and metadata walks consume resources: period and
actual wall-time overhead are recorded. Sampling errors never alter task outcome.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import threading
import time


# Run with python -I in the container. Only stat/git metadata and cgroup counters;
# no recursive symlink traversal, content reads, user Python imports or writes.
# GNU timeout bounds the in-container process too, not just its docker client.
SAMPLE_SCRIPT = r'''
import json, os, pathlib, stat, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
initial = sys.argv[2]
memory_only = sys.argv[3] == 'memory'
deadline = time.monotonic() + 1.5
out = {'file_count': None, 'source_bytes': None, 'latest_write_epoch_seconds': None,
       'uncommitted_paths': None, 'commits_since_start': None,
       'memory_peak_bytes': None, 'memory_events': None, 'cpu_stat': None, 'issues': []}

def git(*args):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    result = subprocess.run(['git', '--no-optional-locks', '-c', 'core.fsmonitor=false',
                             '-c', 'core.untrackedCache=false', '-C', str(root), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=remaining, check=True)
    return result.stdout

if not memory_only:
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError
        files = size = seen = 0
        newest = None
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    seen += 1
                    if seen > 100000 or time.monotonic() > deadline:
                        raise TimeoutError
                    if entry.name == '.git' or directory == root and entry.name in ('executable', 'reference'):
                        continue
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(pathlib.Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        files += 1
                        size += info.st_size
                        newest = max(newest or 0, info.st_mtime)
                    else:
                        raise ValueError
        out.update(file_count=files, source_bytes=size, latest_write_epoch_seconds=newest)
    except Exception:
        out['issues'].append('delivery_metadata_unavailable')
    try:
        # -z avoids newlines in filenames; rename/copy records contain two paths.
        records = iter(git('status', '--porcelain=v1', '-z', '--untracked-files=all').split(b'\0'))
        count = 0
        for record in records:
            if not record:
                continue
            if len(record) < 4 or record[2:3] != b' ':
                raise ValueError
            path = record[3:]
            if not any(part == b'.git' for part in path.split(b'/')) and path.split(b'/')[0] not in (b'executable', b'reference'):
                count += 1
            if b'R' in record[:2] or b'C' in record[:2]:
                next(records)
        out['uncommitted_paths'] = count
    except Exception:
        out['issues'].append('git_status_unavailable')
    try:
        if initial:
            out['commits_since_start'] = int(git('rev-list', '--count', initial + '..HEAD'))
        else:
            head = git('rev-parse', 'HEAD').decode().strip()
            if len(head) in (40, 64) and all(c in '0123456789abcdef' for c in head):
                out['_initial_head'] = head
                out['commits_since_start'] = 0
    except Exception:
        out['issues'].append('git_commits_unavailable')

# Resolve this process's v2 membership; never substitute the host root counters
# when the container is running with a host cgroup namespace.
try:
    memberships = pathlib.Path('/proc/self/cgroup').read_text().splitlines()
    member = next(line[3:] for line in memberships if line.startswith('0::'))
    if not member.startswith('/') or '..' in pathlib.PurePosixPath(member).parts:
        raise ValueError
    cgroup = pathlib.Path('/sys/fs/cgroup') / member.lstrip('/')
except Exception:
    cgroup = None
try:
    value = (cgroup / 'memory.peak').read_text().strip()
    if not value.isdigit():
        raise ValueError
    out['memory_peak_bytes'] = int(value)
except Exception:
    out['issues'].append('memory_peak_unavailable')
try:
    pairs = [line.split() for line in (cgroup / 'memory.events').read_text().splitlines()]
    allowed = {'low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill', 'sock_throttled'}
    if not pairs or any(len(p) != 2 or not p[1].isdigit() for p in pairs):
        raise ValueError
    out['memory_events'] = {key: int(value) for key, value in pairs if key in allowed}
except Exception:
    out['issues'].append('memory_events_unavailable')
# Cumulative CPU time and throttling of the same cgroup (plan v4 section 10).
try:
    pairs = [line.split() for line in (cgroup / 'cpu.stat').read_text().splitlines()]
    allowed = {'usage_usec', 'user_usec', 'system_usec', 'nr_periods', 'nr_throttled', 'throttled_usec'}
    if not pairs or any(len(p) != 2 or not p[1].isdigit() for p in pairs):
        raise ValueError
    out['cpu_stat'] = {key: int(value) for key, value in pairs if key in allowed}
except Exception:
    out['issues'].append('cpu_stat_unavailable')
print(json.dumps(out, allow_nan=False))
'''


FIELDS = ("file_count", "source_bytes", "latest_write_epoch_seconds", "uncommitted_paths",
          "commits_since_start", "memory_peak_bytes")
ISSUES = frozenset({"delivery_metadata_unavailable", "git_status_unavailable", "git_commits_unavailable",
                    "memory_peak_unavailable", "memory_events_unavailable", "cpu_stat_unavailable"})
CPU_STAT = frozenset({"usage_usec", "user_usec", "system_usec", "nr_periods", "nr_throttled", "throttled_usec"})
MEMORY_EVENTS = frozenset({"low", "high", "max", "oom", "oom_kill", "oom_group_kill", "sock_throttled"})


def _number(value):
    return (value if not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0 else None)


class ProgramBenchObserver:
    """Host-only JSONL observer. Public methods swallow observation failures."""

    def __init__(self, workspace, output: Path, runtime: Path, *, interval_seconds: float, execution_mode: str):
        self.workspace, self.output, self.runtime = workspace, Path(output), Path(runtime)
        self.interval_seconds, self.execution_mode = interval_seconds, execution_mode
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._initial_head = None
        self._stop_requested_monotonic = None
        self._last_inference_memory = None
        self._worker_offset = 0
        self._workers = set()
        self._worker_invalid = False
        self._worker_events_seen = False
        self._summary = {"enabled": True, "interval_seconds": interval_seconds,
                         "container_timeout_seconds": 2, "host_timeout_seconds": 3,
                         "metadata_entry_limit": 100000, "samples": 0, "sample_failures": 0,
                         "write_failures": 0, "observed_wall_seconds": 0.0,
                         "final_snapshot": None, "shutdown_requested_at_epoch_seconds": None,
                         "memory_peak_scope": "last_observed_inference_peak_lower_bound_not_exact_final_peak",
                         "injected_into_agent": False,
                         "schedule": "fixed_delay_after_previous_sample",
                         "overhead_scope": "sum_of_sampling_and_persistence_wall_time_not_cpu_or_causal_slowdown",
                         "delivery_scope": "export_regular_files_excluding_all_dotgit_and_root_executable_reference",
                         "uncommitted_scope": "git_porcelain_nonignored_paths_with_same_export_exclusions",
                         "snapshot_atomic": False,
                         "worker_count_scope": "distinct_relay_worker_registered_events_not_live_workers"}

    def start(self):
        try:
            self.sample("initial")
            self._thread = threading.Thread(target=self._poll, daemon=True, name="programbench-observer")
            self._thread.start()
        except Exception:
            self._summary["sample_failures"] += 1

    def _poll(self):
        while not self._stop.wait(self.interval_seconds):
            self.sample("periodic")

    def request_stop(self):
        # Called before container stop: no sampling, joining or observer lock.
        if self._stop_requested_monotonic is None:
            self._stop_requested_monotonic = time.monotonic()
            self._summary["shutdown_requested_at_epoch_seconds"] = time.time()
        self._stop.set()

    def stop_periodic(self):
        # Join only after the caller has terminated/quiesced the agent container.
        self.request_stop()
        try:
            if self._thread is not None:
                self._thread.join(timeout=3.2)
        except Exception:
            self._summary["sample_failures"] += 1

    def finish(self):
        self.stop_periodic()
        self.sample("final")

    def _worker_count(self):
        if self.execution_mode == "single":
            return 0
        try:
            with (self.runtime / "relay/events.jsonl").open("rb") as stream:
                self._worker_events_seen = True
                if stream.seek(0, 2) < self._worker_offset:
                    self._worker_invalid = True
                    return None
                stream.seek(self._worker_offset)
                # Bounded incremental metadata reads; do not parse an unfinished final line.
                consumed = 0
                while consumed < 2 * 1024 * 1024:
                    line = stream.readline(1024 * 1024)
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        return None
                    consumed += len(line)
                    self._worker_offset = stream.tell()
                    try:
                        event = json.loads(line)
                        if event.get("event") == "worker_registered":
                            name = event.get("name")
                            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                                raise ValueError
                            self._workers.add(name)
                    except (ValueError, TypeError, AttributeError):
                        self._worker_invalid = True
                if stream.read(1):
                    return None
            return None if self._worker_invalid else len(self._workers)
        except FileNotFoundError:
            # The relay creates events.jsonl on its first event: never-existed means zero registered,
            # but a file that vanished after being read stays unknown.
            return 0 if not self._worker_events_seen and self._worker_offset == 0 else None
        except (OSError, ValueError):
            return None

    def sample(self, phase: str, *, memory_only: bool = False):
        if not self._lock.acquire(timeout=3.2):
            self._summary["sample_failures"] += 1
            return
        started = time.monotonic()
        row = {key: None for key in FIELDS}
        row.update(phase=phase, observed_at_epoch_seconds=time.time(), memory_events=None, cpu_stat=None,
                   registered_worker_count=None, staging_without_commits=None, issues=[],
                   memory_scope="post_quiesce_container" if phase == "final" else "inference_container")
        try:
            try:
                result = self.workspace.run(
                    ["timeout", "--signal=KILL", "2s", "python3", "-I", "-c", SAMPLE_SCRIPT,
                     self.workspace.repo_path, self._initial_head or "", "memory" if memory_only else "all"],
                    cwd="/", timeout=3)
                values = json.loads(result.stdout)
                if not isinstance(values, dict):
                    raise ValueError
                row.update({key: _number(values.get(key)) for key in FIELDS})
                memory = values.get("memory_events")
                if isinstance(memory, dict) and all(key in MEMORY_EVENTS and isinstance(value, int)
                        and not isinstance(value, bool) and value >= 0 for key, value in memory.items()):
                    row["memory_events"] = memory
                cpu = values.get("cpu_stat")
                if isinstance(cpu, dict) and all(key in CPU_STAT and isinstance(value, int)
                        and not isinstance(value, bool) and value >= 0 for key, value in cpu.items()):
                    row["cpu_stat"] = cpu
                row["issues"] = [item for item in values.get("issues", []) if isinstance(item, str) and item in ISSUES]
                # Capture the existing starting commit, never create a new hash or baseline artifact.
                head = values.get("_initial_head")
                if phase == "initial" and isinstance(head, str) and re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", head):
                    self._initial_head = head
                if phase != "initial" and self._initial_head is None:
                    row["commits_since_start"] = None
                if not memory_only and all(row[key] is not None for key in
                        ("file_count", "uncommitted_paths", "commits_since_start")):
                    row["staging_without_commits"] = (row["file_count"] > 0 and row["uncommitted_paths"] > 0
                                                       and row["commits_since_start"] == 0)
            except Exception:
                row["issues"] = ["observation_unavailable"]
            row["registered_worker_count"] = self._worker_count()
            if row["registered_worker_count"] is None:
                row["issues"].append("registered_worker_count_unavailable")
            finished = time.monotonic()
            row["observation_wall_seconds"] = round(finished - started, 6)
            row["sample_completed_at_epoch_seconds"] = time.time()
            stop_at = self._stop_requested_monotonic
            if phase != "final" and stop_at is not None and finished >= stop_at:
                row["memory_scope"] = "unknown_shutdown_overlap"
            if row["memory_scope"] == "inference_container" and row["memory_peak_bytes"] is not None:
                self._last_inference_memory = (finished, {
                    key: row[key] for key in ("observed_at_epoch_seconds", "sample_completed_at_epoch_seconds",
                                             "memory_peak_bytes", "memory_events")})
            latest = row["latest_write_epoch_seconds"]
            row["latest_write_age_seconds"] = (max(0, row["observed_at_epoch_seconds"] - latest)
                                                if latest is not None else None)
            self._summary["samples"] += 1
            self._summary["sample_failures"] += bool(row["issues"])
            if phase == "final":
                self._summary["final_snapshot"] = row
            try:
                with self.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            except Exception:
                self._summary["write_failures"] += 1
        except Exception:
            self._summary["sample_failures"] += 1
        finally:
            self._summary["observed_wall_seconds"] += round(time.monotonic() - started, 6)
            self._lock.release()

    def summary(self) -> dict:
        result = dict(self._summary)
        last, stop_at = self._last_inference_memory, self._stop_requested_monotonic
        if last is not None and (stop_at is None or last[0] < stop_at):
            result["last_inference_memory_observation"] = {
                **last[1], "completed_before_shutdown_seconds":
                round(stop_at - last[0], 6) if stop_at is not None else None}
        else:
            result["last_inference_memory_observation"] = None
        return result
