"""Fresh ProgramBench single/team inference followed by separate official evaluation.

This runner records compact runtime facts. It does not copy native reasoning,
provider responses or full workspace diagnostics into experiment summaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
import uuid

from agent_company.runtime_io import write_json_atomic

from agent_company.evolution.loop import RunRequest, TaskResult
from agent_company.runtime.network import NETWORK_POLICY
from agent_company.runtime.node_runtime import node_runtime_mounts
from agent_company.programbench.task import (OFFICIAL_EVALUATOR_CPUS, OFFICIAL_INFERENCE_CPUS,
                          OFFICIAL_INFERENCE_MEMORY_MB, OFFICIAL_INFERENCE_TIMEOUT_SEC,
                          ProgramBenchTask, create_submission, evaluate_submission)
from agent_company.programbench.workspace import ProgramBenchWorkspace, programbench_source_boundary
from agent_company.programbench.observation import ProgramBenchObserver
from agent_company.runtime.qwen import QwenTeamRunner, validate_stop_hook
from agent_company.runtime.relay import ProviderConfig, validate_model_call_limit


def _fields(value, names):
    return {name: value[name] for name in names if name in value} if isinstance(value, dict) else {}


def _runtime_summary(runtime: dict) -> dict:
    summary = _fields(runtime, (
        "model", "execution_mode", "policy_applied", "elapsed_seconds", "stop_reason", "scripted_provider",
        "max_model_calls", "logical_model_calls", "logical_calls_by_role", "logical_calls_by_purpose",
        "model_call_budget_rejections", "model_call_budget_rejections_by_role", "provider_attempts",
        "completed_attempts", "inflight_attempts", "active_requests", "usage_complete", "cost_complete",
        "known_estimated_cost", "currency", "calls_by_role", "deadline_rejections"))
    summary["usage"] = _fields(runtime.get("usage"),
        ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "total_tokens"))
    summary["execution"] = _fields(runtime.get("execution"),
        ("returncode", "timed_out", "elapsed_seconds", "oom_killed", "oom_killed_observed"))
    native = runtime.get("native_result") or {}
    summary["native_result"] = _fields(native, ("explicit_error", "missing_result"))
    summary["native_result"]["result"] = _fields(native.get("result"), ("type", "subtype", "is_error"))
    lifecycle = native.get("lifecycle") or {}
    summary["native_lifecycle"] = {
        "notification_errors_count": len(lifecycle.get("notification_errors") or []),
        "unobserved_failure_notices_count": len(lifecycle.get("unobserved_failure_notices") or []),
    }
    summary["worker_count"] = len(runtime.get("workers") or {})
    summary["provider_incidents"] = _fields(runtime.get("provider_incidents"), ("count", "by_kind", "by_role"))
    # Unknown usage/price is never changed into a measured zero or completeness.
    summary.setdefault("usage_complete", False)
    summary.setdefault("cost_complete", False)
    return summary


def _verification_summary(result: dict) -> dict:
    return _fields(result, (
        "task_id", "repository_id", "status", "reward", "resolved", "almost", "reward_source",
        "official_test_fraction", "passed_tests", "valid_tests", "test_status_counts", "error_code",
        "system_errors", "warnings_count", "submission_error", "verifier_elapsed_seconds",
        "official_evaluator", "official_info_filtering", "inference_environment_reused", "docker_cpus",
        "image_tag", "docker_executable", "evaluator_invocation",
        "evaluator_resource_limits_source", "official_evaluator_docker_cpus",
        "official_compile_timeout_seconds", "official_test_branch_timeout_seconds",
        "verifier_timeout_scope", "evaluator_exception_type", "evaluator_image"))


def _valid_reward(result: dict) -> float | None:
    reward = result.get("reward")
    if (result.get("status") == "completed" and not isinstance(reward, bool)
            and isinstance(reward, (int, float)) and math.isfinite(reward) and 0 <= reward <= 1):
        return float(reward)
    return None


def _usd(runtime: dict, *, complete: bool) -> float | None:
    if runtime.get("currency") != "USD" or (complete and not runtime.get("cost_complete")):
        return None
    value = runtime.get("known_estimated_cost")
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


VERIFIED_AFTER_CLEANUP_SUFFIX = ".before_verification.json"


def _container_absent(docker: str, name: str) -> bool:
    probe = subprocess.run([docker, "container", "inspect", "--format", "{{.State.Status}}", name],
                           capture_output=True, text=True, timeout=60)
    if probe.returncode == 0:
        return False
    if "no such" in (probe.stderr or "").lower():
        return True
    raise RuntimeError("Cannot confirm whether a ProgramBench container still exists")


def complete_verification(directory: Path, task: ProgramBenchTask, *, evaluator_cpus: int = OFFICIAL_EVALUATOR_CPUS,
                          docker: str = "docker") -> dict:
    """Run the official verifier for a trial that skipped it only because inference cleanup failed.

    The runner scores a submission only after confirmed inference cleanup. When removing a container
    timed out, the agent had already stopped and the submission was archived, so the score is still
    that submission's. Here cleanup is confirmed instead by neither of the trial's containers existing
    any more. The cleanup failure stays in the failures; the earlier records are kept beside the new ones.
    """
    directory = Path(directory)
    result = json.loads((directory / "task_result.json").read_text(encoding="utf-8"))
    metrics = result.get("metrics") or {}
    if [item.get("phase") for item in metrics.get("failures") or []] != ["inference_cleanup"]:
        raise ValueError("Only a trial whose single failure was inference cleanup is verified afterwards")
    if ((metrics.get("verifier") or {}).get("status") != "not_started"
            or metrics.get("agent_termination_confirmed") is not True):
        raise ValueError("The trial's agent did not stop cleanly or its verifier already ran")
    if "verification_after_cleanup" in metrics or (directory / "verifier").exists():
        raise ValueError("This trial already has a verifier directory or was verified after cleanup")
    submission = directory / "submission.tar.gz"
    if not submission.is_file() or not isinstance(metrics.get("submission"), dict):
        raise ValueError("The trial has no archived submission")
    name = json.loads((directory / "workspace.json").read_text(encoding="utf-8"))["container_name"]
    if not isinstance(name, str) or not re.fullmatch(r"ac-programbench-[a-f0-9]{32}", name):
        raise ValueError("Invalid ProgramBench container identity")
    if not all(_container_absent(docker, container) for container in (name, f"{name}-network")):
        raise RuntimeError("A container of this trial still exists; remove it before verification")
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    verification = _verification_summary(evaluate_submission(
        task, submission, directory / "verifier", docker_cpus=evaluator_cpus, docker_executable=docker))
    existing = [record for record in ("task_result.json", "runner_result.json", "execution.json")
                if (directory / record).is_file()]
    for record in existing:
        with open(directory / (record[:-5] + VERIFIED_AFTER_CLEANUP_SUFFIX), "x", encoding="utf-8") as stream:
            stream.write((directory / record).read_text(encoding="utf-8"))
    score = _valid_reward(verification)
    metrics["verifier"] = verification
    metrics["task_success"] = verification.get("resolved") if score is not None else None
    metrics["verification_after_cleanup"] = {"cleanup_confirmed": True, "submission_sha256": digest,
                                             "earlier_records_suffix": VERIFIED_AFTER_CLEANUP_SUFFIX}
    runtime = metrics.get(metrics.get("execution_mode")) or {}
    errors = []
    if runtime.get("stop_reason") not in {"native_cli_completed", "model_call_limit_exceeded"}:
        errors.append("Agent did not reach a normal or budget-limited terminal state")
    if score is None:
        errors.append("Official evaluation did not produce a valid complete reward")
    result.update(status="failed" if errors else "completed", score=score, metrics=metrics,
                  error="; ".join(errors) if errors else None,
                  trace_files=[name for name in ("workspace.json", "execution.json", "submission.tar.gz",
                                                 "verifier/result.json") if (directory / name).is_file()])
    for record in existing:
        write_json_atomic(directory / record, metrics if record == "execution.json" else result)
    return result


@dataclass
class ProgramBenchRunner:
    provider: ProviderConfig
    qwen_package: Path
    node_directory: Path
    platform: str = "linux/amd64"
    task_memory_mb: int = OFFICIAL_INFERENCE_MEMORY_MB
    task_cpus: float = OFFICIAL_INFERENCE_CPUS
    agent_timeout_seconds: float = OFFICIAL_INFERENCE_TIMEOUT_SEC
    max_model_calls: int | None = 1000
    evaluator_cpus: int = OFFICIAL_EVALUATOR_CPUS
    team_stall_timeout_seconds: float | None = 1800
    # Opt-in runtime team check-ins to the master; None keeps earlier runs' behavior.
    team_checkin_seconds: float | None = None
    # Opt-in teammate message delivery; native_idle keeps earlier runs' behavior.
    team_message_delivery: str = "native_idle"
    # Opt-in (2026-10-02): runtime waiting/loop-guard disclosure, and master
    # status-read loop recovery (Team with check-ins). False keeps earlier runs' behavior.
    runtime_disclosure: bool = False
    leader_loop_recovery: bool = False
    # Opt-in: "disabled" removes Qwen's per-turn tool-call cap for every role.
    tool_call_cap: str = "native_default"
    stop_hook: dict | None = None
    # Opt-in role-level communication policies for Team arms (global/master/propose).
    role_policy_mode: str = "off"
    role_policy_protocol: str | None = None
    # Active worker slots for Team arms (plan v4 section 10); 10 keeps earlier runs.
    teammate_limit: int = 10
    # B-only opt-in observer; no injection and no change to calls, scoring or stop policy.
    observation_interval_seconds: float | None = None
    upstream: object = None
    provider_supports_seed: bool = False
    docker: str = "docker"

    def __post_init__(self):
        validate_model_call_limit(self.max_model_calls)
        self.stop_hook = validate_stop_hook(self.stop_hook)
        from agent_company.runtime.relay import validate_teammate_limit
        validate_teammate_limit(self.teammate_limit)
        for name in ("task_memory_mb", "evaluator_cpus"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("task_cpus", "agent_timeout_seconds"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.team_stall_timeout_seconds is not None:
            value = self.team_stall_timeout_seconds
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("team_stall_timeout_seconds must be finite and positive or None")
        value = self.team_checkin_seconds
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError("team_checkin_seconds must be finite and positive or None")
        if self.team_message_delivery not in ("native_idle", "between_tool_calls"):
            raise ValueError("team_message_delivery must be native_idle or between_tool_calls")
        for name in ("runtime_disclosure", "leader_loop_recovery"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.leader_loop_recovery and self.team_checkin_seconds is None:
            raise ValueError("leader_loop_recovery requires team_checkin_seconds")
        if self.tool_call_cap not in ("native_default", "disabled"):
            raise ValueError("tool_call_cap must be native_default or disabled")
        if self.role_policy_mode not in ("off", "global", "master", "propose"):
            raise ValueError("role_policy_mode must be off, global, master or propose")
        if self.role_policy_protocol is not None:
            from agent_company.policy.role_package import AGENT_RULES_PROTOCOL, PROTOCOL_MODES, ROLE_POLICY_PROTOCOL
            if self.role_policy_protocol == AGENT_RULES_PROTOCOL:
                if self.role_policy_mode not in PROTOCOL_MODES[AGENT_RULES_PROTOCOL]:
                    raise ValueError("agent_rules_v1 implements mode propose only")
                if self.team_checkin_seconds is None:
                    raise ValueError("agent_rules_v1 requires team_checkin_seconds")
                if self.team_message_delivery != "between_tool_calls":
                    raise ValueError("agent_rules_v1 requires team_message_delivery between_tool_calls")
            elif self.role_policy_protocol != ROLE_POLICY_PROTOCOL or self.role_policy_mode not in ("off", "propose"):
                raise ValueError("role_package_v1 implements off and propose only")
        value = self.observation_interval_seconds
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError("observation_interval_seconds must be finite and positive or None")

    def assert_can_restart(self, directory: Path) -> None:
        """Recovery may only replace an old run whose identified agents stopped."""
        state_path = directory / "workspace.json"
        if not state_path.exists():
            if any((directory / mode).exists() for mode in ("single", "team")):
                raise RuntimeError("Cannot confirm termination without the prior workspace identity")
            return
        name = json.loads(state_path.read_text(encoding="utf-8"))["container_name"]
        if not isinstance(name, str) or not re.fullmatch(r"ac-programbench-[a-f0-9]{32}", name):
            raise ValueError("Invalid prior ProgramBench container identity")
        result = subprocess.run(
            [self.docker, "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.State}}"],
            capture_output=True, text=True, check=True, timeout=30)
        if any(state not in {"exited", "dead", "created"} for state in result.stdout.splitlines()):
            raise RuntimeError("Prior ProgramBench agents are still active")

    def __call__(self, request: RunRequest) -> TaskResult:
        started = time.monotonic()
        mode = request.execution_mode
        if mode not in {"single", "team"}:
            raise ValueError("execution_mode must be single or team")
        if not request.fresh_start:
            raise ValueError("ProgramBench trials require fresh task environments")
        effective_role_mode = self.role_policy_mode
        policy_package = None
        if self.role_policy_protocol is not None:
            from agent_company.policy.role_package import PROTOCOL_MODES, policy_protocol, validate_policy_package
            policy_package = validate_policy_package(request.policy_package)
            if policy_protocol(policy_package) != self.role_policy_protocol:
                raise ValueError("RunRequest policy package protocol disagrees with the runner")
            if mode == "team":
                effective_role_mode = request.role_policy_mode or self.role_policy_mode
                if effective_role_mode not in PROTOCOL_MODES[self.role_policy_protocol]:
                    raise ValueError("role_package_v1 Team arms allow off or propose only"
                                     if self.role_policy_protocol == "role_package_v1"
                                     else f"{self.role_policy_protocol} Team arms allow propose only")
                if (request.assignment_policy != policy_package["team"]["assignment"]
                        or request.communication_policy != policy_package["team"]["communication"]):
                    raise ValueError("RunRequest policy texts disagree with its role package")
            elif request.role_policy_mode is not None or request.assignment_policy or request.communication_policy:
                raise ValueError("Single role-package requests cannot apply Team policies or a Team mode")
        elif request.policy_package is not None or request.role_policy_mode is not None:
            raise ValueError("Per-arm policy settings require role_package_v1")
        task = ProgramBenchTask.load(
            request.task.payload["path"], instruction_path=request.task.payload["instruction_path"],
            agent_timeout_sec=self.agent_timeout_seconds, cpus=self.task_cpus, memory_mb=self.task_memory_mb)
        if task.task_id != request.task.task_id or task.repository_id != request.task.repo_id:
            raise ValueError("Manifest task/repository identity disagrees with public ProgramBench metadata")
        package, node = Path(self.qwen_package).resolve(), Path(self.node_directory).resolve()
        if not (package / "cli.js").is_file() or not (node / "bin/node").is_file():
            raise FileNotFoundError("Provide the Qwen package and a Linux Node >=22.15 directory")
        if mode == "team" and (not request.assignment_policy.strip() or not request.communication_policy.strip()):
            raise ValueError("Team mode requires both assignment and communication policies")
        output = Path(request.output_dir)
        if any((output / name).exists() for name in
               ("workspace.json", "execution.json", "runner_result.json", "submission.tar.gz",
                "source", "single", "team", "verifier")):
            raise FileExistsError("Use a fresh ProgramBench task output directory")
        output.mkdir(parents=True, exist_ok=True)
        provider = self.provider
        if request.seed is not None and self.provider_supports_seed:
            provider = replace(provider, extra_body={**provider.extra_body, "seed": request.seed})
        container_name = "ac-programbench-" + uuid.uuid4().hex
        resource_config = {"agent_timeout_seconds": task.agent_timeout_sec, "cpus": task.cpus,
                           "memory_mb": task.memory_mb, "memory_limit_scope": "task-shared",
                           "max_model_calls": self.max_model_calls,
                           "model_call_limit_scope": "task-shared" if self.max_model_calls is not None else None,
                           "evaluator_cpus": self.evaluator_cpus,
                           "resource_limits_source": task.metadata["resource_limits_source"],
                           "model_call_limit_protocol": "experiment mapping; not official multi-agent protocol"}
        if self.observation_interval_seconds is not None:
            resource_config["observation_interval_seconds"] = self.observation_interval_seconds
        metrics = {"task": task.describe(), "execution_mode": mode, "policy_applied": mode == "team",
                   "score_semantics": "official_filtered_test_fraction_not_task_success",
                   "resource_config": resource_config, "requested_seed": request.seed,
                   "provider_seed_applied": request.seed is not None and self.provider_supports_seed,
                   "provider_seed_determinism_claimed": False, "scripted_provider": self.upstream is not None,
                   "network_policy_requested": NETWORK_POLICY, "official_verifier_environment": "separate"}
        workspace_record = {"container_name": container_name, "execution_mode": mode,
                            "resource_config": resource_config, "network_policy_requested": NETWORK_POLICY}
        write_json_atomic(output / "workspace.json", workspace_record)
        workspace = None
        observer = None
        runtime: dict = {}
        verification: dict = {"status": "not_started", "reward": None}
        failures: list[dict] = []
        cleanup_attempted = False
        cleanup_confirmed = False
        phase = "workspace_configuration"

        def failure(where, error):
            # Never stringify subprocess errors: their command/output can contain
            # credentials, prompts or verifier details.
            failures.append({"phase": where, "error_type": type(error).__name__})

        def observe_image():
            # Unobserved stays unknown: the workspace reports error="not_observed" before
            # docker run, and an object without the attribute records {}, never an invented ID.
            metrics["inference_image"] = workspace_record["inference_image"] = _fields(
                getattr(workspace, "image_identity", None), ("reference", "image_id", "error", "source"))

        def observe_boundary():
            if workspace is None:
                return
            metrics["network_policy"] = _fields(workspace.network_policy_status,
                ("policy", "enforced", "namespace_container_id", "allowed_relay", "dependency_access"))
            workspace_record["network_policy"] = metrics["network_policy"]
            observe_image()

        try:
            workspace = ProgramBenchWorkspace(task, mounts=[(package, "/opt/ac-qwen", True),
                                                    *node_runtime_mounts(node, self.platform)],
                                              platform=self.platform, container_name=container_name,
                                              docker=self.docker)
            workspace.diagnostic_secrets = (provider.api_key,) if provider.api_key else ()
            phase = "workspace_start"
            workspace.start()
            if self.observation_interval_seconds is not None:
                observer = ProgramBenchObserver(workspace, output / "observations.jsonl", output / mode,
                    interval_seconds=self.observation_interval_seconds, execution_mode=mode)
                workspace.observation_before_quiesce = observer.request_stop
                observer.start()
            observe_image()
            try:
                # Persist provenance now: an interrupted run never reaches the final rewrite.
                write_json_atomic(output / "workspace.json", workspace_record)
            except OSError as exc:
                failure("workspace_record", exc)
            metrics["resources"] = _fields(workspace.resources,
                ("cpus", "memory_mb", "memory_swap_mb", "shared_by_all_agents",
                 "storage_mb_requested", "storage_quota_enforced", "additional_swap_allowed",
                 "agent_user", "reference_execute_only"))
            phase = "agent"
            try:
                raw_runtime = QwenTeamRunner(
                    "/opt/ac-qwen/cli.js", node="/opt/ac-node/bin/node", upstream=self.upstream,
                    team_stall_timeout_seconds=self.team_stall_timeout_seconds,
                    checkin_interval_seconds=self.team_checkin_seconds,
                    team_message_delivery=self.team_message_delivery,
                    runtime_disclosure=self.runtime_disclosure,
                    leader_loop_recovery=self.leader_loop_recovery,
                    tool_call_cap=self.tool_call_cap,
                    role_policy_mode=effective_role_mode,
                    stop_hook=self.stop_hook,
                    **({"policy_package": policy_package} if policy_package is not None else {}),
                    **({"teammate_limit": self.teammate_limit} if self.teammate_limit != 10 else {}),
                ).run(workspace, task.instruction,
                      request.assignment_policy if mode == "team" else "",
                      request.communication_policy if mode == "team" else "", output / mode,
                      provider, task.agent_timeout_sec, execution_mode=mode,
                      max_model_calls=self.max_model_calls, source_boundary=programbench_source_boundary())
                runtime = _runtime_summary(raw_runtime)
            except Exception as exc:
                failure("agent", exc)
                runtime = _runtime_summary({"stop_reason": "launcher_error"})
            phase = "agent_quiesce"
            workspace.finish_agent()
            metrics["agent_termination_confirmed"] = True
            try:
                phase = "source_export"
                workspace.export_source(output / "source")
                phase = "submission_archive"
                metadata = create_submission(output / "source", output / "submission.tar.gz")
                metrics["submission"] = _fields(metadata, ("path", "files", "source_bytes", "excluded_paths"))
                try:
                    # Optional evidence for pass rates over time; never a run failure. After the
                    # archive, so git cannot change the delivered source.
                    metrics["delivery_history"] = workspace.export_git_bundle(output / "delivery_history.bundle")
                except Exception as exc:
                    metrics["delivery_history"] = {"error_type": type(exc).__name__}
            finally:
                if observer is not None:
                    # Git status may invoke an agent-configured filter. Sample
                    # only after export/archive attempts, including their failure
                    # paths; a failed delivery still needs final diagnostics.
                    observer.finish()
            observe_boundary()
            phase = "inference_cleanup"
            cleanup_attempted = True
            workspace.stop()
            cleanup_confirmed = True
            metrics["inference_cleanup_confirmed"] = True
            phase = "official_verifier"
            verification = _verification_summary(evaluate_submission(
                task, output / "submission.tar.gz", output / "verifier", docker_cpus=self.evaluator_cpus,
                docker_executable=self.docker))
        except Exception as exc:
            failure(phase, exc)
        finally:
            if observer is not None:
                observer.request_stop()
            if workspace is not None and not cleanup_attempted:
                try:
                    observe_boundary()
                except Exception as exc:
                    failure("network_observation", exc)
                try:
                    cleanup_attempted = True
                    workspace.stop()
                    cleanup_confirmed = True
                except Exception as exc:
                    failure("inference_cleanup", exc)
            if observer is not None:
                # Never delay agent termination for a sample or a thread join.
                observer.stop_periodic()
                metrics["observation"] = observer.summary()
            metrics["inference_cleanup_confirmed"] = cleanup_confirmed
            workspace_record["inference_cleanup_confirmed"] = cleanup_confirmed
            try:
                write_json_atomic(output / "workspace.json", workspace_record)
            except OSError as exc:
                failure("workspace_record", exc)

        runtime.setdefault("usage_complete", False)
        runtime.setdefault("cost_complete", False)
        metrics.update({mode: runtime, "verifier": verification, "failures": failures,
                        "runtime_outcome": ("budget_limited" if runtime.get("stop_reason") == "model_call_limit_exceeded"
                                            else runtime.get("stop_reason", "not_started"))})
        score = _valid_reward(verification)
        metrics["task_success"] = verification.get("resolved") if score is not None else None
        errors = [f"{item['phase']}: {item['error_type']}" for item in failures]
        if runtime.get("stop_reason") not in {"native_cli_completed", "model_call_limit_exceeded"}:
            errors.append("Agent did not reach a normal or budget-limited terminal state")
        if score is None:
            errors.append("Official evaluation did not produce a valid complete reward")
        write_json_atomic(output / "execution.json", metrics)
        result = TaskResult(
            status="failed" if errors else "completed", score=score, metrics=metrics,
            usage=runtime.get("usage", {}), cost_usd=_usd(runtime, complete=True),
            known_cost_usd=_usd(runtime, complete=False), elapsed_seconds=time.monotonic() - started,
            error="; ".join(errors) if errors else None,
            trace_files=[name for name in ("workspace.json", "execution.json", "submission.tar.gz", "verifier/result.json")
                         if (output / name).is_file()],
        )
        write_json_atomic(output / "runner_result.json", asdict(result))
        return result
