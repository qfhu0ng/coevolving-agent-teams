"""Run a single Qwen agent or native team in fresh RoadmapBench environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
import subprocess
import uuid

from agent_company.runtime_io import write_json_atomic

from agent_company.evolution.loop import RunRequest, TaskResult
from agent_company.runtime.network import NETWORK_POLICY
from agent_company.runtime.node_runtime import node_runtime_mounts
from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig
from agent_company.roadmap.task import Task, score
from agent_company.runtime.workspace import DockerWorkspace, exception_diagnostic


@dataclass
class RoadmapRunner:
    provider: ProviderConfig
    qwen_package: Path
    node_directory: Path
    platform: str = "linux/amd64"
    team_stall_timeout_seconds: float | None = 1800
    upstream: object = None
    provider_supports_seed: bool = False
    task_memory_mb: int | None = None

    def __post_init__(self) -> None:
        if self.task_memory_mb is not None and (
            isinstance(self.task_memory_mb, bool)
            or not isinstance(self.task_memory_mb, int)
            or self.task_memory_mb <= 0
        ):
            raise ValueError("task_memory_mb must be a positive integer (MiB)")

    def assert_can_restart(self, directory: Path) -> None:
        state = directory / "workspace.json"
        if not state.exists():
            if any((directory / mode).exists() for mode in ("single", "team")):
                raise RuntimeError("Legacy task has no container identity; cannot verify agents stopped")
            return
        name = json.loads(state.read_text())["container_name"]
        # Listing succeeds only with a reachable daemon; absence then means removed.
        result = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^/{name}$",
                                 "--format", "{{.State}}"], capture_output=True,
                                text=True, check=True, timeout=30)
        states = result.stdout.splitlines()
        if any(state not in {"exited", "dead", "created"} for state in states):
            raise RuntimeError(f"Old task container {name} is still active; stop its agents before retry")

    def __call__(self, request: RunRequest) -> TaskResult:
        """Each call starts a new image instance, including paired test trials."""
        started = time.monotonic()
        mode = request.execution_mode
        if mode not in {"single", "team"}:
            raise ValueError("execution_mode must be single or team")
        output = request.output_dir
        output.mkdir(parents=True, exist_ok=True)
        runtime, verification, metrics = {}, {}, {}
        task = Task.load(request.task.payload["path"], repository_id=request.task.repo_id)
        if task.task_id != request.task.task_id:
            raise ValueError("Manifest task_id disagrees with task.toml")
        if not request.fresh_start:
            raise ValueError("Stage-one task comparisons require fresh task environments")
        # Preserve the official task and on-disk benchmark metadata. Every request
        # (training, replay, single or team evaluation) uses this same runner limit.
        effective_task = (replace(task, memory_mb=self.task_memory_mb)
                          if self.task_memory_mb is not None else task)
        prepared_image = request.task.payload.get("prepared_image")
        if prepared_image is not None:
            if not isinstance(prepared_image, str) or not prepared_image or any(c.isspace() for c in prepared_image):
                raise ValueError("prepared_image must be a nonempty Docker image reference")
            effective_task = replace(effective_task, image=prepared_image)
            metrics["image_config"] = {"official_image": task.image,
                                       "prepared_image": prepared_image}
        resource_config = {"official_memory_mb": task.memory_mb,
                           "memory_mb_override": self.task_memory_mb,
                           "effective_memory_mb": effective_task.memory_mb,
                           "memory_limit_scope": "task-shared"}
        package, node = self.qwen_package.resolve(), self.node_directory.resolve()
        if not (package / "cli.js").is_file() or not (node / "bin/node").is_file():
            raise FileNotFoundError("Provide the Qwen package directory and a Linux Node >=22.15 directory")
        provider = self.provider
        if request.seed is not None and self.provider_supports_seed:
            provider = replace(provider, extra_body={**provider.extra_body, "seed": request.seed})
        metrics.update(task=task.describe(), execution_mode=mode, policy_applied=mode == "team",
                       resource_config=resource_config,
                       network_policy_requested=NETWORK_POLICY,
                       requested_seed=request.seed,
                       provider_seed_applied=request.seed is not None and self.provider_supports_seed,
                       provider_seed_determinism_claimed=False,
                       scripted_provider=self.upstream is not None)
        container_name = "ac-stage1-" + uuid.uuid4().hex
        # Persist the intended name before Docker starts, including a crash during startup.
        workspace_record = {"container_name": container_name, "execution_mode": mode,
                            "resource_config": resource_config,
                            "network_policy_requested": NETWORK_POLICY}
        write_json_atomic(output / "workspace.json", workspace_record)
        workspace = DockerWorkspace(effective_task, mounts=[(package, "/opt/ac-qwen", True),
                                                  *node_runtime_mounts(node, self.platform)],
                                    platform=self.platform, container_name=container_name)
        workspace.diagnostic_secrets = (provider.api_key,) if provider.api_key else ()
        error = None
        phase = "workspace_start"
        try:
            workspace.start()
            phase = "delivery_initial_state"
            metrics["resources"] = workspace.resources
            metrics["initial_revision"] = workspace.run(["git", "rev-parse", "HEAD"]).stdout.strip()
            phase = "agent"
            runtime = QwenTeamRunner(
                "/opt/ac-qwen/cli.js", node="/opt/ac-node/bin/node", upstream=self.upstream,
                team_stall_timeout_seconds=self.team_stall_timeout_seconds,
            ).run(workspace, task.instruction,
                  request.assignment_policy if mode == "team" else "",
                  request.communication_policy if mode == "team" else "",
                  output / mode, provider, task.agent_timeout_sec, execution_mode=mode)
            phase = "delivery_capture"
            # The Docker backend has stopped all agent processes. The unchanged
            # repository filesystem is now the submitted candidate, not a worker's
            # best unmerged branch. Include ordinary newly-created files in the patch.
            workspace.run(["git", "add", "-N", "--", "."])
            patch = workspace.run(["git", "diff", "--binary", metrics["initial_revision"]])
            (output / "delivery.patch").write_text(patch.stdout, encoding="utf-8")
            metrics["delivery_revision"] = workspace.run(["git", "rev-parse", "HEAD"]).stdout.strip()
            metrics["delivery_status"] = workspace.run(["git", "status", "--porcelain"]).stdout
            if runtime.get("runtime_dir"):
                phase = "native_trace_download"
                workspace.download(runtime["runtime_dir"], output / "native_runtime")
            phase = "official_verifier"
            verification = score(task, workspace, output / "verifier")
            if runtime.get("stop_reason") not in {"native_cli_completed", "task_timeout"}:
                error = f"{mode} execution ended with {runtime.get('stop_reason')}"
            if verification.get("status") != "completed":
                error = f"Final verification ended with {verification.get('status')}"
        except Exception as exc:
            detail = exception_diagnostic(exc, workspace.diagnostic_secrets)
            metrics["failure"] = {"phase": phase, **detail}
            # Subprocess str(exc) includes command/env/prompt arguments. Preserve
            # the error type and structured outputs without serializing that command.
            description = detail.get("message") or ", ".join(
                f"{key}={detail[key]}" for key in ("timeout", "returncode") if key in detail)
            error = f"{detail['type']}: {description or 'subprocess failed'}"
        finally:
            # Capture the runtime's observed boundary before cleanup tears it
            # down. A requested policy alone is not proof enforcement succeeded.
            network_status = getattr(workspace, "network_policy_status", None)
            if isinstance(network_status, dict):
                metrics["network_policy"] = dict(network_status)
                workspace_record["network_policy"] = dict(network_status)
                try:
                    write_json_atomic(output / "workspace.json", workspace_record)
                except OSError as exc:
                    metrics["workspace_metadata_write_error"] = type(exc).__name__
                    error = error or "Task workspace metadata write failed"
            try:
                workspace.stop()
            except Exception as exc:
                metrics["cleanup_error"] = type(exc).__name__
                metrics["cleanup_diagnostic"] = exception_diagnostic(exc, workspace.diagnostic_secrets)
                error = error or "Task container cleanup failed"
            diagnostics = getattr(workspace, "diagnostics", [])
            if diagnostics:
                metrics["workspace_diagnostics"] = "workspace_diagnostics.json"
                try:
                    write_json_atomic(output / "workspace_diagnostics.json", diagnostics)
                except OSError as exc:
                    metrics["workspace_diagnostics_write_error"] = type(exc).__name__
        metrics.update({mode: runtime, "verifier": verification})
        (output / "execution.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        cost = (float(runtime["known_estimated_cost"])
                if runtime.get("cost_complete") and runtime.get("currency") == "USD" else None)
        result = TaskResult(
            status="failed" if error else "completed",
            # The official verifier grades the delivered filesystem, not the
            # agent's exit code. Keep runtime/cleanup errors without masking a
            # completed verification; an unfinished verifier is still unscored.
            score=verification.get("reward") if verification.get("status") == "completed" else None,
            metrics=metrics, usage=runtime.get("usage", {}), cost_usd=cost,
            elapsed_seconds=time.monotonic() - started, error=error,
            trace_files=[str(path.relative_to(output)) for path in sorted(output.rglob("*"))
                         if path.is_file()],
            known_cost_usd=(float(runtime["known_estimated_cost"])
                            if runtime.get("currency") == "USD"
                            and runtime.get("known_estimated_cost") is not None else None),
        )
        write_json_atomic(output / "runner_result.json", asdict(result))
        return result


# Existing scripts importing the team-only name continue to work. The request
# explicitly selects the runtime; a one-worker team is never treated as single.
RoadmapTeamRunner = RoadmapRunner
