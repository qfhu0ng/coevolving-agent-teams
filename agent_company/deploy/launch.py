"""One explicitly selected operation on a reviewed ProgramBench deployment.

Runner validation stays in the runners. This module owns cross-process launch
allocation, deployment identity, input-run locality and host capacity checks.
It neither makes policy decisions nor retries a step.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from agent_company.runtime_io import write_json_atomic
from agent_company.runtime.provider import public_provider as describe_provider
from agent_company.deploy.deployment import Deployment, capacity, verify_current
from agent_company.deploy.run_health import runtime_call_outcomes
from agent_company.deploy.ssh import clean_environment, credential
from agent_company.programbench.total_evidence import series_repeat_indices, trial_record
from agent_company.programbench.total_selection import MODE as TOTAL_MODE
from agent_company.runtime.relay import ProviderConfig, TokenRates


RUNNERS = {
    "comparison": "run_programbench_policy_comparison.py",
    "followup": "run_programbench_evolved_followup.py",
}
RUNNER_FLAGS = {
    "comparison": {"--repeat-index", "--parallelism", "--pair-id", "--batch-index"},
    "followup": {"--repeat-index", "--source-run", "--evolution-round", "--previous-round",
                 "--approved-trial", "--history-trial"},
}
MODES = ("off", "global", "master", "propose")
FIRST_ROUND = {mode: index for index, mode in enumerate(MODES, 1)}
STEPS = ("s0_propose", "s0_global", "first_round", "role_series", "agent_rules", "confirm", "approved", "followup")
SERIES_STEPS = ("role_series", "agent_rules")
# The user-approved starting master strategy of agent_rules_v1 (plan v4 M4).
AGENT_RULES_M0 = "configs/stage1/policies/agent_rules_m0_20261017"
ARMS = ("single", *(f"team_p0_{mode}" for mode in MODES))
MAX_PARALLEL_ARMS = 2
RUNNING_STATES = {"running", "preparing", "optimizing", "reviewing", "running_evolved"}
TERMINAL_STATES = {"completed", "blocked_executability", "blocked_official_score",
                   "blocked_proposal", "blocked_provider", "blocked_review"}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {Path(path).name}")
    return value


def arguments(spec):
    """Keep each explicit runner argument value unchanged; reject other flags."""
    runner = spec.get("runner")
    if runner not in RUNNERS:
        raise ValueError("Select comparison or followup")
    raw = list(spec.get("arguments", ()))
    pairs = []
    position = 0
    while position < len(raw):
        token = raw[position]
        flag, equals, inline = token.partition("=")
        if flag not in RUNNER_FLAGS[runner]:
            raise ValueError("Runner argument is outside the explicit allowlist")
        if equals:
            value = inline
        else:
            position += 1
            if position >= len(raw):
                raise ValueError("Runner argument needs a value")
            value = raw[position]
        if not isinstance(value, str) or not value:
            raise ValueError("Runner argument needs a nonempty value")
        pairs.append((flag, value))
        position += 1
    for flag in RUNNER_FLAGS[runner] - {"--previous-round", "--approved-trial", "--history-trial"}:
        if sum(key == flag for key, _ in pairs) > 1:
            raise ValueError("Singleton runner argument repeated")
    return raw, pairs


def input_paths(deployment, pairs):
    """Scope only; source/history/chain validity remains a runner responsibility."""
    root = (deployment.final / "paid_runs").resolve()
    paths = []
    for flag, value in pairs:
        if flag not in {"--source-run", "--previous-round", "--approved-trial", "--history-trial"}:
            continue
        if flag == "--approved-trial":
            if "=" not in value:
                raise ValueError("approved-trial needs ROUND=PATH")
            value = value.split("=", 1)[1]
        path = Path(value).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Every input run must be within this deployment's paid_runs")
        paths.append(str(path))
    return paths


def runner_plan(deployment, spec, raw):
    command = [str(deployment.python), "-u", str(deployment.code / "scripts" / RUNNERS[spec["runner"]]),
               "--profile", spec["profile"], "--project-root", str(deployment.code), *raw]
    result = subprocess.run(command, cwd=deployment.code,
                            env={**clean_environment(), "PYTHONDONTWRITEBYTECODE": "1",
                                 "PYTHONPATH": f"{deployment.evaluator}:{deployment.code}"},
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError("Runner read-only planning failed; no credentials read or output created\n"
                         + result.stderr[-800:])
    plan = json.loads(result.stdout)
    if not isinstance(plan, dict):
        raise ValueError("Runner returned no readable plan")
    return command, plan


def public_provider(path):
    data = read_json(path)
    if "api_key" in data:
        raise ValueError("Provider configuration must not contain a credential")
    data.pop("api_key_env", None)
    if data.get("rates") is not None:
        data["rates"] = TokenRates(**data["rates"])
    return describe_provider(ProviderConfig(**data))


def confirmation_inputs(deployment, spec, plan):
    if not spec.get("trigger_report"):
        raise ValueError("Confirmation needs an explicitly saved trigger report")
    trigger = read_json(spec["trigger_report"])
    if trigger.get("decision") != "needs_confirmation":
        raise ValueError("Saved report did not trigger confirmation")
    records = [trial_record(trigger[key]) for key in ("incumbent_trial", "candidate_trial")]
    root = (deployment.final / "paid_runs").resolve()
    if any(not row["path"].is_relative_to(root) for row in records):
        raise ValueError("Trigger trials must be in this deployment's paid_runs")
    arms, tasks = plan["arms"], plan["tasks"]
    if (len(tasks) != 1 or len(arms) != 2 or plan["parallelism"] != 2
            or len(plan["waves"]) != 1 or len(plan["waves"][0]) != 2):
        raise ValueError("Confirmation requires exactly two arms in one concurrent wave")
    provider = public_provider(plan["provider_config"])
    matched = []
    for arm, old in zip(arms, records):
        condition = old["condition"]
        inference = dict(plan["inference"])
        role_mode = inference.pop("role_policy_mode", "off") if arm["execution_mode"] == "team" else None
        if arm["execution_mode"] != "team":
            inference.pop("role_policy_mode", None)
        policies = tuple(Path(arm[key]).read_bytes().decode("utf-8") if arm.get(key) else ""
                         for key in ("assignment_policy_path", "communication_policy_path"))
        wanted = (condition["assignment_policy"], condition["communication_policy"])
        if (tasks[0]["task_id"] != old["shared"]["task_id"]
                or tasks[0]["repository_id"] != old["shared"]["repo_id"]
                or policies != wanted
                or arm["execution_mode"] != condition["execution_mode"]
                or role_mode != condition["role_policy_mode"]
                or arm["policy_version"] != condition["policy_version"]
                or inference != old["shared"]["inference"]
                or plan["evaluation"] != old["shared"]["evaluation"]
                or provider != old["shared"]["provider"]):
            raise ValueError("Confirmation profile differs from its triggering arm conditions")
        matched.append(str(old["path"]))
    return matched


def approved_input(deployment, spec, plan):
    """Bind a separately authorized dispatch to the saved, reviewed proposal."""
    if not spec.get("paused_round"):
        raise ValueError("Approved dispatch needs an explicit paused-round")
    source = Path(spec["paused_round"]).resolve(strict=True)
    root = (deployment.final / "paid_runs").resolve()
    if not source.is_relative_to(root) or source == root:
        raise ValueError("Paused round must be within this deployment's paid_runs")
    progress = read_json(source / "progress.json")
    review = read_json(source / "review_summary.json")
    if progress.get("state") != "blocked_executability" or review.get("status") != "completed":
        raise ValueError("Approved dispatch needs a blocked_executability round with completed review")
    saved = read_json(source / "run_settings.json")
    proposal = read_json(source / "proposal_summary.json")
    teams = [arm for arm in saved.get("arms", []) if arm.get("execution_mode") == "team"]
    if (len(teams) != 1 or len(plan["arms"]) != 1 or plan["parallelism"] != 1
            or plan["arms"][0]["execution_mode"] != "team"):
        raise ValueError("Approved dispatch requires exactly one Team arm with parallelism 1")
    arm = plan["arms"][0]
    for key in ("assignment_policy", "communication_policy"):
        policy = arm.get(key + "_path")
        if not policy or Path(policy).read_bytes().decode("utf-8") != proposal.get(key):
            raise ValueError("Approved dispatch policies must exactly match the saved proposal")
    if (arm["policy_version"] != teams[0].get("policy_version")
            or plan["inference"] != saved.get("inference")
            or plan["evaluation"] != saved.get("evaluation")
            or public_provider(plan["provider_config"]) != saved.get("provider")):
        raise ValueError("Approved dispatch conditions differ from its paused round")
    return str(source)


def pid_alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.is_file() or stat.read_text().rsplit(")", 1)[1].strip().split()[0] != "Z"
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def active_runner_processes(deployment):
    """Inspect this launch user's processes in this deployment; never save argv/env."""
    if not Path("/proc").is_dir():
        raise RuntimeError("Linux process inventory is unavailable")
    markers = {str(deployment.code / "scripts" / name) for name in
               (*RUNNERS.values(), "s0_native_smoke.py", "s0_cache_probe.py")}
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            # Read cmdline before cwd: unrelated same-UID processes can have
            # unreadable cwd links, but they cannot be this deployment's runner.
            if process.stat().st_uid != os.getuid():
                continue
            argv = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not markers.intersection(argv):
                continue
            if not (process / "cwd").resolve(strict=True).is_relative_to(deployment.final.resolve()):
                continue
            if pid_alive(int(process.name)):
                found.append(int(process.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            raise RuntimeError("Cannot determine whether this deployment has an active runner") from None
    return sorted(found)


def refused_before_output(directory, record, alive):
    return (record.get("status") == "started" and type(record.get("pid")) is int
            and record["pid"] > 0 and not alive and not (directory / "run").exists())


def paid_records(deployment):
    root = deployment.final / "paid_runs"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        launch = read_json(path / "launch.json")
        progress_path = path / "run/progress.json"
        progress = read_json(progress_path) if progress_path.is_file() else None
        alive = pid_alive(launch.get("pid"))
        refused = refused_before_output(path, launch, alive)
        if progress is None and not alive and not refused:
            raise ValueError("Previous launch has no live process or progress; inspect before continuing")
        if progress is not None and (progress.get("cleanup_attention") != []
                                     or progress.get("state") == "blocked_cleanup"):
            raise ValueError("Previous inference cleanup is not confirmed")
        terminal = refused or (progress is not None and not alive and progress.get("running") == []
                               and progress.get("state") in TERMINAL_STATES)
        rows.append({"directory": str(path), "launch": launch, "progress": progress,
                     "pid_alive": alive, "terminal": terminal,
                     "refused_before_output": refused})
    return rows


def reserved_arms(plan):
    """Reserve the runner's maximum concurrent arms, including its startup gap."""
    parallelism = plan.get("parallelism")
    if type(parallelism) is not int or not 1 <= parallelism <= MAX_PARALLEL_ARMS:
        raise ValueError("Runner concurrency must be one or two arms")
    count = len(plan["tasks"]) * len(plan["arms"])
    if count < 1:
        raise ValueError("Runner must plan at least one arm")
    return min(count, parallelism)


def active_arm_slots(rows, active):
    """A live runner retains its allocation until exit; unknown is not zero."""
    count, known = 0, True
    allocated_pids = set()
    for row in rows:
        if row["terminal"]:
            continue
        record, progress = row["launch"], row["progress"]
        if not row["pid_alive"]:
            known = False
            continue
        allocated_pids.add(record.get("pid"))
        size = record.get("reserved_arm_count")
        if (record.get("status") != "started" or type(size) is not int
                or not 1 <= size <= MAX_PARALLEL_ARMS):
            known = False
            continue
        if progress is not None:
            state = progress.get("state")
            if state not in RUNNING_STATES | TERMINAL_STATES:
                known = False
            if not isinstance(progress.get("running"), list):
                known = False
        count += size
    # An unallocated runner could be a smoke/probe or an untracked launch.
    # Neither can be safely assigned a guessed arm count.
    known = known and set(active).issubset(allocated_pids)
    return count, known


def runtime_conditions(deployment, plan, step, index):
    rows = paid_records(deployment)
    active = active_runner_processes(deployment)
    occupied, known = active_arm_slots(rows, active)
    requested = reserved_arms(plan)
    conditions = {"known_execution_state": known,
                  "arm_slots_available": known and occupied + requested <= MAX_PARALLEL_ARMS}
    host = capacity(deployment)
    memory = plan["inference"].get("memory_mb")
    if type(memory) is not int or memory <= 0:
        raise ValueError("New arm memory requirement is unknown")
    required_kb = requested * memory * 1024
    available = host.get("mem_available_kb")
    conditions["concurrent_memory_available"] = type(available) is int and available >= required_kb
    return {"conditions": conditions, "satisfied": all(conditions.values()),
            "active_runner_pids": active, "active_arm_count": occupied if known else None,
            "new_arm_count": requested, "max_parallel_arms": MAX_PARALLEL_ARMS,
            "required_memory_kb": required_kb, "capacity": host}


def _s0_contract(deployment, plan, step):
    from agent_company.deploy.s0_contract import POLICY, POLICY_SHA256, inference
    mode = step.removeprefix("s0_")
    if (plan["inference"] != inference(mode) or plan["new_rollouts"] != 1
            or plan["parallelism"] != 1 or plan["repeat_index"] != 0
            or [(a["arm_id"], a["execution_mode"], a["policy_version"]) for a in plan["arms"]]
            != [(f"team_p0_{mode}", "team", 0)]):
        raise ValueError("S0 must use its one fresh P0 Team, fixed inference and repeat_index 0")
    for key, name in zip(("assignment_policy_path", "communication_policy_path"), POLICY_SHA256):
        path = Path(plan["arms"][0][key])
        if path.resolve() != (deployment.code / POLICY / name).resolve() or hashlib.sha256(
                path.read_bytes()).hexdigest() != POLICY_SHA256[name]:
            raise ValueError("S0 must use the saved fresh P0 policy text")


def _role_series_parent(deployment, plan):
    """Bind batch 2 to the P0 actually handed to batch 1's native runners."""
    from agent_company.policy.role_package import ROLE_POLICY_PROTOCOL
    root = deployment.final / "paid_runs/role_series_batch1_p0/run"
    settings = read_json(root / "run_settings.json")
    if (settings.get("batch_index") != 1
            or settings.get("inference", {}).get("role_policy_protocol") != ROLE_POLICY_PROTOCOL):
        raise ValueError("Batch 2 requires the recorded first role-series batch")
    arms = settings.get("arms", [])
    if [(a.get("execution_mode"), a.get("role_policy_mode"), a.get("policy_version"))
            for a in arms] != [("single", None, 0), ("team", "propose", 0)]:
        raise ValueError("Batch 1 actual arms do not identify the prescribed P0 inputs")
    p0 = plan["arms"][1]["policy_package"]
    task_id = plan["tasks"][0]["task_id"]
    for arm in arms:
        if arm.get("policy_package") != p0:
            raise ValueError("Batch 2 off P0 differs from the first batch's P0 package")
        trial = root / task_id / arm["arm_id"]
        request = read_json(trial / "run_request.json")
        native = read_json(trial / arm["execution_mode"] / "launch.json")
        if (request.get("policy_package") != p0
                or request.get("execution_mode") != arm["execution_mode"]
                or request.get("policy_version") != 0
                or request.get("role_policy_mode") != arm.get("role_policy_mode")
                or native.get("policy_package") != p0
                or native.get("role_policy_mode") != arm.get("role_policy_mode")
                or native.get("role_policy_protocol") != ROLE_POLICY_PROTOCOL):
            raise ValueError("Batch 1 recorded package disagrees with its actual native launch")
    return str(root)


def _agent_rules_batch1(deployment, plan, approved):
    """The recorded agent_rules_v1 batch 1 that batch 2 is compared with; it may live in an earlier deployment."""
    from agent_company.policy.role_package import AGENT_RULES_PROTOCOL
    # Batch 1 records of other tasks (user, 2026-10-19: ctags beside Pandoc) are not candidates.
    task_ids = [t["task_id"] for t in plan["tasks"]]
    found = [path for path in sorted(deployment.final.parent.glob("*/paid_runs/agent_rules_batch1_m0/run/run_settings.json"))
             if [t.get("task_id") for t in read_json(path).get("tasks", [])] == task_ids]
    if len(found) != 1:
        raise ValueError("Batch 2 requires exactly one recorded agent_rules_v1 batch 1 of its task beside this deployment")
    settings = read_json(found[0])
    progress = read_json(found[0].parent / "progress.json")
    if progress.get("state") != "completed" or progress.get("cleanup_attention"):
        raise ValueError("The recorded batch 1 has not finished cleanly")
    arms = settings.get("arms", [])
    if (settings.get("batch_index") != 1
            or settings.get("inference", {}).get("role_policy_protocol") != AGENT_RULES_PROTOCOL
            or [(a.get("execution_mode"), a.get("role_policy_mode"), a.get("policy_version")) for a in arms]
            != [("single", None, 0), ("team", "propose", 0)]
            or arms[1].get("policy_package", {}).get("team") != approved):
        raise ValueError("The recorded batch 1 is not Single and the approved M0 Team")
    # Batch 2 is compared with batch 1's M0 Team, so everything but the master strategy must match it.
    if (plan["inference"] != settings.get("inference") or plan["evaluation"] != settings.get("evaluation")
            or plan.get("selection_protocol") != settings.get("selection_protocol")
            or [(t["task_id"], t["repository_id"]) for t in plan["tasks"]]
            != [(t.get("task_id"), t.get("repository_id")) for t in settings.get("tasks", [])]
            or public_provider(Path(plan["provider_config"])) != settings.get("provider")):
        raise ValueError("Batch 2 must use batch 1's task, inference, evaluation, selection and provider")
    from agent_company.evolution.harness_capabilities import native_version

    def instructions(task):
        return hashlib.sha256(Path(task["instructions"]).read_bytes()).hexdigest()
    try:
        same_task = ([(t["image"], instructions(t)) for t in plan["tasks"]]
                     == [(t["image"], instructions(t)) for t in settings["tasks"]])
    except (OSError, KeyError):
        same_task = False
    if not same_task or native_version(deployment.qwen) != settings.get("qwen_version"):
        raise ValueError("Batch 2 must use batch 1's task image, instruction text and Qwen version")
    result = read_json(found[0].parent / plan["tasks"][0]["task_id"] / arms[1]["arm_id"] / "task_result.json")
    if result.get("status") != "completed" or not isinstance(result.get("score"), (int, float)):
        raise ValueError("The recorded batch 1 M0 Team has no completed official result")
    return str(found[0].parent), arms[1]["policy_package"]


def _agent_rules_batch(deployment, plan, profile, profile_data, pairs, index, inputs):
    """agent_rules_v1 batch 1: Single and the approved M0 Team, concurrently, both given the M0 package.
    Batch 2: the reviewed candidate M1 Team alone, under batch 1's recorded conditions (user, 2026-10-19)."""
    from agent_company.policy.role_package import AGENT_RULES_PROTOCOL
    batches = [value for flag, value in pairs if flag == "--batch-index"]
    if len(batches) != 1 or int(batches[0]) not in (1, 2) or index != int(batches[0]):
        raise ValueError("The agent_rules_v1 launch is batch 1 or 2, with repeat-index equal to the batch")
    batch = int(batches[0])
    inputs_root = (deployment.final / "profiles").resolve()
    if (not profile.is_relative_to(inputs_root) or plan.get("batch_index") != batch or "first_round_wave" in profile_data
            or plan.get("selection_protocol") != TOTAL_MODE
            or plan["inference"].get("role_policy_protocol") != AGENT_RULES_PROTOCOL):
        raise ValueError("The agent_rules_v1 profile belongs in profiles/ with its batch, TOTAL and agent_rules_v1")
    arms_expected = 2 if batch == 1 else 1
    if (plan["parallelism"] != arms_expected or len(plan["arms"]) != arms_expected
            or len(plan["waves"]) != 1 or len(plan["waves"][0]) != arms_expected):
        raise ValueError("The agent_rules_v1 batch requires its prescribed concurrent arms")
    m0 = deployment.code / AGENT_RULES_M0
    approved = {part: (m0 / f"{part}.md").read_bytes().decode("utf-8") for part in ("assignment", "communication")}
    for arm in plan["arms"]:
        if not Path(arm["policy_package_path"]).resolve().is_relative_to(inputs_root):
            raise ValueError("agent_rules_v1 policy packages belong in this deployment's profiles/")
    if batch == 1:
        if any(arm["policy_package"]["team"] != approved for arm in plan["arms"]):
            raise ValueError("Both arms must carry the approved M0 texts")
        return "agent_rules_batch1_m0"
    run, incumbent = _agent_rules_batch1(deployment, plan, approved)
    arm = plan["arms"][0]
    candidate = arm["policy_package"]
    review_ref = candidate.get("review_ref")
    review = (Path(arm["policy_package_path"]).resolve().parent / str(review_ref or "")).resolve()
    if (candidate["parent"] != {"package_id": incumbent["package_id"], "revision": incumbent["revision"]}
            or not isinstance(review_ref, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", review_ref)
            or not review.is_file() or not review.is_relative_to(inputs_root)
            or read_json(review).get("status") != "completed" or candidate["team"] == incumbent["team"]):
        raise ValueError("The batch 2 candidate must be a reviewed revision of batch 1's M0, with its completed "
                         "review beside it in profiles/ and new texts")
    inputs.append(run)
    return "agent_rules_batch2_m1"


def prepare(deployment: Deployment, spec: dict):
    raw, pairs = arguments(spec)
    repeat = [value for flag, value in pairs if flag == "--repeat-index"]
    if len(repeat) != 1:
        raise ValueError("Explicit repeat-index is required")
    index = int(repeat[0])
    step = spec.get("step")
    if step not in STEPS or (step == "followup") != (spec["runner"] == "followup"):
        raise ValueError("Step and selected runner disagree")
    if (step.startswith("s0_") and index != 0) or (not step.startswith("s0_") and index < 1):
        raise ValueError("S0 index is 0; every other step requires a positive index")
    inputs = input_paths(deployment, pairs)
    command, plan = runner_plan(deployment, spec, raw)
    if len(plan["tasks"]) != 1 or plan["tasks"][0]["task_id"] != deployment.task_id:
        raise ValueError("Plan must use this deployment's selected task")
    mode = plan["inference"].get("role_policy_mode", "off")
    if mode not in MODES:
        raise ValueError("Unknown Team mode")
    profile = Path(spec["profile"]).resolve(strict=True)
    profile_data = read_json(profile)
    if step.startswith("s0_"):
        committed = deployment.code / f"configs/stage1/s0_deepseek_{step.removeprefix('s0_')}.json"
        if profile.read_bytes() != committed.read_bytes():
            raise ValueError("S0 profile must equal the committed profile byte for byte")
        _s0_contract(deployment, plan, step)
        name = step
    elif step == "first_round":
        if index != FIRST_ROUND[mode] or profile_data.get("first_round_wave") != index:
            raise ValueError("First-round index must equal its fixed wave")
        committed = deployment.code / f"configs/stage1/first_round_deepseek_pandoc_{mode}.json"
        if profile.read_bytes() != committed.read_bytes():
            raise ValueError("First-round profile must equal the committed profile byte for byte")
        expected_arms = 2 if mode == "propose" else 1
        if (plan["parallelism"] != expected_arms or len(plan["arms"]) != expected_arms
                or len(plan["waves"]) != 1 or len(plan["waves"][0]) != expected_arms):
            raise ValueError("First-round effective concurrency must match its prescribed wave")
        name = f"wave{index}_{mode}_r{index}"
    elif step == "role_series":
        from agent_company.policy.role_package import ROLE_POLICY_PROTOCOL
        batches = [value for flag, value in pairs if flag == "--batch-index"]
        if len(batches) != 1 or int(batches[0]) not in (1, 2):
            raise ValueError("Role-series launch requires explicit batch-index 1 or 2")
        batch = int(batches[0])
        inputs_root = (deployment.final / "profiles").resolve()
        if (not profile.is_relative_to(inputs_root) or plan.get("batch_index") != batch
                or index != batch or "first_round_wave" in profile_data
                or plan.get("selection_protocol") != TOTAL_MODE
                or plan["inference"].get("role_policy_protocol") != ROLE_POLICY_PROTOCOL):
            raise ValueError("Role-series profile belongs in profiles/ and must match its explicit batch and repeat index")
        if (plan["parallelism"] != 2 or len(plan["arms"]) != 2
                or len(plan["waves"]) != 1 or len(plan["waves"][0]) != 2):
            raise ValueError("Role-series batch requires its two concurrent arms")
        for arm in plan["arms"]:
            if not Path(arm["policy_package_path"]).resolve().is_relative_to(inputs_root):
                raise ValueError("Role-series policy packages belong in this deployment's profiles/")
        if batch == 2:
            inputs.append(_role_series_parent(deployment, plan))
        name = "role_series_batch1_p0" if batch == 1 else "role_series_batch2_v1_p0"
    elif step == "agent_rules":
        name = _agent_rules_batch(deployment, plan, profile, profile_data, pairs, index, inputs)
    else:
        if (not profile.is_relative_to((deployment.final / "profiles").resolve())
                or profile_data.get("selection_protocol") != TOTAL_MODE or "first_round_wave" in profile_data):
            raise ValueError("New confirmation/followup/approved profiles belong in profiles/ with TOTAL and no first_round_wave")
        name = f"{step}_{mode}_r{index}"
    if step not in SERIES_STEPS and any(flag == "--batch-index" for flag, _ in pairs):
        raise ValueError("batch-index is only for the role-series and agent_rules steps")
    if step == "confirm":
        inputs.extend(confirmation_inputs(deployment, spec, plan))
    elif spec.get("trigger_report"):
        raise ValueError("trigger-report is only for confirmation")
    if step == "approved":
        inputs.append(approved_input(deployment, spec, plan))
    elif spec.get("paused_round"):
        raise ValueError("paused-round is only for approved dispatch")
    output = deployment.final / "paid_runs" / name
    if output.exists():
        raise FileExistsError("This step was already allocated; never retry or reuse it")
    series = deployment.final / "paid_runs"
    if index > 0 and series.is_dir() and index in series_repeat_indices(series):
        raise ValueError("repeat_index was already allocated in this deployment")
    identity = verify_current(deployment)
    runtime = runtime_conditions(deployment, plan, step, index)
    command += ["--run", "--qwen-package", str(deployment.qwen), "--node-directory", str(deployment.node),
                "--docker", str(deployment.docker), "--output", str(output / "run")]
    return {"status": "prepared_not_launched", "step": step, "repeat_index": index, "name": name,
            **({"batch_index": plan["batch_index"]} if step in SERIES_STEPS else {}),
            "directory": str(output), "command": command, "cwd": str(deployment.code),
            "pythonpath": f"{deployment.evaluator}:{deployment.code}", "plan": plan,
            "manifest_sha256": identity["manifest_sha256"],
            "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(), "input_runs": inputs,
            **({"policy_package_sha256_by_arm": {
                arm["arm_id"]: hashlib.sha256(Path(arm["policy_package_path"]).read_bytes()).hexdigest()
                for arm in plan["arms"]}} if step in SERIES_STEPS else {}),
            **({"policy_review_sha256_by_arm": {
                arm["arm_id"]: hashlib.sha256((Path(arm["policy_package_path"]).resolve().parent
                                               / arm["policy_package"]["review_ref"]).read_bytes()).hexdigest()
                for arm in plan["arms"] if arm["policy_package"].get("review_ref")}}
               if step == "agent_rules" and plan.get("batch_index") == 2 else {}),
            **({"trigger_report": str(Path(spec["trigger_report"]).resolve()),
                "trigger_report_sha256": hashlib.sha256(Path(spec["trigger_report"]).read_bytes()).hexdigest()}
               if step == "confirm" else {}),
            "runtime": runtime,
            "followup_runtime_validation": "runner_before_output_and_provider; not established by prepare"
            if spec["runner"] == "followup" else None}


@contextmanager
def allocation_lock(deployment):
    # Two different step names could otherwise pass the slot/index checks at
    # once. Lock the existing directory, creating no preflight file or output.
    descriptor = os.open(deployment.final, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def launch(deployment, spec, payload):
    with allocation_lock(deployment):
        prepared = prepare(deployment, spec)
        if not prepared["runtime"]["satisfied"]:
            raise ValueError("Timing or capacity conditions are not yet satisfied")
        key = credential(payload)
        root = Path(prepared["directory"])
        root.parent.mkdir(exist_ok=True)
        root.mkdir(exist_ok=False)
        record = {key: prepared[key] for key in ("step", "repeat_index", "name", "command", "cwd",
                  "pythonpath", "manifest_sha256", "profile_sha256", "input_runs")}
        if "batch_index" in prepared:
            record["batch_index"] = prepared["batch_index"]
            record["policy_package_sha256_by_arm"] = prepared["policy_package_sha256_by_arm"]
        if "policy_review_sha256_by_arm" in prepared:
            record["policy_review_sha256_by_arm"] = prepared["policy_review_sha256_by_arm"]
        if "trigger_report" in prepared:
            record.update(trigger_report=prepared["trigger_report"],
                          trigger_report_sha256=prepared["trigger_report_sha256"])
        record.update(status="launching", created_at_utc=datetime.now(timezone.utc).isoformat(),
                      capacity=prepared["runtime"]["capacity"], pid=None,
                      reserved_arm_count=prepared["runtime"]["new_arm_count"])
        with (root / "launch.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        try:
            with (root / "controller.log").open("xb") as log:
                process = subprocess.Popen(prepared["command"], cwd=prepared["cwd"],
                    env={**clean_environment(), "DEEPSEEK_API_KEY": key,
                         "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": prepared["pythonpath"]},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as exc:
            raise RuntimeError(f"Launch outcome unclear ({type(exc).__name__}); inspect, do not retry") from None
        record.update(status="started", pid=process.pid)
        write_json_atomic(root / "launch.json", record)
        return record


def status(deployment):
    rows = []
    root = deployment.final / "paid_runs"
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir():
            continue
        record = read_json(directory / "launch.json")
        progress = directory / "run/progress.json"
        alive = pid_alive(record.get("pid"))
        refused = refused_before_output(directory, record, alive)
        official = []
        for path in (directory / "run").glob("*/*/task_result.json"):
            value = read_json(path)
            verifier = value.get("metrics", {}).get("verifier", {})
            official.append({"trial": str(path.parent), "status": verifier.get("status"),
                             "score": value.get("score"), "reward": verifier.get("reward")})
        rows.append({"name": directory.name, "pid": record.get("pid"),
                     "pid_alive": alive,
                     "state": "refused_before_output" if refused else record.get("status"),
                     "refused_before_output": refused,
                     "progress": read_json(progress) if progress.is_file() else None,
                     "official_results": official})
    return {"runs": rows, "active_runner_pids": active_runner_processes(deployment)}


def preparation(deployment, *, kind, arm=None, sequence, payload=None, prepare_only=False):
    if kind not in {"smoke_scripted", "smoke_real", "cache_probe"}:
        raise ValueError("Unknown preparation operation")
    if type(sequence) is not int or sequence < 1:
        raise ValueError("Preparation needs an explicit positive sequence")
    if kind == "cache_probe":
        arm = "team_p0_propose"
        profile_name = "s0_deepseek_propose.json"
    else:
        if arm not in ARMS:
            raise ValueError("Select one supported arm")
        mode = "propose" if arm == "single" else arm.removeprefix("team_p0_")
        profile_name = f"first_round_deepseek_pandoc_{mode}.json"
    output = deployment.final / "preparations" / f"{kind}_{arm}_{sequence}"
    with allocation_lock(deployment):
        if output.exists():
            raise FileExistsError("Preparation output exists; do not repeat this step")
        verify_current(deployment)
        idle = not active_runner_processes(deployment) and all(
            row["terminal"] for row in paid_records(deployment))
        profile = deployment.code / "configs/stage1" / profile_name
        if not profile.is_file():
            raise FileNotFoundError("Committed preparation profile missing")
        command = [str(deployment.python), "-u", str(deployment.code / "scripts" /
                   ("s0_cache_probe.py" if kind == "cache_probe" else "s0_native_smoke.py")),
                   "--profile", str(profile), "--output", str(output)]
        if kind != "cache_probe":
            command += ["--arm", arm, "--assets-root", str(deployment.assets),
                        "--scripted" if kind == "smoke_scripted" else "--real",
                        "--max-calls", "40", "--timeout", "420"]
        if prepare_only:
            return {"status": "prepared_not_launched", "kind": kind, "arm": arm,
                    "sequence": sequence, "output": str(output), "command": command,
                    "runtime": {"satisfied": idle, "conditions": {"preparation_exclusive": idle}}}
        if not idle:
            raise ValueError("Previous operation is still running or its termination is unconfirmed")
        key = credential(payload) if kind != "smoke_scripted" else None
        env = {**clean_environment(), "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": f"{deployment.evaluator}:{deployment.code}"}
        if key:
            env["DEEPSEEK_API_KEY"] = key
        try:
            result = subprocess.run(command, cwd=deployment.code, env=env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=1900)
        except Exception as exc:
            raise RuntimeError(f"Preparation outcome unclear ({type(exc).__name__}); inspect, do not retry") from None
        stdout, stderr = result.stdout, result.stderr
        if key:
            stdout, stderr = stdout.replace(key, "[REDACTED]"), stderr.replace(key, "[REDACTED]")
        report = {"kind": kind, "arm": arm, "output": str(output), "returncode": result.returncode,
                  "stdout": stdout, "stderr": stderr}
        if kind == "smoke_real":
            # The native smoke's wiring checks alone do not establish that each
            # logical provider call eventually succeeded. A replacement worker
            # is a new call, not a retry that repairs its predecessor's usage.
            assessment = runtime_call_outcomes(output / "runtime")
            report["logical_call_assessment"] = assessment
            if not assessment["passed"]:
                report["returncode"] = result.returncode or 1
        if key:
            # Include newly added structured diagnostics in the existing
            # credential-redaction boundary, not only subprocess text.
            report = json.loads(json.dumps(report).replace(key, "[REDACTED]"))
        return report
