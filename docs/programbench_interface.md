# ProgramBench interface

This is the ProgramBench adapter for the existing Qwen Single / frozen-P3 Team
architecture. It does not use the Roadmap task loader or its in-container verifier,
and it is not a reproduction of the official mini-SWE runtime. Implemented on
2026-09-23; implementing this interface does **not** launch the authorized model pair.

## Contract

```python
from pathlib import Path
from agent_company.evolution.loop import RunRequest, TaskSpec
from agent_company.programbench.runner import ProgramBenchRunner

runner = ProgramBenchRunner(provider, Path(qwen_package), Path(linux_node_directory))
request = RunRequest(
    task=TaskSpec(task_id, repository_id, {
        "path": str(public_task_directory),
        "instruction_path": str(public_instruction_file),
    }),
    split="integration", policy_version=3,
    assignment_policy=assignment_P3, communication_policy=communication_P3,
    output_dir=Path(new_output_directory), execution_mode="team",
)
result = runner(request)  # TaskResult: execution status and official score separately
```

`TaskResult.score` is the official filtered **test fraction**, not the fraction of
completed tasks. `metrics.task_success` / `metrics.verifier.resolved` records whether
the task was fully solved; `almost` records the official near-solved threshold.
A program that does not implement the target can still pass some checks, so a high
partial fraction alone is not evidence of successful reimplementation.

Single uses `execution_mode="single"` and receives empty A/C even if the caller
accidentally supplies policies. Team requires both original policy texts. Each call
uses a fresh container, source tree and Qwen session. There is no evolution or
automatic whole-task retry in this adapter.

| Layer | Implemented behavior |
|---|---|
| Public task input | `task.yaml` + explicit public instructions; no tests/solutions loaded into prompts |
| Inference | Official `task_cleanroom_v6` image, non-root `agent`, 20 CPU / 60 GiB / no extra swap, 6 h |
| Team allocation | All agents share the task's resources; historical capped profiles share 1000 logical relay requests, while no-call-cap profiles use `max_model_calls: null`; HTTP retries do not consume another logical call |
| Reference | Execute-only `/opt/ac-programbench-reference/executable`, outside source and worker Git trees |
| Agent network | Only the exact host-local model relay endpoint; no internet, host filesystem, or Docker socket |
| Submission | Full `/workspace/solution` source, including untracked files; root `compile.sh` rebuilds `./executable` |
| Official evaluator | ProgramBench 1.2.4, separate containers, default 10 CPU, compile 900 s / branch 3600 s |
| Score | Official active-branch/ignored-test filtering; fraction + Resolved + Almost; no valid reward stays unknown |

The Team step mapping is our extension of the official single-agent baseline,
not an official multiagent protocol. Qwen's tool semantics, shell-command limits,
team inactivity behavior and model/retry configuration remain different from
mini-SWE. In particular, the official config's 180-second shell-command timeout
and 7-hour container lifetime are not separately reproduced; the shared 6-hour
agent deadline and owned-container cleanup apply. No claim of full official-runtime
equivalence is made.

## Tracked profiles and policy inputs

The reusable profiles no longer depend on local experiment output or private
working-document directories for their policy texts:

| Profile in `configs/stage1/` | Arms | Logical model-call limit |
|---|---|---|
| `programbench_glm_p3.json` | Historical gron Single/P3 pair | 1000 per arm |
| `programbench_fd_glm_p3.json` | Historical fd Single/P3 pair | 1000 per arm |
| `programbench_fd_no_call_cap_glm_p3.json` | fd Single/P3 uncapped variant | None; metering retained |
| `programbench_ac_pb1_heldout_20260924.json` | sd/lz4 P3/PB1 comparison, four Team runs | None; metering retained |

All four retain the six-hour task deadline, 20 CPU quota and 60 GiB memory per
arm. Removing the logical call cap does not remove request/output limits or the
task deadline, and does not make this the official mini-SWE runtime. The provider
profile contains only an environment-variable reference for its credential;
no inline API key or known billing rate is committed.

`configs/stage1/policies/scorefix_p3/` contains the unchanged P3 assignment and
communication texts. `configs/stage1/policies/programbench_pb1/` contains the
unchanged experimental PB1 texts. `configs/stage1/programbench_instruction.md`
contains the public adapted instruction shared by the comparison. Original
local policy and experiment artifacts are preserved separately.

**PB1 is not adopted.** The sd/lz4 comparison was stopped by the user for API
maintenance on 2026-09-24; sd's two arms were interrupted and lz4 was not started.
The profile's status fields record that historical stop, not an automatic resume
capability or a runtime launch gate. Do not restart or evaluate it without new
authorization. These tracked inputs are not raw trajectories or model rationale.

## Source and scoring lifecycle

Public inputs → fresh cleanroom → Single/Team → terminate all agent processes →
export source → remove inference container and network helper → official evaluator.

Workers start in their own Git worktrees. They must integrate into the delivery
root; exporting all workers indiscriminately would mix alternative answers.
The reference binary, `.git`, and root build output `executable` are not submitted.
The current archive transport rejects symlinks and special files instead of
following them into host/reference material. A missing `compile.sh` is a submission
failure, not an invented official zero. Officially counted compilation failures
can retain the evaluator's valid zero. CLI errors/OOM/timeouts remain execution
failures even when official evaluation produces a valid score.

`runner_result.json` / `execution.json` contain compact whitelisted metrics.
`verifier/result.json` contains official aggregate results. Native trajectories and
official detailed logs remain separate artifacts; do not print their reasoning,
credentials, or hidden test content when inspecting the result. Unknown usage or
cost is not filled with zero; fixed scripted canary usage is not real model usage.

## CLI

Install the optional evaluator dependency on the Linux host:

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[programbench]'
```

Prepare instructions from the public official mini-SWE config (strips only its
runtime-specific command format/sentinel and adds the Qwen path mapping):

```sh
.venv/bin/python -m agent_company.programbench.cli prepare-instructions \
  --official-config external/mini-swe-agent-study/src/minisweagent/config/benchmarks/programbench.yaml \
  --output instruction.md

.venv/bin/python -m agent_company.programbench.cli describe \
  --task external/programbench-study/src/programbench/data/tasks/tomnomnom__gron.88a6234 \
  --instructions instruction.md
```

`describe` and instruction preparation do not load credentials or call models.
`evaluate --task ... --instructions ... --submission ... --output ...` evaluates
an existing full-source archive exactly once. Both execution entry points accept
`--docker /absolute/path/to/wrapper`; the official evaluator receives it through
a fresh subprocess's `PROGRAMBENCH_DOCKER_EXECUTABLE`, avoiding import-cache
and global-environment changes. The optional `scripts/programbench_docker_sudo.sh`
keeps the Docker client's UID while granting its process the docker group: a root
client creates root-owned host files on `docker cp`, breaking evaluator temp-file
operations. The wrapper does not alter permanent user/group membership.

The real model-backed pair is a separate explicit command, not executed by these
docs. Invocation may incur provider charges; actual billing is unknown. After
authorization, the following example starts a **new** uncapped fd pair, not a
resume of either historical run:

```sh
.venv/bin/python -m agent_company.programbench.cli run-pair \
  --parallelism 2 \
  --profile configs/stage1/programbench_fd_no_call_cap_glm_p3.json --project-root . \
  --task external/programbench-study/src/programbench/data/tasks/sharkdp__fd.40d8eb3 \
  --instructions configs/stage1/programbench_instruction.md \
  --qwen-package external/qwen-study/package \
  --node-directory external/stage1-runtime/node-v22.23.2-linux-x64 \
  --output runs/NEW_PAIR/run
```

The shared provider profile selects GLM + retry10/524 and reads its key from the
named environment variable; no key belongs in the config/CLI. Output directories
must be new. `--parallelism 1` (the default) runs sequentially and requires confirmed
cleanup before starting the second arm. `--parallelism 2` submits Single and P3
exactly once using independent runners, containers, relays, sessions and outputs.
Each arm has its own full task budget; the controller alone writes shared progress.
A failure never cancels/retries the other arm; unconfirmed cleanup is explicitly
reported as requiring user attention. `progress.json` records terminal arms.
There is no implicit resume/relaunch.

For policy comparisons, `scripts/run_programbench_policy_comparison.py` describes
the configured task/arm plan without running models unless `--run` is explicitly
supplied. It still needs the referenced public task metadata installed locally:

```sh
.venv/bin/python scripts/run_programbench_policy_comparison.py \
  --profile configs/stage1/programbench_ac_pb1_heldout_20260924.json \
  --project-root .
```

The generic scheduler and its synthetic tests are tracked. Dated deployment
launchers and their host-specific tests remain local operations material; their
SSH configuration and credentials are not part of this interface.

Later fd/PB1 experiments used a separate evaluator copy with the upstream
`pytest-rerunfailures==16.4` change. That copy and its historical outputs are not
vendored here, and the `programbench==1.2.4` optional dependency alone does not
reproduce that evaluator patch. Preserve evaluator-version provenance and audit
unique test identities/duplicate rows before comparing fractions; do not silently
repair or replace historical results.

## No-model server verification

The 2026-09-23 interface was first checked with `scripts/check_programbench_interface.py`
and `scripts/check_programbench_team_interface.py`, which drove real Qwen with fixed
responses and a trivial Go program through the real official evaluator. Both scripts
were removed on 2026-10-15 (recover them from commit `e6a9790`). Zero-API checks of a
deployment now go through `scripts/launch_programbench_formal.py --verify` and
`--smoke --scripted`.

Primary references: [ProgramBench](https://github.com/facebookresearch/ProgramBench),
[public baseline config](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/benchmarks/programbench.yaml).
