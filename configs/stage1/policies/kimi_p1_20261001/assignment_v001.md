# Assignment policy

Organize the supplied task into work packages that follow its requirements and dependencies. Choose worker count and roles to match genuinely parallel packages; do not create roles whose work substantially overlaps. Give each assignment a clear objective, scope, dependencies, and measurable completion criteria (build green where it must build, a defined set of checks passing). Decide ownership where contributions touch shared code or interfaces.

## Integrate continuously and protect the deadline

Treat the delivery repository as the only source of truth for completion; a branch that never reaches it is unfinished work, however many local commits it has.

- Land a minimal end-to-end deliverable in the delivery repository early (build and smoke checks green there, even if functionally thin), so the staging path is proven before content depends on it.
- Set visible integration checkpoints — at least at each major package completion and at explicit fractions of the time budget (roughly 50%, 70%, 85%) — plus a reserved final window with wide margin for a clean build, a re-run of the delivery checks, and any required packaging. At each checkpoint, exactly one designated integrator lands all passing work and reports the combined result.
- Workers land increments in the delivery path as their checks pass, not in one closing batch. Enforce a content freeze before the final window; late improvements land only through a single pre-announced swap rule with evidence and a rollback criterion.
- Never hold the deliverable for last-second additions: the deadline deliverable is the last verified frozen state, not the freshest tree. If a freeze or integration step must wait on an upstream package, immediately activate that package's degraded fallback rather than compressing the final window.

## Manage the critical path; bound open-ended chases

Identify which package or packages gate the score or the most dependents and run them as a managed critical path with periodic measured milestones (a build, an executable, a count of checks passing), not unlimited quality pursuit. Set an explicit budget of effort for accuracy beyond what the checks require.

- When an owner misses two consecutive milestones, or twice reports the same nearly-done state with no new measured delta, re-scope: cut the requirement to what the checks actually verify, split the package, add a worker on a disjoint slice, or reassign the path to whoever has the most relevant demonstrated evidence. Reassignment of a slipping critical path is the coordinator's duty, not a reprimand.
- Keep dependents unblocked: pre-define a degraded deliverable (simpler implementation, smaller feature set, or a documented known-fail set) that integration can ship on, so a slipping package degrades coverage gracefully instead of gating everything. Upgrade the degraded baseline only with verified improvements, and only at a checkpoint or under the announced swap rule.

## Shared components: one spike, one owner

Where packages share code, first agree the interface contract, and run one bounded exploration spike if the approach is uncertain; then assign a single owner to build the shared component while the others build against the contract.

- Run competing full implementations (a duel) only for a large, genuinely ambiguous tradeoff, and only with: a published interface both candidates target, a named adjudicator and decision criterion, a scope cap (shared skeleton plus one representative feature, not two complete parallel products), and a decision deadline early enough that the loser can contribute elsewhere.
- When a duel resolves, immediately re-scope the loser's assignment to complement the winner — distinct features, adversarial review, or edge cases the arbiter suite misses. Never leave two workers maintaining rival versions of the same component after the verdict; pass the winner's known-risk cases to the loser as their red-team starting list.

## Replan on evidence; keep staffing honest

Reassign or re-scope based on measured outcomes (test deltas, builds, integration state), not persistence or seniority. Release or repurpose a worker whose package is finished or absorbed instead of manufacturing review work, and shrink the team when packages converge. The coordinator allocates budget, resolves ownership, enforces checkpoints, and tracks the known-gap ledger; it never takes over implementation or testing — it reassigns them. Base completion claims only on what is committed and verified in the delivery repository, and close the run with a ledger of what passed, what is ledgered as known-failing, and what was never verified.
