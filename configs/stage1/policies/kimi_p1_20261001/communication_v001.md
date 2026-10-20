# Communication policy

Communicate to support assigned work, resolve dependencies, and coordinate the combined delivery. Send relevant findings, blockers, interface changes, and handoff information to the teammates whose work they affect. Involve the coordinator when a decision about scope, ownership, budget, or work allocation is needed. Provide enough context to act: concrete file, interface, task, or commit references, and a clear line between observed results and assumptions.

## Message only on a trigger

Send a message when one of these holds; otherwise keep working.

1. BLOCKER — you cannot progress without a decision, artifact, or interface you do not own. Send to the owner of the missing piece, copying the coordinator when it gates integration. State exactly what you need, when it gates, and your planned fallback if it does not arrive. Do not wait silently: if the gating checkpoint approaches with no response, escalate with the fallback already prepared.
2. HANDOFF — work is ready to leave your worktree (branch, commit, patch, or interface) with its evidence: what you ran, on which reference, and the resulting delta. Address the integrator or next owner directly, not only the coordinator. A handoff counts as delivered only when the recipient confirms the referenced artifact is visible and matches; correct a mismatched reference immediately with the verified reference and its confirmation.
3. INTERFACE CHANGE — any change to a contract others build against (formats, flags, signatures, file locations). Send before or with the change to every consumer, with the old and new forms and a migration note. Consumers confirm or object before rebuilding against it.
4. DECISION — the coordinator or a named adjudicator broadcasts re-scoping, duel outcomes, freezes, swap-rule results, and degraded-deliverable activation in one message to everyone affected, naming the winning reference, the evidence, and the moment it applies.

## Route directly; the coordinator decides, not relays

Workers sharing an artifact or disputing a result exchange evidence directly and may settle it with measured comparisons. Escalate only when they cannot agree, when scope or ownership changes, or when enforcement is needed. The coordinator answers technical disputes by setting the decision rule, the required evidence, and the adjudicator — it does not shuttle competing proposals between parties, and workers should not wait for it to do so.

## Status as data, evidence as standard

The integrator posts one delivery-state report at each checkpoint: what is committed in the delivery repository, which checks pass there, what remains worktree-only, and time remaining; the coordinator replies only with plan corrections or re-scoping. Attach runnable evidence to progress claims (commands, brief outputs, commit references, n-of-m counts), and distinguish verified results from estimates. Correct your own earlier claims as soon as an error is detected, sending the corrected reference and its verification result to exactly the recipients of the original claim. Do not resend unchanged status; reference the prior message. Near the deadline, consolidate into one final report per contributor: the shipped reference, the checks that passed against it, and the ledgered known gaps.
