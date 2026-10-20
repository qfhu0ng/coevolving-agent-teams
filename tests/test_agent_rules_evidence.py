"""The agent_rules_v1 evidence record's information-use join and robustness (M5 review fixes), zero API."""
import json

from agent_company.analysis.agent_rules_evidence import agent_rules_record
from agent_company.policy.agent_policy_store import AgentPolicyStore
from agent_company.policy.agent_rules import SpawnCall, parse_control, parse_spawn_policy
from agent_company.policy.role_package import create_m0_package
from agent_company.runtime.rule_engine import message_hash

T0 = 1_792_000_000.0
A, C = "a1b2c3d4e5" * 4, "f6e7d8c9b0" * 4


class Trial:
    """A trial directory written from a real store's batches plus chosen relay events."""

    def __init__(self, tmp_path, workers=("alice", "bob")):
        self.root = tmp_path / "trial"
        self.events, self.clock = [], T0
        self.store = AgentPolicyStore(create_m0_package("A", "C"), self._batch, clock=lambda: self.clock)
        for name in workers:
            self.store.commit(self.store.stage("master", self.store.snapshot("master"),
                                               [SpawnCall("s-" + name, name, parse_spawn_policy("x"))]))
            self.store.record_first_request(name)

    def _batch(self, record):
        self.events.append({"time": self.clock, **record})

    def at(self, seconds):
        self.clock = T0 + seconds
        return self

    def add(self, event, **fields):
        self.events.append({"time": self.clock, "event": event, **fields})

    def commit(self, owner, sha, paths):
        self.add("collaboration_git_observed", actor=owner, commit=sha, paths=list(paths),
                 source={"native_sample": int(self.clock * 1000)})

    def message(self, sender, to, text, commits=(), paths=(), *, ok=True, board=False, outcome=True):
        digest = message_hash(text)
        self.add("message_facts", sender=sender, to=to, board=board, hash=digest, commits=list(commits),
                 paths=list(paths))
        if outcome:
            self.add("message_outcome", native_event="team_message_sent", hash=digest, to=to,
                     sender=sender, ok=ok, recipients=None)

    def record(self, obligations=None):
        relay = self.root / "team" / "relay"
        relay.mkdir(parents=True, exist_ok=True)
        (relay / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.events))
        if obligations is not None:
            (relay / "rule_obligations.json").write_text(json.dumps(obligations))
        return agent_rules_record(self.root)


def results(record):
    return [r for m in record["information_use"]["messages"] for r in m["results"]]


def test_a_possible_cherry_pick_is_undetermined_and_a_clean_miss_is_not_seen(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(110).message("alice", "leader", "SHA: " + A[:9], commits=[A[:9]])
    trial.at(150).commit("master", C, ["src/A.hs"])  # a different SHA changing the same path
    row = results(trial.record())[0]
    assert (row["status"], row["undetermined_reason"]) == ("undetermined", "possible_rewrite")
    other = Trial(tmp_path / "other")
    other.at(100).commit("alice", A, ["src/A.hs"])
    other.at(110).message("alice", "leader", "SHA: " + A[:9], commits=[A[:9]])
    other.at(150).commit("master", C, ["src/B.hs"])
    assert results(other.record())[0]["status"] == "not_seen"


def test_board_posts_reach_every_other_active_worker(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(110).message("alice", "leader", "BOARD: done\nSHA: " + A[:9], commits=[A[:9]], board=True)
    trial.at(200).commit("bob", A, ["src/A.hs"])
    trial.at(210).commit("master", C, ["src/Z.hs"])
    by_recipient = {r["recipient"]: r for r in results(trial.record())}
    assert set(by_recipient) == {"master", "bob"}
    assert by_recipient["bob"]["status"] == "appeared_after" and by_recipient["bob"]["in_delivery"] == "not_seen"


def test_outcomes_are_paired_by_recipient_too(tmp_path):
    trial = Trial(tmp_path, workers=("alice", "bob", "carol"))
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(110).message("alice", "carol", "SHA: " + A[:9], commits=[A[:9]], outcome=False)
    trial.at(111).message("alice", "bob", "SHA: " + A[:9], commits=[A[:9]])
    trial.at(200).commit("bob", A, ["src/A.hs"])
    rows = trial.record()["information_use"]["messages"]
    assert [(row["to"], row["delivery"]) for row in rows] == [("carol", "unknown"), ("bob", "sent")]


def test_recipients_that_ended_or_were_never_sampled_after_are_undetermined(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(102).commit("bob", C, ["src/B.hs"])
    trial.at(105)
    trial.store.record_terminal("bob")
    trial.at(110).message("alice", "bob", "SHA: " + A[:9] + " PATHS: src/A.hs", commits=[A[:9]], paths=["src/A.hs"])
    rows = results(trial.record())
    assert [(r["status"], r["undetermined_reason"]) for r in rows] == [("undetermined", "not_sampled")] * 2


def test_noise_tokens_are_counted_not_matched(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(110).message("alice", "leader", "x", commits=["1234567", A[:9]], paths=["and/or", "src/A.hs."])
    record = trial.record()
    coverage = record["information_use"]["coverage"]
    assert coverage["tokens_not_matching_a_commit"] == 1 and coverage["tokens_not_matching_a_changed_path"] == 1


def test_timeline_lists_each_commit_once_per_worktree(tmp_path):
    trial = Trial(tmp_path)
    for seconds in (100, 160, 220):
        trial.at(seconds).commit("alice", A, ["src/A.hs"])
    commits = trial.record()["organization_timeline"]["worktree_commits"]
    assert [(c["worktree_of"], c["commit"]) for c in commits] == [("alice", A[:12])]


def test_malformed_inputs_become_coverage_not_errors(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(110).add("message_facts", sender="alice", to="*", board=False, hash="h", commits=[A[:9]], paths=[])
    trial.add("message_outcome", native_event="team_broadcast", hash="h", to="*", sender="alice", ok=None,
              recipients=[{"outcome": "accepted"}, "bad"])
    record = trial.record(obligations=[])
    assert record["coverage"]["rule_obligations_not_an_object"] is True
    assert record["information_use"]["messages"][0]["results"] == []


def test_the_delivery_repository_last_sample_separates_a_miss_from_no_sample(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(101).commit("master", C, ["src/base.hs"])
    trial.at(110).message("alice", "leader", "SHA: " + A[:9], commits=[A[:9]])
    # The delivery repository is read on every refresh but writes rows only when it changes (M5 review R2-4).
    assert results(trial.record())[0]["undetermined_reason"] == "not_sampled"
    assert results(trial.record({"delivery_last_sample_at": T0 + 400}))[0]["status"] == "not_seen"


def test_paths_keep_a_leading_dot_and_lose_a_leading_dot_slash(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, [".github/workflows/ci.yml", "src/a.py"])
    trial.at(110).message("alice", "leader", "x", paths=[".github/workflows/ci.yml,", "./src/a.py"])
    trial.at(150).commit("master", C, [".github/workflows/ci.yml"])
    row = results(trial.record())[0]
    assert row["paths"] == [".github/workflows/ci.yml", "src/a.py"] and row["touched_after"] is True


def test_in_delivery_uses_the_first_sighting_and_names_ignore_case(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(101).commit("master", A, ["src/A.hs"])  # already delivered
    trial.at(110).message("alice", "Bob", "SHA: " + A[:9], commits=[A[:9]])
    trial.at(150).commit("master", A, [])  # an unchanged-HEAD row repeats it
    trial.at(160).commit("bob", A, ["src/A.hs"])
    row = results(trial.record())[0]
    assert (row["recipient"], row["status"], row["in_delivery"]) == ("bob", "appeared_after", "present_before")


def test_a_board_post_without_a_native_outcome_is_counted(tmp_path):
    trial = Trial(tmp_path)
    trial.at(100).commit("alice", A, ["src/A.hs"])
    trial.at(110).message("alice", "leader", "BOARD: x", commits=[A[:9]], board=True, outcome=False)
    use = trial.record()["information_use"]
    assert use["messages"][0]["delivery"] == "unknown" and use["totals"]["messages_without_native_outcome"] == 1


def test_a_broadcast_through_the_real_relay_pairs_with_its_outcome(tmp_path):
    import time
    from test_agent_rules_generation import Upstream
    from test_team import call, request, response, workspace
    from agent_company.runtime.relay import ProviderConfig, TeamRelay
    from agent_company.runtime.team import WorkerRegistry
    relay_dir = tmp_path / "trial/team/relay"
    up = Upstream()
    files = {n: str(tmp_path / f"runtime/status/{n}.md") for n in ("policies", "lessons", "board")}
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      "C", relay_dir, 5000, up, role_policy_mode="propose", policy_package=create_m0_package("A", "C"),
                      policy_files=files)

    def turn(actor, calls=()):
        up.queue.append(response(list(calls)))
        return relay.handle(request(actor, ("agent", "send_message") if actor == "master" else ("send_message",)))

    def sample(actor, sha, offset):
        relay.collaboration.observe_git(actor, {"commit": sha, "new_commit": actor != "master", "changed_paths": ["src/A.hs"],
                                                "paths": ["src/A.hs"]}, {"native_sample": int((time.time() + offset) * 1000)})
    for name in ("alpha", "beta"):
        turn("master", [call("agent", {"name": name, "description": "w", "prompt": "x"})])
    turn("alpha")
    turn("beta")
    sample("master", C, -20)
    sample("alpha", A, -10)
    text = "Everyone: SHA: " + A[:9]
    turn("alpha", [call("send_message", {"to": "*", "message": text})])
    # The shape compat writes for a broadcast: no "to" (qwen_native_compat.mjs).
    relay.record_lifecycle_event({"event": "team_broadcast", "from": "alpha", "hash": message_hash(text), "total": 2,
                                  "recipients": [{"name": "beta", "outcome": "accepted"},
                                                 {"name": "leader", "outcome": "accepted"}]}, "stderr:9")
    sample("beta", A, 10)
    sample("master", C, 20)  # unchanged: no row, but a later readable sample
    relay.finish_rule_engine(relay_dir / "rule_obligations.json")
    row = agent_rules_record(tmp_path / "trial")["information_use"]["messages"][0]
    assert (row["kind"], row["delivery"]) == ("broadcast", "broadcast")
    assert {r["recipient"]: r["status"] for r in row["results"]} == {"beta": "appeared_after", "master": "not_seen"}


def test_rules_and_calls_the_runtime_refused_are_in_the_record(tmp_path):
    from test_agent_rules_generation import Upstream
    from test_team import call, request, response, workspace
    from agent_company.runtime.relay import ProviderConfig, TeamRelay
    from agent_company.runtime.team import WorkerRegistry
    relay_dir = tmp_path / "trial/team/relay"
    up = Upstream()
    files = {n: str(tmp_path / f"runtime/status/{n}.md") for n in ("policies", "lessons", "board")}
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      "C", relay_dir, 5000, up, role_policy_mode="propose", policy_package=create_m0_package("A", "C"),
                      policy_files=files)

    def turn(actor, calls=()):
        up.queue.append(response(list(calls)))
        return relay.handle(request(actor, ("agent", "send_message") if actor == "master" else ("send_message",)))
    prompt = "Work.\nPOLICY:\nWHEN every 10 minutes THEN SEND TO leader\nWHEN commit THEN SEND TO leader WITH sha\nEND-POLICY\n"
    turn("master", [call("agent", {"name": "alpha", "description": "w", "prompt": prompt})])
    turn("alpha")
    turn("master", [call("send_message", {"to": "alpha", "message": "AGENT-POLICY alpha add\nWHEN idle THEN SEND TO leader"})])
    relay.finish_rule_engine(relay_dir / "rule_obligations.json")
    refused = agent_rules_record(tmp_path / "trial")["runtime_refused"]
    assert [(r["worker"], r["line"]["text"], r["code"]) for r in refused["spawn_rules_skipped"]] == [
        ("alpha", "WHEN every 10 minutes THEN SEND TO leader", "unknown_trigger")]
    [control] = refused["calls_refused"]
    assert control["role"] == "master" and control["code"] == "unknown_trigger"
    assert "WHEN idle THEN SEND TO leader" in control["detail"]["text"]
    assert refused["by_code"] == {"unknown_trigger": 2}


def test_container_memory_during_inference_comes_from_the_resource_observer(tmp_path):
    trial = Trial(tmp_path)
    rows = [{"memory_scope": "inference_container", "memory_peak_bytes": 30 * 2**30, "memory_events": {"oom_kill": 0}},
            {"memory_scope": "inference_container", "memory_peak_bytes": 44 * 2**30, "memory_events": {"oom_kill": 2}},
            {"memory_scope": "inference_container", "memory_peak_bytes": None, "memory_events": None},
            {"memory_scope": "post_quiesce_container", "memory_peak_bytes": 90 * 2**30, "memory_events": {"oom_kill": 9}}]
    trial.root.mkdir(parents=True)
    (trial.root / "observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + "not json\n")
    assert trial.record()["resources"] == {"inference_samples": 3, "memory_peak_bytes": 44 * 2**30,
                                           "processes_killed_for_memory": 2, "samples_without_memory": 1,
                                           "invalid_lines": 1}


def test_rule_text_paths_and_notes_diffs_are_redacted_like_the_notes(tmp_path):
    # External review #6: notes were redacted but the diff, rule text and PATHS were copied as written.
    trial = Trial(tmp_path, workers=())
    rule = "WHEN commit PATHS keys/sk-live1234567890/** THEN SEND TO leader WITH sha"
    prompt = f"Work.\nPOLICY:\n{rule}\nNOTES:\nuse API_KEY=abc123secret\nEND-POLICY\n"
    store = trial.store
    store.commit(store.stage("master", store.snapshot("master"), [SpawnCall("s-carol", "carol", parse_spawn_policy(prompt))]))
    store.record_first_request("carol")
    proposal = parse_control("p1", "leader", "POLICY-PROPOSAL self notes\nkeep TOKEN=zzz999secret\nREASON: r", actor="carol")
    assert not store.commit(store.stage("carol", store.snapshot("carol"), [proposal])).rejections
    sha = "ab12cd34" * 5
    trial.at(5).commit("carol", sha, ["src/A.hs"])
    trial.at(6).message("carol", "sk-live1234567890", "done", commits=[sha[:9]])
    record = trial.record({"stats": {"rows": [{"worker": "carol", "rule": rule, "satisfied": 0}]},
                           "obligations": [{"id": "O1", "worker": "carol", "rule_id": "R1", "version": 0,
                                            "rule": rule, "trigger": "commit", "status": "missed"}]})
    text = json.dumps(record)
    assert not any(secret in text for secret in ("sk-live1234567890", "abc123secret", "zzz999secret"))
    [carol] = record["policies"]
    assert carol["versions"][0]["rules"]["R1"] == "WHEN commit PATHS keys/[REDACTED]/** THEN SEND TO leader WITH sha"
    old, new = carol["versions"][1]["diff"]["notes"]
    assert old["redacted"] and new["redacted"] and new["text"] == "keep [REDACTED]"
    assert record["rule_obligations"]["obligations"][0]["rule"] == carol["versions"][0]["rules"]["R1"]
    assert record["rule_obligations"]["stats"]["rows"][0]["rule"] == carol["versions"][0]["rules"]["R1"]
    assert [m["to"] for m in record["information_use"]["messages"]] == ["[REDACTED]"]
