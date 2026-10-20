"""Worker reflection prompts and their funnel for agent_rules_v1 (plan v4 section 9).

A teammate gets a one-time anchored prompt when a task it owns is completed and
when the master's shutdown request reaches it. Each prompt is followed through a
funnel: queued -> shown in a request -> delivered responses after it -> what the
teammate did in those responses (changed its own policy, proposed a change for
another teammate, wrote a lesson, posted to the board, or nothing). "Shown but
no response" and "responded without output" stay separate; no response is never
read as "no problem". No extra model call is ever made for a prompt.

Locking: the collaboration runtime calls ``injected`` with its own lock held;
this module never calls back into it.
"""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Callable

from agent_company.policy import agent_rules_text as text

COOLDOWN_SECONDS = 600
# The response to the request that showed the prompt, plus three later ones (same as rule windows).
WINDOW_RESPONSES = 4
KINDS = {"task_done": "reflect_task_done", "shutdown": "reflect_shutdown"}
OUTPUTS = ("own_policy_changes", "proposals_for_others", "lessons", "board_posts")


def prompt_text(kind: str, detail: dict) -> str:
    if kind == "task_done":
        return text.REFLECT_TASK_DONE.format(task_id=detail.get("task_id"))
    if kind == "shutdown":
        return text.REFLECT_SHUTDOWN
    raise ValueError("Unknown reflection prompt")


def _digest(value: str) -> str:
    return hashlib.sha256(" ".join(str(value).split()).lower().encode("utf-8")).hexdigest()[:16]


class ReflectionFunnel:
    def __init__(self, writer: Callable[..., None], *, clock: Callable[[], float] = time.time):
        self.writer, self.clock = writer, clock
        self._lock = threading.RLock()
        self.sequence = 0
        self.prompts: dict[str, dict] = {}
        self.last_queued: dict[tuple, float] = {}
        self.fired: set[tuple] = set()
        self.ended: set[str] = set()
        self.lesson_digests: dict[str, set] = {}
        self.finished = False

    def queue(self, worker: str, kind: str, detail: dict | None = None) -> tuple[str, str] | None:
        """(prompt id, text) to anchor at the worker's next request, or None (cooled down or repeated)."""
        detail = dict(detail or {})
        with self._lock:
            if self.finished or worker in self.ended or kind not in KINDS:
                return None
            key = (worker, kind, tuple(sorted(detail.items())))
            if key in self.fired:
                return None
            self.fired.add(key)
            now = self.clock()
            last = self.last_queued.get((worker, kind))
            if last is not None and now - last < COOLDOWN_SECONDS:
                self.writer("reflection_prompt_cooled_down", worker=worker, kind=kind, detail=detail,
                            since_previous_s=round(now - last, 3))
                return None
            self.last_queued[(worker, kind)] = now
            self.sequence += 1
            prompt_id = f"F{self.sequence}"
            self.prompts[prompt_id] = {"id": prompt_id, "worker": worker, "kind": kind, "detail": detail,
                                       "queued_at": now, "shown_at": None, "responses": 0,
                                       **{name: [] for name in OUTPUTS}, "status": "open"}
            self.writer("reflection_prompt_queued", prompt_id=prompt_id, worker=worker, kind=kind, detail=detail)
            return prompt_id, prompt_text(kind, detail)

    def injected(self, worker: str, prompt_id) -> bool:
        """Collaboration lock held: whether the queued prompt is still shown."""
        with self._lock:
            prompt = self.prompts.get(prompt_id)
            if prompt is None or prompt["status"] != "open" or prompt["worker"] != worker:
                return False
            if prompt["shown_at"] is None:
                prompt["shown_at"] = self.clock()
                self.writer("reflection_prompt_shown", prompt_id=prompt_id, worker=worker, kind=prompt["kind"])
            return True

    def lesson_trigger(self, worker: str) -> str:
        """The prompt a lesson in this worker's next response answers, else voluntary."""
        with self._lock:
            shown = [p for p in self.prompts.values()
                     if p["worker"] == worker and p["status"] == "open" and p["shown_at"] is not None]
            return max(shown, key=lambda p: p["shown_at"])["kind"] if shown else "voluntary"

    def response(self, worker: str, outputs: dict | None = None) -> None:
        """One delivered coding response of ``worker`` and what it committed."""
        outputs = outputs or {}
        with self._lock:
            lessons = []
            for lesson in outputs.get("lessons") or []:
                digest = _digest(lesson.get("text", ""))
                seen = self.lesson_digests.setdefault(worker, set())
                lessons.append({"id": lesson.get("id"), "digest": digest, "repeated_text": digest in seen})
                seen.add(digest)
            for prompt in [p for p in self.prompts.values()
                           if p["worker"] == worker and p["status"] == "open" and p["shown_at"] is not None]:
                prompt["responses"] += 1
                for name in OUTPUTS:
                    prompt[name].extend(lessons if name == "lessons" else outputs.get(name) or [])
                if prompt["responses"] >= WINDOW_RESPONSES:
                    self._close(prompt, "window_ended")

    def worker_ended(self, worker: str) -> None:
        with self._lock:
            self.ended.add(worker)
            for prompt in [p for p in self.prompts.values() if p["worker"] == worker and p["status"] == "open"]:
                self._close(prompt, "worker_ended")

    def finish(self) -> dict:
        with self._lock:
            for prompt in [p for p in self.prompts.values() if p["status"] == "open"]:
                self._close(prompt, "run_ended")
            self.finished = True
            return self.stats()

    def _close(self, prompt, reason):
        prompt["status"] = "closed"
        produced = [name for name in OUTPUTS if prompt[name]]
        outcome = ("not_shown" if prompt["shown_at"] is None else
                   "no_opportunity" if prompt["responses"] == 0 else
                   "output" if produced else "no_output")
        prompt.update(close_reason=reason, outcome=outcome, produced=produced)
        self.writer("reflection_funnel", prompt_id=prompt["id"], worker=prompt["worker"], kind=prompt["kind"],
                    detail=prompt["detail"], queued_at=prompt["queued_at"], shown_at=prompt["shown_at"],
                    responses=prompt["responses"], close_reason=reason, outcome=outcome, produced=produced,
                    **{name: prompt[name] for name in OUTPUTS})

    def stats(self) -> dict:
        with self._lock:
            rows = {}
            for prompt in self.prompts.values():
                row = rows.setdefault(prompt["kind"], {"queued": 0, "shown": 0, "with_opportunity": 0,
                                                       "with_output": 0, **{name: 0 for name in OUTPUTS}})
                row["queued"] += 1
                row["shown"] += int(prompt["shown_at"] is not None)
                row["with_opportunity"] += int(prompt["responses"] > 0)
                row["with_output"] += int(bool(prompt.get("produced")))
                for name in OUTPUTS:
                    row[name] += int(bool(prompt[name]))
            return rows
