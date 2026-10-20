"""Deterministic evidence paging for outer research loops; no semantic reducer.

Full turns remain in caller traces, without reasoning (plan v4 12, B1). Once, at
a completed-turn boundary, a fresh submission context can replace accumulated
history with original inputs and bounded evidence notes. This never adds model-call rounds or splits tool pairs.

`WorkingMemory` is an opt-in alternative for long evidence reviews. In the
default paged-history mode every page stays in context, so a 180k-character
session closed reading after about a dozen 12k pages and the one-time rotation
was unreachable (reading closed ~10k before it could trigger). Working memory
keeps pages verbatim for a short window, then replaces them with provenance
stubs, and rotates to a fresh context carrying notes while calls remain.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import copy


def size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def content_size(value) -> int:
    """Tool JSON is itself encoded as a string inside the message JSON."""
    return size(json.dumps(value, ensure_ascii=False)) - 2


def _union(intervals: list[list[int]]) -> int:
    total, end = 0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


@dataclass(frozen=True)
class WorkingMemory:
    """Evidence pages are working context; research notes are the durable memory.

    page_window: completed model calls a page stays verbatim before becoming a stub.
    notebook_chars: notes carried across windows and rotations.
    final_calls: trailing submission-only calls, e.g. 2 leaves one correction.
    min_page_chars: free context needed to keep reading without rotating.
    """
    page_window: int = 2
    notebook_chars: int = 40_000
    final_calls: int = 1
    min_page_chars: int = 12_000
    # None keeps every assistant reasoning_content (required by some thinking
    # APIs). N keeps the latest N verbatim and stubs older ones, as rotation
    # already drops all of them; nothing is summarized. PB2 attempt 1: GLM
    # reasoning of 20–37k characters per turn filled half of each context.
    reasoning_window: int | None = None

    def __post_init__(self):
        for name in ("page_window", "notebook_chars", "final_calls", "min_page_chars", "reasoning_window"):
            value = getattr(self, name)
            if name == "reasoning_window" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


# Reasoning is never written to disk (plan v4 section 12, B1); user 2026-10-18 dropped "remains in the trace".
REASONING_STUB = "[Earlier reasoning omitted from the working context.]"


class OuterBudget:
    def __init__(self, system: str, limit: int, rounds: int, submit: str,
                 max_output_tokens: int | None, inspection_max_output_tokens: int = 4096,
                 memory: WorkingMemory | None = None, final_calls: int = 1):
        self.system, self.limit, self.rounds = system, limit, rounds
        # Without working memory: how many last calls offer only the submit tool (default 1, as before).
        if isinstance(final_calls, bool) or not isinstance(final_calls, int) or not 1 <= final_calls <= rounds:
            raise ValueError("final_calls must be an integer from 1 to the number of rounds")
        self.final_calls = final_calls
        self.submit, self.max_output_tokens = submit, max_output_tokens
        if (isinstance(inspection_max_output_tokens, bool)
                or not isinstance(inspection_max_output_tokens, int)
                or inspection_max_output_tokens < 1):
            raise ValueError("inspection_max_output_tokens must be a positive integer")
        self.inspection_max_output_tokens = inspection_max_output_tokens
        # Leave room for call arguments, paired acknowledgements and the next
        # request's budget notice. This is a character budget, not a claim about
        # a provider tokenizer or the model's ability to obey the output limit.
        self.reserve = min(8192, max(512, limit // 4))
        self.submission_only = False
        self.initial = None
        self.notes = []
        self.excerpts = []
        self.rollovers = 0
        self.memory = memory
        self.note_limit = memory.notebook_chars if memory else min(12000, max(256, limit // 8))
        # Reading ledger: what the model was actually shown (after paging), for
        # coverage reporting and for checking rationale identifiers afterwards.
        self.step = 0
        self.totals: dict[str, dict[str, int]] = {}
        # Optional path -> {"bytes", "chars"}: opened files count in characters,
        # like the pages shown; unopened files stay estimated by their bytes.
        self.measure = None
        self.reads: dict[str, list[list[int]]] = {}
        self.structured: list[dict] = []
        self.searches = self.listings = self.elided = 0
        # Search excerpts are evidence shown too (PB2 attempt 2 ran 56 searches).
        self.searched: dict[str, int] = {}
        self.seen: list[str] = []
        self.visible: list[dict] = []
        # Working memory: "inspect", "consolidate" (notes only, before a
        # rotation) or "submit". The largest completed turn predicts whether the
        # next call still fits; model reasoning, not evidence, can dominate it.
        self.phase = "inspect"
        self.largest_turn = 0
        self.recent_growth: list[int] = []
        self._last_size = None
        self.consolidations = self.reasoning_elided = 0

    def record_note(self, note: str, evidence_refs: list[str], evidence) -> dict:
        longest = 4000 if self.memory else 2000  # Notes are the only memory across rotations.
        if not isinstance(note, str) or not note.strip() or len(note) > longest:
            raise ValueError(f"Research note must be nonempty and at most {longest} characters")
        if (not isinstance(evidence_refs, list) or not 1 <= len(evidence_refs) <= 6
                or any(not isinstance(p, str) for p in evidence_refs)):
            raise ValueError("Supply 1–6 paths from the supplied evidence interface")
        refs = [evidence.read_file(p, limit=1)["path"] for p in evidence_refs]
        item = {"note": note, "evidence_refs": refs,
                "status": "model_observation_or_hypothesis_not_verified_fact"}
        if size(self.notes + [item]) > self.note_limit:
            raise ValueError("Research notebook full; use existing notes and submit")
        self.notes.append(item)
        return {"recorded": True, "note_count": len(self.notes)}

    def state(self):
        return {"rollovers": self.rollovers, "notes": self.notes, "evidence_excerpts": self.excerpts,
                "call_budget_reset": False}

    def output_limit(self):
        return (self.max_output_tokens if self.submission_only else
                min(self.max_output_tokens or self.inspection_max_output_tokens,
                    self.inspection_max_output_tokens))

    def prepare(self, messages: list[dict], step: int) -> bool:
        messages[0]["content"] = self.system
        if self.initial is None:
            self.initial = copy.deepcopy(messages[:2])
        self.step = step
        if self.memory is not None:
            return self._prepare_working_memory(messages, step)
        remaining = self.limit - size(messages)
        if remaining < 1024 and len(messages) > 2 and not self.rollovers:
            # Called at a completed-turn boundary, never split a tool-call pair.
            notebook = {"session_transition": "Completed turns archived; fresh submission session.",
                        "instruction": "Use supported notes; if evidence is insufficient, retain the policy "
                        "or review as uncertain. Do not invent missing observations.", **copy.deepcopy(self.state()),
                        "omitted_note_entries": 0, "rollovers": self.rollovers + 1}
            while size(self.initial) + size(notebook) + 1024 > self.limit:
                entries = notebook["evidence_excerpts"] or notebook["notes"]
                if not entries:
                    break  # Never truncate the mandatory original policies/input.
                entries.pop(0)
                notebook["omitted_note_entries"] += 1
            messages[:] = copy.deepcopy(self.initial) + [{"role": "user", "content": json.dumps(notebook)}]
            self.rollovers += 1
            remaining = self.limit - size(messages)
        self.submission_only = (bool(self.rollovers) or step >= self.rounds - self.final_calls
                                or remaining < self.reserve + 2048)
        notice = {"remaining_model_calls_including_this": self.rounds - step,
                  "remaining_context_chars_before_notice": max(0, remaining),
                  "max_output_tokens": self.max_output_tokens,
                  "phase": "submit_now" if self.submission_only else "inspect_or_submit",
                  "instruction": f"Use {self.submit} with complete required fields. "
                                 "Evidence pages may be shortened to preserve submission space. "
                                 "Do not spend the last call on reading. Never abbreviate required policy text."}
        messages[0]["content"] += "\n\nResearch-loop budget: " + json.dumps(notice)
        if size(messages) > self.limit:
            raise ValueError("Outer-loop context limit reached; complete submission has no input space")
        return self.submission_only

    # ----- Working-memory mode

    def _prepare_working_memory(self, messages: list[dict], step: int) -> bool:
        window = self.memory.page_window
        for entry in self.visible:
            # A page returned by call s is visible verbatim to calls s+1..s+window.
            if not entry["elided"] and step - entry["step"] > window:
                entry["message"]["content"] = json.dumps(self._stub(entry["result"]), ensure_ascii=False)
                entry["elided"] = True
                self.elided += 1
        if self.memory.reasoning_window is not None:
            kept = 0
            for message in reversed(messages):
                if message.get("role") != "assistant" or not message.get("reasoning_content"):
                    continue
                if kept < self.memory.reasoning_window:
                    kept += 1
                elif message["reasoning_content"] != REASONING_STUB:
                    message["reasoning_content"] = REASONING_STUB  # The key stays present.
                    self.reasoning_elided += 1
        # Net growth per completed turn, after this call's stubbing; the system
        # notice is rebuilt every call and excluded. The first turns after a
        # rotation grow without anything to stub, so only recent turns predict.
        current = size(messages[1:])
        if self._last_size is not None:
            self.recent_growth = (self.recent_growth + [current - self._last_size])[-3:]
            self.largest_turn = max(self.largest_turn, self.recent_growth[-1])
        remaining = self.limit - size(messages)
        final = step >= self.rounds - self.memory.final_calls
        # Rotate before the next completed turn could overflow, not after. A turn
        # is its assistant message (including reasoning) plus its tool results.
        crowded = remaining < self.reserve + 2048 + max([self.memory.min_page_chars, *self.recent_growth])
        consolidated = self.phase == "consolidate"
        self.phase = "inspect"
        if not final and len(messages) > 2 and crowded:
            if consolidated or remaining < self.reserve + 2048:
                self._rotate(messages)  # At a completed-turn boundary; tool pairs stay whole.
                remaining = self.limit - size(messages)
            else:
                self.phase = "consolidate"  # One notes-only call, then rotate.
                self.consolidations += 1
        self.submission_only = final or remaining < self.reserve + 2048
        if self.submission_only:
            self.phase = "submit"
        instruction = (
            "CONSOLIDATE NOW: the next call starts a fresh context. Evidence tools are paused for "
            "this call. Save every finding, open question and planned next read you still need with "
            "record_research_note; notes are the only memory carried forward."
            if self.phase == "consolidate" else
            f"Evidence results stay verbatim for {window} model calls, then become provenance stubs. "
            "Save findings with record_research_note before they leave; notes persist across context "
            f"rotations. Re-read if a detail matters. Use {self.submit} with complete required fields; "
            "never abbreviate required policy text.")
        notice = {"remaining_model_calls_including_this": self.rounds - step,
                  "remaining_context_chars_before_notice": max(0, remaining),
                  "max_output_tokens": self.max_output_tokens,
                  "phase": {"inspect": "inspect_or_submit", "consolidate": "consolidate_notes",
                            "submit": "submit_now"}[self.phase],
                  "working_memory": {"page_visible_for_calls": window,
                                     "notebook_chars_used": size(self.notes),
                                     "notebook_chars_limit": self.note_limit,
                                     "rotations": self.rollovers, "elided_pages": self.elided},
                  "reading_coverage": self.coverage(),
                  "instruction": instruction}
        messages[0]["content"] += "\n\nResearch-loop budget: " + json.dumps(notice)
        if size(messages) > self.limit:
            raise ValueError("Outer-loop context limit reached; complete submission has no input space")
        self._last_size = size(messages[1:])
        return self.submission_only

    def _rotate(self, messages: list[dict]) -> None:
        notebook = {"session_transition": "Working context rotated. Earlier turns and evidence pages are "
                                          "archived on disk and no longer visible.",
                    "instruction": "Continue from these notes and the reading ledger. Re-read evidence when "
                                   "a detail matters. Do not invent missing observations.",
                    "notes": copy.deepcopy(self.notes), "omitted_note_entries": 0,
                    "reading_coverage": self.coverage(),
                    "files_shown": self._files_shown()[:80], "rotations": self.rollovers + 1}
        while size(self.initial) + size(notebook) + self.reserve + 2048 > self.limit and notebook["notes"]:
            notebook["notes"].pop(0)
            notebook["omitted_note_entries"] += 1
        messages[:] = copy.deepcopy(self.initial) + [
            {"role": "user", "content": json.dumps(notebook, ensure_ascii=False)}]
        self.visible.clear()
        self.rollovers += 1

    @staticmethod
    def _stub(result: dict) -> dict:
        stub = {"elided_from_working_context": True,
                "reason": "Visibility window passed; the full result remains in the caller trace. "
                          "Rely on recorded notes or re-read."}
        for key in ("path", "offset", "next_offset", "pointer", "start", "next_start", "total_items", "total"):
            if key in result:
                stub[key] = result[key]
        if isinstance(result.get("text"), str):
            stub["chars"] = len(result["text"])
        for key in ("files", "matches", "items", "keys"):
            if isinstance(result.get(key), list):
                stub[key + "_count"] = len(result[key])
        return stub

    # ----- Reading ledger (both modes; accounting only)

    def _account(self, result: dict, content: str) -> None:
        path = result.get("path")
        if isinstance(path, str) and isinstance(result.get("text"), str):
            start = result["offset"] if isinstance(result.get("offset"), int) else 0
            self.reads.setdefault(path, []).append([start, start + len(result["text"])])
        elif isinstance(path, str) and any(key in result for key in ("items", "value", "keys")):
            self.structured.append({"path": path, "pointer": result.get("pointer", ""),
                                    "start": result.get("start"), "chars": len(content)})
        elif isinstance(result.get("matches"), list):
            self.searches += 1
            for match in result["matches"]:
                if isinstance(match, dict) and isinstance(match.get("path"), str):
                    self.searched[match["path"]] = (self.searched.get(match["path"], 0)
                                                    + len(str(match.get("excerpt", ""))))
        elif isinstance(result.get("files"), list):
            self.listings += 1
        self.seen.append(content)

    def _files_shown(self) -> list[dict]:
        rows = {path: {"path": path, "page_chars": _union(spans)} for path, spans in self.reads.items()}
        for read in self.structured:
            row = rows.setdefault(read["path"], {"path": read["path"], "page_chars": 0})
            row.setdefault("structured_pointers", []).append(read["pointer"] or "/")
        return sorted(rows.values(), key=lambda row: row["path"])

    def coverage(self) -> dict:
        """Characters shown per task directory against its bytes; approximate by design."""
        rows = {}
        for directory, total in self.totals.items():
            prefix = directory.rstrip("/") + "/"

            def inside(path, directory=directory, prefix=prefix):
                return path == directory or path.startswith(prefix)
            pages = sum(_union(spans) for path, spans in self.reads.items() if inside(path))
            structured = sum(read["chars"] for read in self.structured if inside(read["path"]))
            opened = ({path for path in self.reads if inside(path)}
                      | {read["path"] for read in self.structured if inside(read["path"])})
            searched = {path for path in self.searched if inside(path)}
            excerpts = sum(self.searched[path] for path in searched)
            shown = pages + structured + excerpts
            extent = total["bytes"]
            if self.measure is not None:
                for path in opened | searched:
                    measured = self.measure(path)
                    extent += measured["chars"] - measured["bytes"]
            rows[directory] = {"files_opened": len(opened), "files_touched": len(opened | searched),
                               "search_excerpt_chars": excerpts, "files_total": total["files"],
                               "chars_shown": shown, "bytes_total": total["bytes"],
                               "approx_fraction": round(min(1.0, shown / extent), 3) if extent else None}
        return rows

    def reading_audit(self) -> dict:
        return {"mode": "working_memory" if self.memory else "paged_history",
                "page_visible_for_calls": self.memory.page_window if self.memory else None,
                "rotations": self.rollovers, "consolidations": self.consolidations,
                "largest_turn_chars": self.largest_turn, "elided_pages": self.elided,
                "reasoning_elided": self.reasoning_elided,
                "searches": self.searches, "listings": self.listings,
                "structured_reads": len(self.structured), "notes": len(self.notes),
                "coverage": self.coverage(), "files_shown": self._files_shown(),
                "caveat": "Coverage counts characters shown to the model (pages deduplicated; structured "
                          "reads and search excerpts not) against touched files' characters plus other "
                          "files' bytes. It is approximate, and being shown text is not the same as using "
                          "it correctly."}

    def append_result(self, messages: list[dict], call_id: str, result: dict,
                      remaining_calls: int) -> dict:
        # Small source excerpts survive rotation; reasoning is never summarized.
        if isinstance(result.get("path"), str) and isinstance(result.get("text"), str):
            self.excerpts.append({"path": result["path"], "offset": result.get("offset", 0),
                                  "text": result["text"][:480], "excerpt_only": True})
            while self.excerpts and size(self.excerpts) > self.note_limit:
                self.excerpts.pop(0)
        shell = {"role": "tool", "tool_call_id": call_id, "content": ""}
        # Budget all outstanding paired tool messages, not just the first one.
        available = max(0, (self.limit - size(messages) - self.reserve
                            - (remaining_calls + 1) * (size(shell) + 8)) // (remaining_calls + 1))
        # A valid final submission needs only an acknowledgement and no next
        # provider request. Do not relabel it as an omitted evidence read.
        projected = result if result.get("submitted") is True else self._page(result, available)
        shell["content"] = json.dumps(projected, ensure_ascii=False)
        messages.append(shell)
        self._account(projected, shell["content"])
        if self.memory is not None and any(key in projected for key in
                                           ("text", "files", "matches", "items", "value", "keys")):
            self.visible.append({"message": shell, "step": self.step, "result": projected, "elided": False})
        return projected

    @staticmethod
    def _page(result: dict, budget: int) -> dict:
        if content_size(result) <= budget:
            return result
        page = dict(result)
        page["context_limited"] = True
        if isinstance(page.get("text"), str):
            full = page["text"]
            # Bound the encoded JSON, including quotes and escaped controls.
            low, high = 0, len(full)
            while low < high:
                middle = (low + high + 1) // 2
                page.update(text=full[:middle], next_offset=page.get("offset", 0) + middle)
                if content_size(page) <= budget:
                    low = middle
                else:
                    high = middle - 1
            page.update(text=full[:low], next_offset=page.get("offset", 0) + low)
            if low and content_size(page) <= budget:
                return page
        for key in ("files", "matches", "items"):
            if isinstance(page.get(key), list):
                page[key] = list(page[key])
                while page[key] and content_size(page) > budget:
                    page[key].pop()
                    if key == "items":  # Structured JSON pages continue by item index.
                        page["next_start"] = page.get("start", 0) + len(page[key])
                    else:
                        page["next_offset"] = page.get("offset", 0) + len(page[key])
                if page[key] and content_size(page) <= budget:
                    return page
        # The original result remains in the trace; this is not an empty file,
        # a failed search or a successful read. No arbitrary JSON truncation.
        return {"context_limited": True,
                "error": "No room for this evidence page. Submit using the evidence already available."}


def require_complete_output(response: dict, label: str) -> None:
    """A truncated tool response must never become an adopted partial policy."""
    if response["choices"][0].get("finish_reason") in ("length", "max_tokens"):
        raise ValueError(f"{label} output token limit reached before a complete submission; "
                         "increase its configured output allowance or shorten the proposed full texts")
