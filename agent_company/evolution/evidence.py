"""Bounded, read-only access to the training runs available to an optimizer."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Sequence

# Structural helpers parse whole files. Larger artifacts stay reachable through
# character pages and search, which never materialize the file.
STRUCTURED_MAX_BYTES = 4_000_000
PAGE_CHARS = 12000


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _string_leaves(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_leaves(item)


class TrainingEvidence:
    """Expose training artifacts without exposing the surrounding experiment.

    This is the capability passed to a tool-based optimizer, not a sandbox for
    arbitrary Python callbacks. The optimizer host must expose these methods
    rather than unrestricted filesystem or shell tools.
    """

    def __init__(self, root: Path, allowed_directories: Sequence[Path]):
        self.root = Path(root).resolve()
        self._allowed = tuple(Path(path).resolve() for path in allowed_directories)
        if any(not path.is_relative_to(self.root) for path in self._allowed):
            raise ValueError("Evidence directories must belong to the training root")
        self._digests: dict[tuple, dict] = {}

    @staticmethod
    def _page(offset: int, limit: int, maximum: int) -> tuple[int, int]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        return offset, min(limit, maximum)

    def _resolve(self, path: str) -> Path:
        supplied = Path(path)
        if ".." in supplied.parts:
            raise PermissionError("Parent traversal is not an evidence path")
        resolved = (supplied if supplied.is_absolute() else self.root / supplied).resolve()
        if not resolved.is_relative_to(self.root) or not any(
            resolved.is_relative_to(directory) for directory in self._allowed
        ):
            raise PermissionError("Only available training-run artifacts may be read")
        return resolved

    def _files(self) -> list[Path]:
        found: set[Path] = set()
        for directory in self._allowed:
            for candidate in directory.rglob("*"):
                if not candidate.is_file():
                    continue
                try:
                    resolved = self._resolve(str(candidate))
                except PermissionError:
                    continue
                found.add(resolved)
        return sorted(found, key=lambda path: str(path.relative_to(self.root)))

    @staticmethod
    def _source_kind(path: Path) -> str:
        # Requests contain copies of earlier observations. A search hit there
        # is evidence of model exposure, not a new occurrence of the event.
        return ("prompt_copy" if re.search(r"(?:^|_)request(?:_|\.)", path.name)
                else "artifact")

    def _selected(self, path: str | None = None, name: str | None = None,
                  source_kind: str = "all") -> list[Path]:
        if source_kind not in ("all", "artifact", "prompt_copy"):
            raise ValueError("source_kind must be all, artifact, or prompt_copy")
        if name is not None and (not isinstance(name, str) or not name or len(name) > 200):
            raise ValueError("name must be a 1–200 character filename substring")
        selected = self._resolve(path) if path is not None else None
        files = ([selected] if selected is not None and selected.is_file() else self._files())
        return [candidate for candidate in files
                if (selected is None or candidate == selected or candidate.is_relative_to(selected))
                and (name is None or name in candidate.name)
                and (source_kind == "all" or self._source_kind(candidate) == source_kind)]

    def overview(self) -> dict:
        """Task entry points, rather than the first page of raw request files."""
        files = self._files()
        entries, interrupted = [], []
        canonical = {"task_result.json", "runner_result.json", "execution.json",
                     "stdout.jsonl", "stderr.log", "events.jsonl", "reward.json",
                     "reward.txt", "usage.json", "delivery.patch",
                     "qwen_stdout.jsonl", "qwen_stderr.txt", "efficiency.json"}
        for directory in dict.fromkeys(self._allowed):
            if any(part.endswith(".interrupted") for part in directory.relative_to(self.root).parts):
                interrupted.append({"directory": str(directory.relative_to(self.root)),
                                    "file_count": sum(path.is_relative_to(directory) for path in files)})
                continue
            current, previous = [], {}
            for path in files:
                if not path.is_relative_to(directory):
                    continue
                relative = path.relative_to(directory)
                if ".interrupted" in relative.parts:
                    index = relative.parts.index(".interrupted")
                    key = str(Path(*relative.parts[:index + 2]))
                    previous[key] = previous.get(key, 0) + 1
                elif path.name in canonical:
                    current.append(str(path.relative_to(self.root)))
            entries.append({"directory": str(directory.relative_to(self.root)),
                            "current_attempt_entrypoints": current[:12],
                            "interrupted_attempts": [{"directory": str(directory.relative_to(self.root) / key),
                                                      "file_count": count}
                                                     for key, count in previous.items()]})
        return {"tasks": entries, "interrupted_attempts": interrupted, "total_files": len(files),
                "navigation": "Use list_training_files(path=task directory, name=filename substring). "
                              "Search defaults to non-request artifacts; source_kind=prompt_copy shows "
                              "carried context, not independent event occurrences."}

    def list_files(self, *, path: str | None = None, name: str | None = None,
                   source_kind: str = "all", offset: int = 0, limit: int = 50,
                   digest: bool = False) -> dict:
        offset, limit = self._page(offset, limit, 100)
        if not isinstance(digest, bool):
            raise ValueError("digest must be true or false")
        files = self._selected(path, name, source_kind)
        page = files[offset:offset + limit]
        return {
            "files": [{"path": str(path.relative_to(self.root)), "bytes": path.stat().st_size,
                       "source_kind": self._source_kind(path),
                       **({"digest": self._digest(path)} if digest else {})}
                      for path in page],
            "offset": offset,
            "next_offset": offset + len(page) if offset + len(page) < len(files) else None,
            "total": len(files),
        }

    def read_file(self, path: str, *, offset: int = 0, limit: int = 12000) -> dict:
        """Return at most 12k characters; offsets are Unicode character offsets."""
        offset, limit = self._page(offset, limit, 12000)
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        # Text iteration avoids materializing a potentially large trajectory.
        with resolved.open(encoding="utf-8", errors="replace") as stream:
            remaining = offset
            while remaining:
                skipped = stream.read(min(remaining, 65536))
                if not skipped:
                    break
                remaining -= len(skipped)
            text = stream.read(limit)
            more = bool(stream.read(1))
        return {"path": str(resolved.relative_to(self.root)), "offset": offset,
                "text": text, "next_offset": offset + len(text) if more else None}

    def search(self, query: str, *, path: str | None = None, name: str | None = None,
               source_kind: str = "artifact",
               offset: int = 0, limit: int = 20) -> dict:
        """Keep separate file/line occurrences, even when their text is identical.

        Request copies are opt-in. Artifact hits are *not* necessarily unique
        real-world events: e.g. stdout and events may describe the same event.
        Always use the returned path/line provenance to establish that claim.
        """
        offset, limit = self._page(offset, limit, 50)
        if not isinstance(query, str) or not query or len(query) > 1000:
            raise ValueError("query must contain 1–1000 characters")
        files = self._selected(path, name, source_kind)
        metadata = {"source_kind": source_kind,
                    "occurrence_note": "File/line matches are not deduplicated events. Request hits are carried context copies."}
        matches, seen = [], 0
        for candidate in files:
            with candidate.open(encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, start=1):
                    position = line.find(query)
                    if position < 0:
                        continue
                    if seen >= offset:
                        if len(matches) >= limit:
                            return {**metadata, "matches": matches, "offset": offset,
                                    "next_offset": offset + len(matches)}
                        start = max(0, position - 160)
                        matches.append({"path": str(candidate.relative_to(self.root)),
                                        "source_kind": self._source_kind(candidate),
                                        "line": line_number,
                                        "excerpt": line[start:start + 800].rstrip("\n"),
                                        "excerpt_start": start,
                                        "truncated": start > 0 or len(line) > start + 800})
                    seen += 1
        return {**metadata, "matches": matches, "offset": offset, "next_offset": None}

    # ----- Structural navigation. Counts describe file shape; they are not findings.

    @staticmethod
    def _kind(value: Any) -> str:
        if isinstance(value, dict):
            return "object"
        if isinstance(value, list):
            return "array"
        if isinstance(value, str):
            return "string"
        if isinstance(value, bool):
            return "boolean"
        return "null" if value is None else "number"

    @staticmethod
    def _entries(items: list) -> dict:
        rows = [item for item in items if isinstance(item, dict)]
        summary: dict[str, Any] = {"count": len(items)}
        for key in ("role", "tool", "event", "kind"):
            counts = Counter(str(row[key]) for row in rows
                             if isinstance(row.get(key), (str, int)) and not isinstance(row.get(key), bool))
            if counts:
                summary[key + "_counts"] = dict(counts.most_common(12))
        attempts = [row["attempt"] for row in rows
                    if isinstance(row.get("attempt"), int) and not isinstance(row.get("attempt"), bool)]
        if attempts:
            summary["attempt_range"] = [min(attempts), max(attempts)]
        return summary

    def _shape(self, data: Any) -> dict:
        if isinstance(data, list):
            return {"type": "array", "entries": self._entries(data)}
        if not isinstance(data, dict):
            return {"type": self._kind(data)}
        keys = [{"key": key, "type": self._kind(value), "chars": len(_json_text(value)),
                 **({"items": len(value)} if isinstance(value, (list, dict)) else {})}
                for key, value in list(data.items())[:30]]
        shape: dict[str, Any] = {"type": "object", "keys": keys}
        if len(data) > 30:
            shape["omitted_keys"] = len(data) - 30
        if isinstance(data.get("entries"), list):
            shape["entries"] = self._entries(data["entries"])
        return shape

    def _digest(self, path: Path) -> dict:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._digests:
            digest: dict[str, Any] = {"bytes": stat.st_size, "type": "text"}
            if stat.st_size <= STRUCTURED_MAX_BYTES and path.suffix == ".jsonl":
                with path.open(encoding="utf-8", errors="replace") as stream:
                    digest.update(type="jsonl", lines=sum(1 for _ in stream))
            elif stat.st_size <= STRUCTURED_MAX_BYTES and path.suffix == ".json":
                try:
                    digest.update(self._shape(json.loads(path.read_text(encoding="utf-8", errors="replace"))))
                except ValueError:
                    pass  # Malformed JSON remains readable as text pages.
            self._digests[key] = digest
        return self._digests[key]

    def measure(self, path: str) -> dict[str, int]:
        """Bytes and Unicode characters of one file; page offsets count characters."""
        resolved = self._resolve(path)
        stat = resolved.stat()
        key = ("measure", str(resolved), stat.st_mtime_ns, stat.st_size)
        if key not in self._digests:
            chars = stat.st_size
            if stat.st_size <= 8 * STRUCTURED_MAX_BYTES:
                with resolved.open(encoding="utf-8", errors="replace") as stream:
                    chars = sum(len(block) for block in iter(lambda: stream.read(65536), ""))
            self._digests[key] = {"bytes": stat.st_size, "chars": chars}
        return self._digests[key]

    def totals(self) -> dict[str, dict[str, int]]:
        """File and byte totals per allowed directory, for reading-coverage accounting."""
        files = self._files()
        return {str(directory.relative_to(self.root)): {
                    "files": sum(path.is_relative_to(directory) for path in files),
                    "bytes": sum(path.stat().st_size for path in files if path.is_relative_to(directory))}
                for directory in dict.fromkeys(self._allowed)}

    def index(self, max_files_per_directory: int = 60) -> dict:
        """An upfront map of small evidence sets; large raw sets remain paged."""
        files = self._files()
        tasks = []
        for directory in dict.fromkeys(self._allowed):
            members = [path for path in files if path.is_relative_to(directory)]
            entry: dict[str, Any] = {"directory": str(directory.relative_to(self.root)),
                                     "files": len(members),
                                     "bytes": sum(path.stat().st_size for path in members)}
            if len(members) <= max_files_per_directory:
                entry["file_digests"] = [{"path": str(path.relative_to(self.root)), **self._digest(path)}
                                         for path in members]
            else:
                entry["file_digests_omitted"] = ("Too many files for the index; use list_training_files "
                                                 "with path and digest=true.")
            tasks.append(entry)
        return {"tasks": tasks,
                "note": "Digests are descriptive file shape (key sizes, field counts), not findings. "
                        "Interpret counted tool calls using each file's own scope notes. Use "
                        "read_training_json with a pointer such as /by_role or /entries to open one part."}

    def read_json(self, path: str, *, pointer: str = "", start: int = 0, count: int = 20) -> dict:
        """Open one value of a parsed JSON file by RFC 6901 pointer; array items stay whole."""
        start, count = self._page(start, count, 200)
        if not isinstance(pointer, str) or len(pointer) > 500 or (pointer and not pointer.startswith("/")):
            raise ValueError("pointer must be empty or start with /, e.g. /entries or /by_role/0")
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        if resolved.stat().st_size > STRUCTURED_MAX_BYTES:
            raise ValueError("File is too large to parse; use read_training_file pages or search")
        target = json.loads(resolved.read_text(encoding="utf-8", errors="replace"))
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and token in target:
                target = target[token]
            elif isinstance(target, list) and token.isdigit() and int(token) < len(target):
                target = target[int(token)]
            else:
                raise ValueError(f"JSON pointer segment not found: {token}")
        result: dict[str, Any] = {"path": str(resolved.relative_to(self.root)), "pointer": pointer,
                                  "type": self._kind(target)}
        if isinstance(target, list):
            chosen = target[start:start + count]
            while chosen and len(_json_text(chosen)) > PAGE_CHARS:
                chosen = chosen[:-1]
            following = start + len(chosen)
            result.update(total_items=len(target), start=start, items=chosen,
                          next_start=following if following < len(target) else None)
            if not chosen and start < len(target):
                result["hint"] = f"Item {start} exceeds one page; open {pointer}/{start} or one of its fields."
            return result
        if isinstance(target, dict) and len(_json_text(target)) > PAGE_CHARS:
            keys = [{"key": key, "type": self._kind(value), "chars": len(_json_text(value))}
                    for key, value in target.items()]
            result.update(too_large_for_one_page=True, keys=keys[:100], omitted_keys=max(0, len(keys) - 100),
                          hint="Open a child pointer.")
            return result
        if isinstance(target, str) and len(target) > PAGE_CHARS:
            result.update(value=target[:PAGE_CHARS], truncated=True, chars=len(target))
            return result
        result["value"] = target
        return result

    def verify_quote(self, path: str, quote: str) -> dict:
        """Check that a citation quote occurs in the cited file.

        Matching ignores whitespace differences and one level of JSON string
        escaping, because pages show raw JSON while structured reads show decoded
        strings. It proves the text exists, not that the claim built on it is true.
        """
        if not isinstance(quote, str) or not 20 <= len(quote.strip()) <= 600:
            raise ValueError("quote must be 20–600 characters copied from the cited file")
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        if resolved.stat().st_size > 8 * STRUCTURED_MAX_BYTES:
            raise ValueError("Cited file is too large to verify; cite a smaller artifact")
        raw = resolved.read_text(encoding="utf-8", errors="replace")
        haystacks = [_normalize(raw)]
        if resolved.stat().st_size <= STRUCTURED_MAX_BYTES and resolved.suffix in (".json", ".jsonl"):
            try:
                documents = ([json.loads(raw)] if resolved.suffix == ".json" else
                             [json.loads(line) for line in raw.splitlines() if line.strip()])
            except ValueError:
                documents = []
            haystacks.append(_normalize("\n".join(leaf for doc in documents for leaf in _string_leaves(doc))))
        needles = {_normalize(quote), _normalize(_json_text(quote)[1:-1])}
        try:
            needles.add(_normalize(json.loads(f'"{quote}"')))
        except ValueError:
            pass  # Not an escaped JSON fragment.
        verified = any(needle and needle in haystack for needle in needles for haystack in haystacks)
        return {"path": str(resolved.relative_to(self.root)), "verified": verified}
