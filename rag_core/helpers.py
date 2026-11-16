# -*- coding: utf-8 -*-
"""
Helpers shared by the four case scripts - paper test harness.

Everything here comes from the scripts that produced the experiments (probe_case/run_probe.py, run_clean.py, ...),
moved into one place with the logic unchanged:
    slim()             trim a retrieval record down to the fields worth persisting
    block_key()        deduplication key for a chunk: source_file::block_id
    parse_json_array() robustly pull the sub-question JSON array out of an LLM response
    text_of()          return the body text of a chunk
    scan()             post-hoc check of where a set of key strings occurs in the chunks (verification only)
    Tee                write stdout to a log file at the same time
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List


# ------------------------------------------------------------- record trimming
def slim(rec: Dict[str, Any], with_text: bool = True) -> Dict[str, Any]:
    """Retrieval record -> the trimmed fields that get persisted. The field set matches the experiment exactly."""
    d = {
        "source_file": rec.get("source_file", ""),
        "doc_title": rec.get("doc_title", ""),
        "block_id": rec.get("block_id", ""),
        "block_type": rec.get("block_type", ""),
        "head_section": rec.get("head_section", ""),
        "head_paragraph": rec.get("head_paragraph", ""),
        "row": rec.get("_row"),
        "dense_dist": round(float(rec.get("_dense_dist", 0.0)), 6)
        if rec.get("_dense_dist") is not None else None,
        "rerank_score": round(float(rec.get("_rerank_score", 0.0)), 6)
        if rec.get("_rerank_score") is not None else None,
        "is_enhanced": rec.get("_is_enhanced", False),
        "merged_count": rec.get("_merged_count", 1),
        "merged_block_ids": rec.get("_merged_block_ids"),
    }
    if with_text:
        d["text"] = rec.get("text", "")
    return d


def block_key(rec: Dict[str, Any]) -> str:
    """Unique key used to deduplicate across sub-questions."""
    return f"{rec.get('source_file', '')}::{rec.get('block_id', '')}"


def text_of(rec: Dict[str, Any]) -> str:
    return rec.get("text") or ""


def dedupe(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by block_key, keeping first-seen order (no sorting, no truncation)."""
    out, seen = [], set()
    for r in blocks:
        k = block_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# --------------------------------------------------------- sub-question parsing
def parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Robustly extract the JSON array from an LLM response; ```json fences and surrounding prose are tolerated."""
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        text = m.group(0)
    text = re.sub(r"```(?:json)?", "", text).strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("the decomposition result is not a JSON array")
    out = []
    for i, it in enumerate(data, 1):
        sq = str(it.get("sub_question", "")).strip()
        if not sq:
            continue
        out.append({"id": int(it.get("id", i)),
                    "target": str(it.get("target", "")).strip(),
                    "sub_question": sq})
    return out


# ------------------------------------------------------------- post-hoc checks
def scan(recs: List[Dict[str, Any]], kws: List[str]) -> List[Dict[str, Any]]:
    """Return the chunks that contain the key strings (rank, document name, matched term, length).

    Used only after a run, to confirm that the target evidence really reached the retrieval layer;
    """
    out = []
    for i, r in enumerate(recs or []):
        t = text_of(r)
        hits = [k for k in kws if k in t]
        if hits:
            out.append({"rank": i + 1,
                        "doc": (r.get("doc_title") or "")[:70],
                        "kws": sorted(set(hits)),
                        "chars": len(t)})
    return out


def norm_ws(s: str) -> str:
    """Normalise U+00A0/U+2007/U+202F and other non-breaking spaces.

    Text extracted from PDFs often contains `10\xa0%` or `600\xa0 degrees C`, so a plain s.count("10 %") misses them.
    This helper is used for post-hoc checks only; it never enters retrieval or generation.
    """
    return re.sub(r"[\u00a0\u2007\u202f]", " ", s or "")


def count_ci(text: str, needle: str) -> int:
    """Count after normalising whitespace, so non-breaking spaces cannot cause false negatives."""
    return norm_ws(text).count(needle)


# ------------------------------------------------------------------------ logging
class Tee:
    """with Tee(path) as tee: ... - mirrors stdout to disk."""

    def __init__(self, path: str):
        self.path = path
        self.buf = []

    def __enter__(self):
        self._orig = sys.stdout

        class _W:
            def __init__(self, outer):
                self.outer = outer

            def write(inner, s):
                self._orig.write(s)
                inner.outer.buf.append(s)

            def flush(inner):
                self._orig.flush()

        sys.stdout = _W(self)  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        sys.stdout = self._orig
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("".join(self.buf))
        return False


def save(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
