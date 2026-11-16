# -*- coding: utf-8 -*-
"""
Case 3 - single-round RAG

What this experiment does
   It sends case 3's full compound question (the baseline plus natural-gas, coke-oven-gas and
   green-hydrogen shaft furnaces, with the retrieval task asking for the system boundary and the
   "direct emissions down, upstream emissions up" trade-off) through the whole SteelPCR-RAG chain
   in one go.

Chain parameters
  dense initial search   FAISS-HNSW, top 50, efSearch = 128
  query expansion        expand_query bilingual terminology alignment
  reranking              Qwen3-Reranker-8B over HTTP, keeps the top 7
  context enhancement    enhance_context: +/-1 neighbour expansion, paragraph merging, table fencing
  context budget         3000 characters per chunk / 12000 in total
  ordering               U-shaped reordering
  generation             deepSeek-flash at $STEELPCR_GEN_STRONG_API_BASE, temperature = 0.0,
                         max_tokens = 32000
  generation prompt      pipeline.SYSTEM_PROMPT (graded evidence)

Running
  <python> probe_case/case3.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cases as C  # noqa: E402
from rag_core import multirag as M  # noqa: E402
from rag_core.helpers import Tee, save  # noqa: E402
from rag_core.pipeline import SteelPCRPipeline, ensure_dir, log  # noqa: E402

CASE_ID = 3


def main() -> None:
    ensure_dir(C.OUT_ROOT)
    log_path = os.path.join(C.OUT_ROOT, f"case{CASE_ID}_S.log")
    with Tee(log_path):
        case = C.get_case(CASE_ID)
        q = C.build_complex_question(case)

        log("=" * 78)
        log(f"case {CASE_ID} single-round RAG - {case['title']}")
        log("=" * 78)
        log(f"question has {len(q)} chars; {len(case['conditions'])} operating conditions")

        pipe = SteelPCRPipeline(enable_expansion=True)
        # expand_query -> embed -> FAISS top-50 -> rerank top-7 -> enhance_context(+/-1)
        # -> build_rag_context(3000/12000) -> chat(SYSTEM_PROMPT, ...)
        res = M.with_case_meta(case, M.run_single(CASE_ID, pipe, q))

        out = os.path.join(C.OUT_ROOT, f"case{CASE_ID}_S.json")
        save(out, res)
        log(f"[save] {out}")
        log(f"[context] {len(res['context'])} characters / {len(res['enhanced'])} evidence chunks")


if __name__ == "__main__":
    main()
