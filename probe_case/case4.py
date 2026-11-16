# -*- coding: utf-8 -*-
"""
Case 4 - multi-round retrieval with full evidence and the refined prompt

What this experiment does - the full one-retrieval, one-generation chain
   1. Decompose: DECOMPOSE_V2 lets the model split the compound question into 7-12 single-intent
      retrieval paths, one of which is forced to ask about the paper's own experimental conditions
      and apparatus (asking only how the conditions were set, never presupposing a value);
   2. Retrieve: each path runs dense top 50 -> Qwen3-Reranker-8B keeps the top 5 -> enhance_context(+/-1);
   3. Assemble: deduplicate every path's chunks by source_file::block_id and hand them to the model
      unchanged, with no selection rule and no context cap;
   4. Generate: give the original question, the sub-question list and all the evidence to the combined
      prompt (SYSTEM_PROMPT plus no-omitted-quantitative-conclusion plus every-value-needs-a-condition).

Configuration notes - what each of the four settings addresses
   1. Context budget: a per-chunk cap of 3000 and a total cap of 12000 cut whole chunks off the end of
      the context, and this case carries more than 120,000 characters of evidence, so no cap is applied;
   2. Merge rule: a global-rerank top-K selection concentrates the pool on high-scoring condition chunks
      and leaves data chunks out, so every chunk is kept deduplicated and the choice is left to the model;
   3. Generation prompt: SYSTEM_PROMPT alone does not require quantitative conclusions to be carried into
      the answer, so NUM_RULE states that requirement and COND_RULE binds every value to its conditions;
   4. Retrieval intents: no path asks how the paper set its conditions and apparatus, so DECOMPOSE_V2
      adds that path (the retrieval terms are still model-generated).


Chain parameters
  decomposition prompt   DECOMPOSE_V2 (rule 6 of DECOMPOSE_SYSTEM is rewritten to force a dedicated
                         sub-question on the paper's conditions and apparatus, with 7-12 sub-questions)
  per-path retrieval     dense top 50 -> rerank top 5 -> enhance_context(+/-1, paragraph merge, table fence)
  evidence pool          every path, deduplicated and kept unchanged; unsorted, untruncated, unfiltered
  context                no per-chunk and no total cap; U-shaped reordering
  generation prompt      pipeline.SYSTEM_PROMPT (graded evidence) plus NUM_RULE (no quantitative
                         conclusion omitted) plus COND_RULE (every value must state its condition)
  generation model       deepSeek-flash at $STEELPCR_GEN_STRONG_API_BASE, temperature = 0.0,
                         max_tokens = 32000

Running
  <python> probe_case/case4.py
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
from rag_core.helpers import Tee, count_ci, save  # noqa: E402
from rag_core.pipeline import SYSTEM_PROMPT, SteelPCRPipeline, ensure_dir, log  # noqa: E402

CASE_ID = 4

ARM_DESC = ("multi-round retrieval (full evidence and the refined prompt): the model decomposes into "
            "7-12 intents that include one on the paper's conditions and apparatus; every path is "
            "retrieved, evidence is deduplicated without a context cap, and the model does the "
            "integration itself")

# Fingerprint strings for post-hoc checks only; they never enter a query or a prompt
NEEDLES = ["79.09", "46.40", "29.44", "28.11", "10 %", "10%", "600", "管式炉"]


def main() -> None:
    ensure_dir(C.OUT_ROOT)
    log_path = os.path.join(C.OUT_ROOT, f"case{CASE_ID}_C.log")
    with Tee(log_path):
        case = C.get_case(CASE_ID)
        q = C.build_complex_question(case)

        log("=" * 78)
        log(f"case {CASE_ID} multi-round retrieval (full evidence and the refined prompt)")
        log("=" * 78)

        pipe = SteelPCRPipeline(enable_expansion=True)
        res = M.run_multi(
            CASE_ID, pipe, q,
            decompose_system=M.DECOMPOSE_V2,
            decompose_range=M.DECOMPOSE_V2_RANGE,
            top_k=M.SUB_K,
            subpoint=False,
            merge_rule="all",
            per_doc=M.NO_CAP,
            total=M.NO_CAP,
            system=SYSTEM_PROMPT + M.NUM_RULE + M.COND_RULE,
            system_label="SYSTEM_PROMPT + NUM_RULE + COND_RULE",
            prompt_builder=M.build_prompt_with_list,
            arm_desc=ARM_DESC,
        )

        res["checks"] = {n: count_ci(res["answer"], n) for n in NEEDLES}
        res["ctx_present"] = {n: count_ci(res["context"], n) for n in NEEDLES}
        res = M.with_case_meta(case, res)

        out = os.path.join(C.OUT_ROOT, f"case{CASE_ID}_C.json")
        save(out, res)
        log(f"[save] {out}  elapsed {res['latency_ms']['total'] / 1000:.1f}s")


if __name__ == "__main__":
    main()
