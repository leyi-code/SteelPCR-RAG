# -*- coding: utf-8 -*-
"""
Case 2 - multi-round retrieval with term optimisation and context expansion

What this experiment does
   It first splits case 2's compound question into 10 single-intent retrieval paths, each probing one
   variable dimension: fuel nitrogen, ash removal and potassium removal, AAEM, ash and calorific value,
   alkali load, sulphur, volatile matter, feasibility of a high substitution ratio, the carbon-pollutant
   trade-off, and pretreatment. Each path runs the same retrieval chain and produces a short answer;
   the answers and the merged evidence are then integrated into one final answer.

Differences from single-round RAG (two):
   (1) retrieval: 10 fine-grained retrieval intents instead of one original question;
   (2) context: the single-point answers use 2500/8000, the integration stage relaxes to 8000/30000.

Chain parameters
  retrieval intents     10 hand-written property/mechanism intents; the terms contain no answer values
  per-path retrieval    dense top 50 -> Qwen3-Reranker-8B keeps the top 5 -> enhance_context(+/-1)
  single-point answer   2500 characters per chunk / 8000 in total; SUBANSWER_SYSTEM, at most 120
                        characters, labelled [片段N]
  evidence merging      chunks taken round-robin across intents (first chunk of every intent, then the
                        second, ...), pool capped at 8
  integration context   8000 characters per chunk / 30000 in total, U-shaped reordering
  integration prompt    INTEGRATE_SYSTEM (graded evidence; the per-sub-question drafts are intermediate
                        only and the evidence wins any conflict)
  generation model      deepSeek-flash, temperature = 0.0, max_tokens = 32000

Running
  <python> probe_case/case2.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cases as C  # noqa: E402
from rag_core import multirag as M  # noqa: E402
from rag_core.helpers import Tee, save, scan  # noqa: E402
from rag_core.pipeline import SteelPCRPipeline, ensure_dir, log  # noqa: E402

CASE_ID = 2

ARM_DESC = ("multi-round retrieval (term optimisation and context expansion): 10 property/mechanism "
            "retrieval intents each retrieved and answered separately, evidence merged round-robin, "
            "one integration prompt")

# ------------------------------------------------------- the 10 retrieval intents
# Each intent locks exactly one variable dimension; the terms are written as Chinese plus
# English/abbreviation to help cross-language retrieval. The digits in formulae (NOx, SO2, H2S, COS,
# NH3, HCN, Fe2O3) name substances, not answers.
SUBS: List[Dict[str, Any]] = [
    {"id": 1, "target": "燃料氮 × NOx 生成",
     "sub_question": "生物质 fuel-N / fuel nitrogen 含量升高对燃烧 NOx 生成的影响"
                      "（torrefied biomass increased fuel-N higher NOx）"},
    {"id": 2, "target": "脱灰除钾 × NOx",
     "sub_question": "ashless treatment 脱除碱金属/钾 potassium removal 对 NOx 排放与未燃碳 UBC 的影响"
                      "（ashless kenaf potassium removal NOx）"},
    {"id": 3, "target": "AAEM × 渣系/结渣",
     "sub_question": "碱金属与碱土金属 AAEM (K, Na, Zn) 在生物质灰分中对 slagging fouling、"
                      "焦炭反应性、渣系与铁水质量的影响"},
    {"id": 4, "target": "灰分/热值 × 替代比例约束",
     "sub_question": "生物质炭灰分 ash content 与热值 HHV 对高炉喷吹替代比例上限的约束"
                      "（ash content window、calorific value window）"},
    {"id": 5, "target": "入炉碱金属负荷限值",
     "sub_question": "高炉入炉碱金属负荷 alkali load 限值（总碱金属氧化物 kg/t 铁）与硫负荷 limit"},
    {"id": 6, "target": "硫含量 × SO2",
     "sub_question": "生物质炭硫含量 sulfur content 低对高炉 SO2 排放的影响"
                      "（biomass low sulfur SO2 reduction）"},
    {"id": 7, "target": "挥发分 × 燃烧/NOx",
     "sub_question": "生物质炭挥发分 volatile matter 与固定碳对喷吹燃烧燃尽与 NOx 的影响"
                      "（high volatile biochar combustion burnout）"},
    {"id": 8, "target": "高替代比例可行性",
     "sub_question": "生物质炭高替代比例喷吹的可行性边界与工业试验证据"
                      "（high replacement ratio biomass injection feasibility）"},
    {"id": 9, "target": "降碳-污排权衡",
     "sub_question": "生物质喷吹降碳与污染物负荷增加的权衡或排放转移证据"
                      "（decarbonization versus pollutant burden shifting biomass ironmaking）"},
    {"id": 10, "target": "预处理工艺",
     "sub_question": "torrefaction 烘焙 / pyrolysis 热解 / ashless 脱灰预处理对生物质燃料品质"
                      "与可磨性 HGI 的改善"},
]

# Fingerprint strings used for post-hoc checks only: they confirm that the target evidence reached the
# retrieval layer. They never enter a query or a prompt.
HIT_KWS = ["fuel-N", "fuel nitrogen", "ashless", "potassium", "43%"]


def main() -> None:
    ensure_dir(C.OUT_ROOT)
    log_path = os.path.join(C.OUT_ROOT, f"case{CASE_ID}_C.log")
    with Tee(log_path):
        case = C.get_case(CASE_ID)
        q = C.build_complex_question(case)

        log("=" * 78)
        log(f"case {CASE_ID} multi-round retrieval (term optimisation and context expansion)")
        log("=" * 78)

        pipe = SteelPCRPipeline(enable_expansion=True)
        res = M.run_multi(
            CASE_ID, pipe, q,
            subs=SUBS,
            top_k=M.SUB_K,
            subpoint=True,
            subpoint_per_doc=M.SUB_PER_DOC,
            subpoint_total=M.SUB_TOTAL,
            merge_rule="round_robin",
            merge_top=M.MERGE_TOP,
            per_doc=M.PER_DOC,
            total=M.TOTAL,
            system=M.INTEGRATE_SYSTEM,
            system_label="SUBANSWER_SYSTEM + INTEGRATE_SYSTEM",
            prompt_builder=M.build_prompt_with_table,
            arm_desc=ARM_DESC,
        )

        for st in res["steps"]:
            st["reranked_hits"] = scan(st["reranked_top"], HIT_KWS)
        res["hit_probe"] = {
            "kws": HIT_KWS,
            "rerank_hits_by_sub": {st["sub"]["id"]: st["reranked_hits"]
                                   for st in res["steps"] if st["reranked_hits"]},
            "context_counts": {k: res["context"].count(k) for k in HIT_KWS},
            "merged_hits": scan(res["merged_evidence"], HIT_KWS),
        }
        res = M.with_case_meta(case, res)

        out = os.path.join(C.OUT_ROOT, f"case{CASE_ID}_C.json")
        save(out, res)
        log(f"[save] {out}")


if __name__ == "__main__":
    main()
