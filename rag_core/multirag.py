# -*- coding: utf-8 -*-
"""
Task decomposition and the multi-round retrieval architecture used by the four case scripts.

SteelPCRPipeline answers one question in one round. This module repeats that round over N
retrieval intents and assembles the result:

    decompose()             question -> N single-intent retrieval paths (model-driven)
    retrieve_paths()        one full pipeline round per path
    answer_subpoints()      optional short answer per path, from that path's own evidence
    merge_round_robin()     evidence pool: one chunk per path in turn, capped
    merge_all()             evidence pool: every chunk, deduplicated, uncapped
    integrate()             final generation over the question, the path list and the pool

run_single() is the one-round arm and run_multi() the multi-round arm; with_case_meta() appends
the case metadata that the four scripts report.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Make the repository-level configuration module importable when this package
# is executed from anywhere inside the repository.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag_core.helpers import block_key, parse_json_array, slim, text_of  # noqa: E402
from rag_core.pipeline import (GEN_MODEL, GENERATION_CONTEXT_MAX_CHARS_PER_DOC,  # noqa: E402
                               GENERATION_CONTEXT_MAX_CHARS_TOTAL,
                               HNSW_EF_SEARCH, TOP_K_RERANK, TOP_K_RETRIEVE,
                               build_rag_context, chat, log)

# ==================== Budgets (paper values) ====================
SUB_K = 5                     # chunks the reranker keeps on one retrieval path
MERGE_TOP = 8                 # size of the evidence pool at the integration stage
PER_DOC = 8000                # per-chunk character cap at the integration stage
TOTAL = 30000                 # total character cap at the integration stage
SUB_PER_DOC = 2500            # per-chunk character cap for a single-point answer
SUB_TOTAL = 8000              # total character cap for a single-point answer
NO_CAP = 10 ** 9              # no upper bound on the context

GENERATE_MAX_TOKENS = 32000
SUBANSWER_MAX_TOKENS = 1500

DECOMPOSE_RANGE: Tuple[int, int] = (6, 10)      # what DECOMPOSE_SYSTEM asks for
DECOMPOSE_V2_RANGE: Tuple[int, int] = (5, 14)   # what case 4 accepts from DECOMPOSE_V2

# ==================== Prompts of the multi-round architecture ====================
DECOMPOSE_SYSTEM = (
    "你是检索任务分解器，服务于钢铁行业减污降碳领域的知识库检索增强生成系统。\n"
    "用户会给出一个复杂问题和一个工况组合清单。你的任务是把这个问题拆成若干**单一检索问题**，"
    "让每一个都能独立地在向量知识库里召回 1~3 个段落。\n"
    "\n"
    "拆分规则：\n"
    "1. 每个子问题只针对一个变量维度：一个工况 × 一种污染物 / 一个约束条件 / 一个系统边界 / 一个机理；\n"
    "2. 子问题必须是**事实检索式**，不要包含“是否存在冲突”“请对比并归纳”“权衡关系如何”这类需要跨证据归纳的问法——"
    "归纳属于后面的整合阶段；\n"
    "3. 不要堆叠多个连词或并列对象，一个问题只问一件事；\n"
    "4. 必须覆盖工况清单里的每个工况，并单独设置针对“适宜比例/阈值”“质量或炉况约束”“系统边界”“机理”的子问题；\n"
    "5. 每个子问题都要把关键术语写成“中文 + 英文/缩写”的形式，便于跨语言检索"
    "（例如：钢渣粉/steel slag powder、焦炉煤气/COG/coke oven gas、直接还原铁/DRI）；\n"
    "6. 数量控制在 6~10 个。\n"
    "\n"
    "只输出 JSON 数组，不要任何解释文字。格式：\n"
    '[{"id": 1, "target": "工况B × NOx", "sub_question": "..."}, ...]'
)

SUBANSWER_SYSTEM = (
    "你是钢铁行业减污降碳领域的证据抽取助手。只依据给定的【证据片段】回答子问题。\n"
    "规则：\n"
    "1. 证据能直接回答，就照证据作答，数值/单位/工况条件一律照原文，并标注 [片段N]；\n"
    "2. 证据只能部分回答，就写出能确认的部分，并说明缺什么；\n"
    "3. 证据完全不能回答，直接输出“资料未提供”，禁止用常识补编数值；\n"
    "4. 不超过 120 字，不要输出与子问题无关的背景。"
)

INTEGRATE_SYSTEM = (
    "【系统提示】\n"
    "你是一名钢铁行业减污降碳与清洁生产技术专家。你将收到一个复杂工程问题、该问题被拆解后的子问题清单及其单点作答，"
    "以及检索到的证据片段。请给出最终综合回答。\n"
    "\n"
    "【核心原则：先判证据，再作答】\n"
    "1. 子问题作答只是中间草稿，你可以判定它是否正确、是否切题；**最终结论必须以【证据片段】为准**，"
    "两者冲突时以证据为准并说明。\n"
    "2. 凡证据片段已直接回答的要点（数值、比例、温度、气源、趋势方向、约束条件），必须严格照证据作答：\n"
    "   - 数值、单位、配比、温度、工况参数一律照原文，不得改写、取整、放大或跨工况套用；\n"
    "   - 结论方向必须与证据一致，不得反转或模糊化；\n"
    "   - 每条结论必须绑定适用条件（原料类型 × 工况参数 × 系统边界）并标注 [证据N]；\n"
    "   - 严禁把单一工况结论泛化成普适结论。\n"
    "3. 证据未覆盖的要点可做工程推理，但必须以【外推】标注并说明依据；不得把外推写成证据结论。\n"
    "4. 证据之间若存在方向性冲突或权衡，必须显式指出冲突是否存在、各自成立条件及可能来源。\n"
    "5. 某要点证据完全缺失，就写“资料未提供”，不要用常识猜测冒充实证。\n"
    "\n"
    "【输出结构】\n"
    "一、核心结论：直接判断（含关键数值与成立条件），3~6 句。\n"
    "二、逐工况/逐要点证据锚定：每条写明「证据结论 + 适用条件 + [证据N]」；工况间不一致时并列给出。\n"
    "三、文献分歧与权衡：列方向性冲突或权衡及其条件依赖；确无则写“未见直接冲突证据”。\n"
    "四、证据缺口与外推：哪些要点资料未覆盖、哪些结论属【外推】。\n"
    "五、参考来源清单：把正文真正引用为依据的 [证据N] 按来源文献归并，一篇一行，格式「- 文献标题：k 条（证据N、证据M…）」：\n"
    "   - 文献标题必须逐字复制参考资料证据行中「来源：」之后的文本；该行若含「；章节：」及其后内容，不得复制；不得改写、缩写、翻译或补全标题；\n"
    "   - 同一文献的多条证据合并为一行，k 为去重后的证据编号个数，编号按升序排列；\n"
    "   - 只列正文中真正引用为依据的证据，不列仅转述而未被引用的证据；\n"
    "   - 各行按该文献在正文中首次被引用的顺序排列；\n"
    "   - 本节只做来源归并，不得新增任何结论、数值或判断；正文未引用任何证据时只写「无」。\n"
    "\n"
    "语言精炼、专业、定量。"
)

# Rule 6 of DECOMPOSE_SYSTEM is rewritten to force one path on the paper's own experimental
# conditions and apparatus, and the sub-question count is relaxed to 7-12.
_ANCHOR = "6. 数量控制在 6~10 个。"
assert _ANCHOR in DECOMPOSE_SYSTEM, (
    "the DECOMPOSE_SYSTEM text changed, so the DECOMPOSE_V2 replacement anchor no longer matches")

DECOMPOSE_V2 = DECOMPOSE_SYSTEM.replace(
    _ANCHOR,
    "6. **必须单独设置一路针对「文献自身实验条件与装置设置」的子问题**，用于核对用户给定的工况"
    "在文献中是否真的成立。这类子问题只问条件是怎么设的——样品制备与混合方式、反应器/炉型、"
    "升温程序与终温、恒温时间、载气种类与流量、检测手段；"
    "**不得预设任何具体条件值**（不要写进任何数字或温度，只问\"怎么设的\"）；\n"
    "7. 数量控制在 7~12 个。"
)

# Extra requirements added to the generation prompt
NUM_RULE = (
    "\n\n【补充要求：定量结论不得遗漏】\n"
    "参考资料中出现的每一个定量结论（百分比、比例、降幅、温度、产率、限值、成分含量等），"
    "都必须逐条写进回答，并标注来源编号；不得因为它与当前问题的关联看起来不明显而省略。\n"
    "只有资料中确实一个数值都没有时，才可以写\"资料未提供\"。"
)

COND_RULE = (
    "\n\n【补充要求：数值必须带条件】\n"
    "每一个数值都必须同时写出它的适用条件（原料、配比、温度、气氛、检测方法）；"
    "原文没写条件的，写\"原文未给条件\"。"
)


# ==================== Decomposition ====================
def decompose(
    question: str,
    system: Optional[str] = None,
    count_range: Tuple[int, int] = DECOMPOSE_RANGE,
    retries: int = 3,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split the question into single-intent retrieval paths, retrying when the JSON or the count is off.

    Returns the paths and every raw response, so a run can show what the model actually produced.
    """
    system = DECOMPOSE_SYSTEM if system is None else system
    lo, hi = count_range
    last: Optional[str] = None
    raws: List[str] = []
    for attempt in range(retries):
        raw = chat(system, f"{question}\n\n请输出子问题 JSON 数组。",
                   temperature=0.0, max_tokens=4096)
        raws.append(raw)
        try:
            subs = parse_json_array(raw)
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            log(f"      \u26a0\ufe0f decomposition attempt {attempt + 1} failed to parse: {last}")
            time.sleep(2)
            continue
        if lo <= len(subs) <= hi:
            return subs, raws
        last = f"unexpected number of sub-questions: {len(subs)}"
        log(f"      \u26a0\ufe0f decomposition attempt {attempt + 1}: {last}")
        time.sleep(2)
    raise RuntimeError(f"decomposition failed: {last} ({len(raws)} raw response(s) retained)")


# ==================== One pipeline round per path ====================
def retrieve_paths(
    pipe: Any,
    subs: Sequence[Dict[str, Any]],
    top_k: int = SUB_K,
) -> List[Dict[str, Any]]:
    """Retrieve one path per intent; the step keeps the full records for later assembly."""
    steps: List[Dict[str, Any]] = []
    for s in subs:
        pack = pipe.retrieve(s["sub_question"], top_k=top_k)
        steps.append({
            "sub": s,
            "expanded_query": pack["expanded_query"],
            "reranked_top": pack["reranked"],
            "enhanced": pack["enhanced"],
            "latency_ms": pack["latency_ms"],
        })
        log(f"   {s['id']:>2}. [{s.get('target', '')}] reranked {len(pack['reranked'])} "
            f"-> enhanced {len(pack['enhanced'])}")
    return steps


def answer_subpoints(
    steps: List[Dict[str, Any]],
    per_doc: int = SUB_PER_DOC,
    total: int = SUB_TOTAL,
) -> List[Dict[str, Any]]:
    """Answer every path from its own evidence only; these drafts are intermediate for the next stage."""
    for st in steps:
        ctx = build_rag_context(st["enhanced"], max_chars_per_doc=per_doc,
                                max_chars_total=total)
        if ctx.strip():
            st["sub_context"] = ctx
            st["sub_answer"] = chat(
                SUBANSWER_SYSTEM,
                f"子问题：{st['sub']['sub_question']}\n\n【证据片段】：\n{ctx}\n\n请只依据上述证据片段作答：",
                temperature=0.0, max_tokens=SUBANSWER_MAX_TOKENS)
        else:
            st["sub_context"] = ""
            st["sub_answer"] = "资料未提供"
        log(f"      {st['sub']['id']:>2}. {st['sub_answer'][:70]}")
    return steps


# ==================== Evidence pool ====================
def merge_round_robin(steps: Sequence[Dict[str, Any]], top: int = MERGE_TOP) -> List[Dict[str, Any]]:
    """Take chunks round-robin across the paths, so every path contributes representative evidence."""
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    queues = [[slim(r) for r in st["enhanced"]] for st in steps]
    depth = 0
    while len(merged) < top and any(len(q) > depth for q in queues):
        for q in queues:
            if len(merged) >= top:
                break
            if len(q) > depth:
                cand = q[depth]
                k = block_key(cand)
                if k not in seen:
                    seen.add(k)
                    merged.append(cand)
        depth += 1
    # Top up from the global rerank score when the pool is short
    if len(merged) < top:
        pool = [slim(r) for st in steps for r in st["enhanced"]]
        pool.sort(key=lambda r: float(r.get("_rerank_score", 0.0)), reverse=True)
        for cand in pool:
            if len(merged) >= top:
                break
            k = block_key(cand)
            if k not in seen:
                seen.add(k)
                merged.append(cand)
    return merged


def merge_all(steps: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Keep every chunk from every path, deduplicated by source_file::block_id, order preserved."""
    blocks: List[Dict[str, Any]] = []
    seen: set = set()
    dup = 0
    for st in steps:
        for r in st["enhanced"]:
            k = block_key(r)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            blocks.append(slim(r))
    return blocks, dup


# ==================== Final generation ====================
def build_prompt_with_table(
    question: str,
    steps: Sequence[Dict[str, Any]],
    evidence: str,
) -> str:
    """Integration prompt carrying the per-path drafts as an explicit intermediate layer."""
    tbl = ["【子问题清单与单点作答】"]
    for st in steps:
        sub = st["sub"]
        tbl.append(f"{sub['id']}. [{sub.get('target', '')}] {sub['sub_question']}")
        tbl.append(f"   单点作答：{st['sub_answer']}")
    return (f"【复杂工程问题】\n{question}\n\n" + "\n".join(tbl) +
            "\n\n（注：子问题作答里的 [片段N] 是各子问题自己的证据编号，"
            "最终引用请统一使用下面【证据片段】的 [证据N] 编号。）\n\n"
            f"【证据片段】：\n{evidence}\n\n请综合上述材料给出最终回答：")


def build_prompt_with_list(
    question: str,
    steps: Sequence[Dict[str, Any]],
    evidence: str,
) -> str:
    """Integration prompt listing the paths only: every chunk is passed on unchanged."""
    sub_list = "\n".join(
        ["【已拆解的检索子问题】"] +
        [f"{st['sub']['id']}. [{st['sub'].get('target', '')}] {st['sub']['sub_question']}"
         for st in steps])
    return (f"问题：{question.strip()}\n\n{sub_list}\n\n"
            f"【参考背景资料】（以上子问题各自检索到的全部证据，已去重，未做任何筛选）：\n{evidence}\n\n"
            "请自行判断哪些证据真正切题、适用于什么条件，哪些证据相互冲突或属于不同体系，"
            "并据此给出专业、定量、精炼的回答：")


def integrate(user: str, system: str, max_tokens: int = GENERATE_MAX_TOKENS) -> str:
    return chat(system, user, temperature=0.0, max_tokens=max_tokens)


# ==================== Result assembly ====================
def _slim_steps(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for st in steps:
        d: Dict[str, Any] = {
            "sub": st["sub"],
            "expanded_query": st["expanded_query"],
            "reranked_top": [slim(r) for r in st["reranked_top"]],
            "enhanced": [slim(r) for r in st["enhanced"]],
            "latency_ms": st["latency_ms"],
        }
        for extra in ("sub_context", "sub_answer"):
            if extra in st:
                d[extra] = st[extra]
        out.append(d)
    return out


def with_case_meta(case: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    """Append the case metadata that every script reports."""
    res["case_title"] = case["title"]
    res["scenario"] = case["scenario"]
    res["conditions"] = [{"tag": t, "desc": d} for t, d in case["conditions"]]
    return res


# ==================== The two arms ====================
def run_single(
    case_id: int,
    pipe: Any,
    question: str,
    arm: str = "S",
    arm_desc: str = ("single-round RAG: ask the compound question directly, "
                     "one retrieval round and one generation"),
) -> Dict[str, Any]:
    """One round: expand -> dense top 50 -> rerank top 7 -> context enhancement -> one generation."""
    t0 = time.time()
    pack = pipe.answer(question)
    log(f"done in {time.time() - t0:.1f}s, answer is {len(pack['answer'])} characters")
    return {
        "case_id": case_id,
        "arm": arm,
        "arm_desc": arm_desc,
        "input_question": question,
        "expanded_query": pack["expanded_query"],
        "candidates_top50": [slim(r) for r in pack["candidates"]],
        "reranked_top": [slim(r) for r in pack["reranked"]],
        "enhanced": [slim(r) for r in pack["enhanced"]],
        "context": pack["context"],
        "answer": pack["answer"],
        "latency_ms": pack["latency_ms"],
        "llm_calls": {"retrieval": 1, "generate": 1},
        "config": {
            "top_k_retrieve": TOP_K_RETRIEVE,
            "top_k_rerank": TOP_K_RERANK,
            "hnsw_ef_search": HNSW_EF_SEARCH,
            "context_chars_per_doc": GENERATION_CONTEXT_MAX_CHARS_PER_DOC,
            "context_chars_total": GENERATION_CONTEXT_MAX_CHARS_TOTAL,
            "u_shape_reorder": True,
            "context_enhancement": "expand_window=1 plus paragraph merging and table fencing",
            "bilingual_expansion": True,
            "generator": GEN_MODEL,
            "temperature": 0.0,
            "max_tokens": GENERATE_MAX_TOKENS,
            "system_prompt": "pipeline.SYSTEM_PROMPT (graded evidence)",
        },
    }


def run_multi(
    case_id: int,
    pipe: Any,
    question: str,
    *,
    subs: Optional[Sequence[Dict[str, Any]]] = None,
    decompose_system: Optional[str] = None,
    decompose_range: Tuple[int, int] = DECOMPOSE_RANGE,
    top_k: int = SUB_K,
    subpoint: bool = False,
    subpoint_per_doc: int = SUB_PER_DOC,
    subpoint_total: int = SUB_TOTAL,
    merge_rule: str = "round_robin",
    merge_top: int = MERGE_TOP,
    per_doc: int = PER_DOC,
    total: int = TOTAL,
    system: Optional[str] = None,
    system_label: str = "INTEGRATE_SYSTEM",
    prompt_builder: Callable[[str, Sequence[Dict[str, Any]], str], str] = build_prompt_with_table,
    arm: str = "C",
    arm_desc: str = "",
) -> Dict[str, Any]:
    """Multi-round: decompose -> one round per path -> [per-path answers] -> pool -> integrate.

    `subs` switches off model decomposition and uses the given intents instead; `merge_rule`
    selects the evidence pool ("round_robin" capped, or "all" deduplicated and uncapped).
    """
    t_all = time.time()
    hand_written = subs is not None
    raws: List[str] = []

    if hand_written:
        subs = list(subs)
        log(f"[(1) intents] {len(subs)} hand-written retrieval intents:")
    else:
        subs, raws = decompose(question, decompose_system, decompose_range)
        log(f"[(1) decompose] {len(subs)} retrieval intents:")
    for s in subs:
        log(f"   {s['id']:>2}. [{s.get('target', '')}] {s['sub_question'][:88]}")

    steps = retrieve_paths(pipe, subs, top_k)
    log(f"[(2) retrieve] {len(steps)} paths, "
        f"{sum(len(st['enhanced']) for st in steps)} chunks after enhancement")

    if subpoint:
        log(f"[(3) subpoints] {subpoint_per_doc}/{subpoint_total} characters per path")
        answer_subpoints(steps, subpoint_per_doc, subpoint_total)

    if merge_rule == "round_robin":
        merged, dup = merge_round_robin(steps, merge_top), 0
        merge_desc = f"round-robin across retrieval intents, capped at {merge_top}"
        log(f"[(4) pool] round-robin: {len(merged)} chunks (cap {merge_top})")
    elif merge_rule == "all":
        merged, dup = merge_all(steps)
        merge_desc = "every path kept unchanged after deduplication; unsorted, untruncated, unfiltered"
        log(f"[(4) pool] all chunks: {len(merged)} after deduplication ({dup} duplicates removed)")
    else:
        raise ValueError(f"unknown merge_rule: {merge_rule}")

    ctx = build_rag_context(merged, max_chars_per_doc=per_doc, max_chars_total=total)
    user = prompt_builder(question, steps, ctx)
    system = INTEGRATE_SYSTEM if system is None else system
    answer = integrate(user, system)
    log(f"[(5) generate] input {len(user)} characters -> answer {len(answer)} characters")

    uncapped = per_doc >= NO_CAP and total >= NO_CAP
    res: Dict[str, Any] = {
        "case_id": case_id,
        "arm": arm,
        "arm_desc": arm_desc,
        "input_question": question,
    }
    if not hand_written:
        res["decompose_system"] = DECOMPOSE_SYSTEM if decompose_system is None else decompose_system
        res["decompose_raw"] = raws
    res.update({
        "sub_questions": [{"id": s["id"], "target": s.get("target", ""),
                           "sub_question": s["sub_question"]} for s in subs],
        "steps": _slim_steps(steps),
        "merged_evidence": [slim(r) for r in merged],
        "blocks_sent": len(merged),
        "dup_removed": dup,
        "blocks_chars": sum(len(text_of(r)) for r in merged),
        "context": ctx,
        "context_chars": len(ctx),
        "integrate_prompt_user": user,
        "prompt_chars": len(user),
        "system_prompt": system,
        "answer": answer,
        "latency_ms": {"total": round((time.time() - t_all) * 1000, 1)},
        "llm_calls": {"retrieval": len(subs),
                      "subanswer": len(subs) if subpoint else 0,
                      "integrate": 1},
        "context_budget": {
            "sub_per_doc": subpoint_per_doc if subpoint else None,
            "sub_total": subpoint_total if subpoint else None,
            "per_doc": None if per_doc >= NO_CAP else per_doc,
            "total": None if total >= NO_CAP else total,
        },
        "config": {
            "subs": len(subs),
            "sub_intents": ("hand-written" if hand_written else
                            f"model decomposition, accepted {decompose_range[0]}-{decompose_range[1]}"),
            "top_k_retrieve": TOP_K_RETRIEVE,
            "sub_k_rerank": top_k,
            "merge_rule": merge_desc,
            "context_limit": "no upper bound" if uncapped else f"{per_doc} per chunk / {total} in total",
            "u_shape_reorder": True,
            "context_enhancement": "expand_window=1 plus paragraph merging and table fencing",
            "generator": GEN_MODEL,
            "temperature": 0.0,
            "max_tokens": GENERATE_MAX_TOKENS,
            "system_prompt": system_label,
        },
    })
    return res
