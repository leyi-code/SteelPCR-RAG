# -*- coding: utf-8 -*-
"""
================================================================================
SteelPCR-RAG, Experiment 3: RAG versus native models, and the value of the domain knowledge base
================================================================================

Design notes (the same engineering structure as Experiment 1):
1. The comparison the paper rests on:
   - a lightweight production model with domain RAG (DeepSeek-V4-Flash + RAG) against a stronger
2. The Dense retrieval layer is reused from Experiment 1:
   - candidates_dense.jsonl from Experiment 1 is read verbatim, so the top-50 initial
3. A single reranker, Qwen3-Reranker-8B over HTTP:
   - the top 50 are reranked down to the top 7, then expanded by one adjacent paragraph
4. Generator and judge are strictly separated, so no same-family preference bias can enter:
   - generation: three model slots, the third optional and switched off automatically when unset;
   - judging: GLM-5.3-Flash, an independent cross-family judge.
5. The same resumable checkpointing as Experiment 1:
   - graded re-run diagnosis: LEVEL_2A (evidence intact, only generation lost) versus LEVEL_2B (reranking lost);
   - single-metric repair: only a NaN metric is re-run, healthy fields are never recomputed
   - three independent intermediate slices: candidates_pool, contexts_strict and contexts_pool;
   - atomic row replacement, so one item never leaves duplicate rows in the checkpoint;
   - two-layer validation: fields first, punctuation second;
   - three exported tables: per-item detail, the summary comparison and the stratified breakdowns.
================================================================================
"""
import torch
import os
import sys
import re
import json
import time
import math
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ==============================================================================
# 0. Base paths and root resolution
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# The repository root must be importable before steelpcr_config is used, so this
# file also runs stand-alone from the evaluation/ subdirectory.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from steelpcr_config import env, env_float, env_int  # noqa: E402

# Workspace root. Defaults to the repository root; can be pointed at an
# external working tree that holds the indexes and caches.
_BASE_WORKSPACE_ENV = env("STEELPCR_WORKSPACE_ROOT", "")
BASE_WORKSPACE = Path(_BASE_WORKSPACE_ENV) if _BASE_WORKSPACE_ENV else ROOT_DIR

# The repository root must be importable so this file also runs from the evaluation/ subdirectory
for p in [str(BASE_WORKSPACE), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import rag_core
except ModuleNotFoundError:
   
    pass

# ==============================================================================
# 1. Global configuration
# ==============================================================================

# ---- 1.1 Experiment 1 inputs and intermediate paths ----
EVAL1_CACHE_DIR = BASE_WORKSPACE / "evaluation" / "cache" / "cache1"
DEFAULT_CANDIDATES_DENSE_PATH = str(EVAL1_CACHE_DIR / "candidates_pool" / "candidates_dense.jsonl")

# Dataset path (355 items)
SELECTED_DIR = str(BASE_WORKSPACE / "QA dataset")

# Experiment 3 checkpoint and report directories (mirrors eval1's cache/results layout)
EVAL3_CACHE_DIR = str(BASE_WORKSPACE / "evaluation" / "cache" / "cache3")
EVAL3_OUTPUT_DIR = str(BASE_WORKSPACE / "eval_results" / "results3")

# ---- 1.2 Retrieval and reranking controls ----
TOP_K_RETRIEVE = 50                              # Candidates reused from Experiment 1's Dense top-50
TOP_K_RERANK = 7                                 # Chunks passed to the generator (context K = 7)
ENABLE_CONTEXT_ENHANCEMENT = True                # Enable neighbour merging and formula pull-in (same as Experiment 1)

# ---- 1.3 Reranker API (Qwen3-Reranker-8B by default) ----
QWEN3_RERANK_URL = env("STEELPCR_QWEN3_RERANK_URL", "your api url")
QWEN3_RERANK_KEY = env("STEELPCR_QWEN3_RERANK_KEY", env("STEELPCR_RERANK_API_KEY", "your api key"))
QWEN3_RERANK_MODEL = env("STEELPCR_QWEN3_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
QWEN3_BATCH_SIZE = env_int("STEELPCR_QWEN3_RERANK_BATCH_SIZE", 25)  # Maximum candidates per API batch
RERANK_TIMEOUT = env_float("STEELPCR_RERANK_API_TIMEOUT", 60.0)
RERANK_MAX_RETRIES = env_int("STEELPCR_RERANK_MAX_RETRIES", 2)

# ---- 1.4 Generation models (OpenAI-compatible; three slots) ----
# Slot 1: production model under test, open-book RAG condition
GEN_DS4F_API_BASE = env("STEELPCR_GEN_API_BASE", "https://api.deepseek.com")
GEN_DS4F_API_KEY = env("STEELPCR_GEN_API_KEY", "your api key")
GEN_DS4F_MODEL = env("STEELPCR_GEN_MODEL", "deepSeek-flash")

# Slot 2: closed-book native baseline condition
GEN_DS41F_API_BASE = env("STEELPCR_GEN_STRONG_API_BASE", "https://api.deepseek.com")
GEN_DS41F_API_KEY = env("STEELPCR_GEN_STRONG_API_KEY", "your api key")
GEN_DS41F_MODEL = env("STEELPCR_GEN_STRONG_MODEL", "deepSeek-flash")

# Slot 3: an optional third-party or open-weight model. Leave it empty to skip it; it never blocks the run.
GEN_MODEL3_API_BASE = env("STEELPCR_GEN_EXTRA_API_BASE", "")  # e.g. "https://your-endpoint/v1"
GEN_MODEL3_API_KEY = env("STEELPCR_GEN_EXTRA_API_KEY", "your api key")    # e.g. "sk-..."
GEN_MODEL3_MODEL = env("STEELPCR_GEN_EXTRA_MODEL", "")        # e.g. "qwen2.5-72b-instruct" or "gpt-4o"

GEN_TEMPERATURE = env_float("STEELPCR_GEN_TEMPERATURE", 0.0)  # Deterministic decoding: temperature is pinned to 0.0
GEN_MAX_TOKENS = env_int("STEELPCR_GEN_MAX_TOKENS", 32000)    # A 32k budget leaves room for reasoning plus the full answer
GEN_TIMEOUT = env_float("STEELPCR_GEN_TIMEOUT", 120.0)
GEN_MAX_RETRIES = env_int("STEELPCR_GEN_MAX_RETRIES", 2)

# ---- 1.5 Judge model (GLM, a cross-family judge by default) ----
EVAL_LLM_API_BASE = env("STEELPCR_EVAL_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
EVAL_LLM_API_KEY = env("STEELPCR_EVAL_API_KEY", "your api key")
EVAL_LLM_MODEL = env("STEELPCR_EVAL_MODEL", "glm-5.3-flash")   # Any stable snapshot of the same family works
EVAL_LLM_TIMEOUT = env_float("STEELPCR_EVAL_TIMEOUT", 360.0)
EVAL_MAX_RETRIES = env_int("STEELPCR_EVAL_MAX_RETRIES", 2)

# Evaluation embedding, used by the RAGAS semantic-similarity component (same as Experiment 1)
EVAL_EMBED_API_BASE = env("STEELPCR_EVAL_EMBED_API_BASE", "your api url")
EVAL_EMBED_API_KEY = env("STEELPCR_EVAL_EMBED_API_KEY", env("STEELPCR_EMBED_API_KEY", "your api key"))
EVAL_EMBED_MODEL = env("STEELPCR_EVAL_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
EVAL_EMBED_DIM = env_int("STEELPCR_EVAL_EMBED_DIM", 4096)

# ---- 1.6 Systems compared by default ----
DEFAULT_SYSTEMS = [
    "DSV4F_ClosedBook",
    "DSV4F_DomainRAG",
    "DSV4.1F_ClosedBook",
]

def resolve_active_systems(configured_systems: List[str]) -> List[str]:
    """Enable the model-3 conditions only when GEN_MODEL3_MODEL is set; drop them silently otherwise."""
    active = []
    has_model3 = bool(GEN_MODEL3_MODEL and GEN_MODEL3_MODEL.strip())
    for s in configured_systems:
        if "Model3" in s:
            if has_model3:
                active.append(s)
            else:
                print(f"\u2139\ufe0f GEN_MODEL3_MODEL is unset; skipping condition [{s}]")
        else:
            active.append(s)

    if has_model3 and not any("Model3" in s for s in active):
        active.append("Model3_ClosedBook")
        print(f"\u2705 Third generation model [{GEN_MODEL3_MODEL}] is configured; [Model3_ClosedBook] added to the evaluation matrix.")
    return active

# ==============================================================================
# 2. Topic normalisation. Keys and values stay in Chinese: they must match the dataset's `category` field.
# ==============================================================================
STANDARD_TOPIC_MAP = {
    "三废治理": "Control of Waste Gas, Wastewater and Solid Waste",
    "三废治理（常规与特征污染物排放及处理）": "Control of Waste Gas, Wastewater and Solid Waste",
    "产业耦合": "Industrial Coupling",
    "产业耦合（新能源与跨行业协同）": "Industrial Coupling",
    "末端捕集": "Carbon Capture, Utilization and Storage",
    "末端捕集（CCUS）": "Carbon Capture, Utilization and Storage",
    "源头减排_原料替代": "Source Reduction and Feedstock Substitution",
    "源头减排/原料替代": "Source Reduction and Feedstock Substitution",
    "资源化利用": "Resource Utilization",
    "资源化利用（副产品、固废、水、余热高值利用）": "Resource Utilization",
    "跨维度综合协同": "Integrated Synergies across Dimensions",
    "过程节能减排": "Process Energy Conservation and Emissions Reduction",
    "过程节能减排（含碳排放计算）": "Process Energy Conservation and Emissions Reduction",
    "其他": "Other",
}

def standardize_topic(raw_topic: str) -> str:
    return STANDARD_TOPIC_MAP.get(raw_topic.strip(), "Other")

# ==============================================================================
# 3. Dataset and candidate-pool loading
# ==============================================================================
def load_all_qa_datasets(selected_dir: str = SELECTED_DIR) -> List[Dict[str, Any]]:
    """Recursively read the 355 benchmark items under "QA dataset/"."""
    records = []
    p = Path(selected_dir)
    if not p.exists():
        raise FileNotFoundError(f"Dataset directory not found: {selected_dir}")

    for file_path in sorted(p.rglob("*.jsonl")):
        if "paragraphs" in file_path.parts or file_path.parent == p:
            continue
        rel_parts = file_path.relative_to(p).parts
        task_type = rel_parts[0] if len(rel_parts) > 1 else "unknown"
        raw_topic = file_path.stem

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    std_topic = standardize_topic(data.get("category", raw_topic))
                    records.append({
                        "qid": data.get("qid", f"qa_{len(records)+1:04d}"),
                        "task_type": task_type,
                        "topic": std_topic,
                        "question": data.get("question", "").strip(),
                        "reference_answer": data.get("answer", "").strip(),
                        "gold_paragraph": data.get("paragraph", "").strip(),
                    })
                except Exception as e:
                    print(f"\u26a0\ufe0f Failed to parse QA file {file_path}: {e}")
    print(f"\u2705 Loaded {len(records)} benchmark items from {selected_dir}")
    if any(not r["gold_paragraph"] for r in records):
        print(f"[data] {sum(1 for r in records if not r['gold_paragraph'])}/{len(records)} items have no gold "
              f"evidence text: retrieval metrics need it (see Data availability in the README)")
    return records


def load_dense_candidates_pool(pool_path: str = DEFAULT_CANDIDATES_DENSE_PATH) -> Dict[str, List[Dict[str, Any]]]:
    """Read the Dense top-50 candidate pool persisted by Experiment 1."""
    pool_file = Path(pool_path)
    if not pool_file.exists():
        candidates = [
            BASE_WORKSPACE / "evaluation" / "cache" / "cache1" / "candidates_pool" / "candidates_dense.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "cache1" / "candidates_pool" / "candidates_dense.jsonl",
        ]
        for c in candidates:
            if c.exists():
                pool_file = c
                break

    if not pool_file.exists():
        raise FileNotFoundError(f"\u274c Experiment 1 Dense candidate pool not found: {pool_path}")

    pool_dict = {}
    with open(pool_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get("qid")
            if qid:
                pool_dict[qid] = data.get("candidates", [])
    print(f"\u2705 Loaded the Dense top-50 candidate pool for {len(pool_dict)} questions")
    return pool_dict

# ==============================================================================
# 4. Evidence matching and retrieval metrics (Recall@5, MRR@5, nDCG@5)
# ==============================================================================
def normalize_text_for_match(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())

def is_evidence_match(gold_paragraph: str, retrieved_doc: Dict[str, Any]) -> Tuple[bool, int]:
    """Evidence-matching rule, identical to eval1.py."""
    ret_text = retrieved_doc.get("text", "") or retrieved_doc.get("content", "")
    norm_gold = normalize_text_for_match(gold_paragraph)
    norm_ret = normalize_text_for_match(ret_text)

    if not norm_gold or not norm_ret:
        return False, 0

    if norm_gold in norm_ret or norm_ret in norm_gold:
        return True, 2

    gold_header_match = re.match(r"^（(.*?)）", gold_paragraph)
    if gold_header_match:
        gold_header = gold_header_match.group(1)
        ret_doc_name = str(retrieved_doc.get("source_file", "") or retrieved_doc.get("doc_title", ""))
        if ret_doc_name and ret_doc_name in gold_header:
            body_gold = gold_paragraph[len(gold_header_match.group(0)):].strip()
            norm_body = normalize_text_for_match(body_gold)
            if norm_body and (norm_body[:100] in norm_ret or norm_ret[:100] in norm_body):
                return True, 2

    n = 4
    if len(norm_gold) >= n and len(norm_ret) >= n:
        set_gold = set(norm_gold[i:i+n] for i in range(len(norm_gold)-n+1))
        set_ret = set(norm_ret[i:i+n] for i in range(len(norm_ret)-n+1))
        intersection = len(set_gold & set_ret)
        union = len(set_gold | set_ret)
        overlap = intersection / union if union > 0 else 0.0
        contain_ratio = intersection / min(len(set_gold), len(set_ret)) if min(len(set_gold), len(set_ret)) > 0 else 0.0

        if overlap >= 0.3 or contain_ratio >= 0.6:
            return True, 2
        elif overlap >= 0.15 or contain_ratio >= 0.35:
            return True, 1

    return False, 0

def compute_retrieval_metrics(
    gold_paragraph: str,
    retrieved_docs: List[Dict[str, Any]],
    k: int = TOP_K_RERANK
) -> Dict[str, Optional[float]]:
    """Compute the evidence-coverage metrics over the Qwen3 top-7 ranking."""
    if not (gold_paragraph or "").strip():
        return {f"Recall@{k}": None, f"MRR@{k}": None, f"nDCG@{k}": None}
    top_k_docs = retrieved_docs[:k]
    hits = []
    grades = []
    first_rank = None

    for rank, doc in enumerate(top_k_docs, start=1):
        matched, grade = is_evidence_match(gold_paragraph, doc)
        grades.append(grade)
        if matched:
            hits.append(rank)
            if first_rank is None:
                first_rank = rank

    recall_k = 1.0 if len(hits) > 0 else 0.0
    mrr_k = (1.0 / first_rank) if first_rank is not None else 0.0

    dcg = 0.0
    for r, rel in enumerate(grades, start=1):
        if rel > 0:
            dcg += (2**rel - 1) / np.log2(r + 1)
    idcg = (2**2 - 1) / np.log2(1 + 1)
    ndcg_k = min(1.0, dcg / idcg) if idcg > 0 else 0.0

    return {
        f"Recall@{k}": recall_k,
        f"MRR@{k}": mrr_k,
        f"nDCG@{k}": ndcg_k,
    }

# ==============================================================================
# 5. Qwen3-Reranker-8B client
# ==============================================================================
class Qwen3RerankerAPI:
    """Call a hosted or self-hosted Qwen3-Reranker-8B endpoint to rerank precisely."""
    def __init__(
        self,
        url: str = QWEN3_RERANK_URL,
        api_key: str = QWEN3_RERANK_KEY,
        model: str = QWEN3_RERANK_MODEL,
        batch_size: int = QWEN3_BATCH_SIZE,
        timeout: float = RERANK_TIMEOUT,
        max_retries: int = RERANK_MAX_RETRIES,
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.uses_native_rerank_format = url.rstrip("/").endswith("/text-rerank/text-rerank")

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: int = TOP_K_RERANK) -> Tuple[List[Dict], float, bool, Optional[str]]:
        if not candidates:
            return [], 0.0, True, None

        t0 = time.perf_counter()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }

        scored = []
        is_success = True
        error_msg = None

        for start_idx in range(0, len(candidates), self.batch_size):
            batch = candidates[start_idx : start_idx + self.batch_size]
            docs = [c.get("text", "") or c.get("content", "") for c in batch]

            if self.uses_native_rerank_format:
                payload = {
                    "model": self.model,
                    "input": {"query": query, "documents": docs},
                    "parameters": {"top_n": len(batch), "return_documents": False}
                }
            else:
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": docs,
                    "top_n": len(batch),
                }

            data = None
            last_err = None
            for attempt in range(self.max_retries + 1):
                try:
                    resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                    else:
                        last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
                        if attempt < self.max_retries:
                            time.sleep(2.0)
                except Exception as exc:
                    last_err = str(exc)
                    if attempt < self.max_retries:
                        time.sleep(2.0)

            results_list = []
            if data:
                if "results" in data:
                    results_list = data["results"]
                elif "output" in data and "results" in data["output"]:
                    results_list = data["output"]["results"]

            if results_list:
                for item in results_list:
                    rel_idx = item.get("index")
                    score = float(item.get("relevance_score", 0.0))
                    if 0 <= rel_idx < len(batch):
                        orig_item = dict(batch[rel_idx], _rerank_score=score)
                        scored.append(orig_item)
            else:
                is_success = False
                error_msg = f"Qwen3 rerank batch [{start_idx}:{start_idx+len(batch)}] failed: {last_err}"
                for c in batch:
                    scored.append(dict(c, _rerank_score=0.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        scored.sort(key=lambda c: c.get("_rerank_score", 0.0), reverse=True)
        return scored[:top_k], latency_ms, is_success, error_msg

# ==============================================================================
# 6. Context enhancement and the open-book / closed-book generation engines
# ==============================================================================
def apply_context_enhancement(strict_top5: List[Dict]) -> List[Dict]:
    """Merge and expand the reranked paragraphs (same construction as Experiment 1, at k = 7)."""
    if not ENABLE_CONTEXT_ENHANCEMENT or not strict_top5:
        return strict_top5

    try:
        from rag_core.general_RAG import enhance_context
        return enhance_context(retrieved_records=strict_top5, expand_window=1, fill_internal_gaps=True)
    except Exception as e:
        print(f"      \u26a0\ufe0f Context enhancement failed; falling back to the raw reranked evidence: {e}")
        return list(strict_top5)


def generate_rag_answer(
    question: str,
    enhanced_docs: List[Dict],
    model_name: str,
    api_base: str,
    api_key: str,
    temperature: float = GEN_TEMPERATURE,
    max_tokens: Optional[int] = GEN_MAX_TOKENS,
    max_retries: int = GEN_MAX_RETRIES,
) -> Tuple[str, str]:
    """Open book: call general_RAG.generate_answer directly, so only one prompt definition exists."""
    from openai import OpenAI
    from rag_core.general_RAG import generate_answer

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=GEN_TIMEOUT)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            ans, finish_reason = generate_answer(
                query=question,
                retrieved_records=enhanced_docs,
                client=client,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if ans:
                return ans, finish_reason
            last_err = f"The API returned empty content (finish_reason={finish_reason})"
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(2.0)

    return f"ERROR: generation failed ({last_err})", "error"


def generate_closed_book_answer(
    question: str,
    model_name: str,
    api_base: str,
    api_key: str,
    temperature: float = GEN_TEMPERATURE,
    max_tokens: Optional[int] = GEN_MAX_TOKENS,
    max_retries: int = GEN_MAX_RETRIES,
) -> Tuple[str, str]:
    """Closed book: probe the native model's own domain knowledge and its hallucination rate."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=GEN_TIMEOUT)
    system_prompt = (
        "【系统提示】\n"
        "你是一名专业的钢铁行业减污降碳与清洁生产技术专家。\n"
        "请依据你掌握的专业领域知识，准确、严谨地回答用户的问题。要求：\n"
        "1. 结论明确，事实准确，涉及数值、单位与专业条件时必须严谨；\n"
        "2. 语言简明扼要，直奔核心；\n"
        "3. 严禁凭空捏造虚假参数或标准，若确实不清楚请如实说明。"
    )
    user_prompt = f"问题：{question.strip()}\n\n请依据你掌握的专业知识给出简洁、准确的回答："

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if max_tokens is not None and max_tokens > 4096:
                request_kwargs["max_tokens"] = max_tokens

            resp = client.chat.completions.create(**request_kwargs)
            ans = resp.choices[0].message.content
            ans = ans.strip() if ans is not None else ""
            finish_reason = getattr(resp.choices[0], "finish_reason", "stop") or "stop"
            if ans:
                return ans, finish_reason
            last_err = f"The API returned empty content (finish_reason={finish_reason})"
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(2.0)

    return f"ERROR: generation failed ({last_err})", "error"

# ==============================================================================
# 7. Persisting the intermediate slices (candidates, strict_top5 and contexts are written independently)
# ==============================================================================
def get_candidates_pool_dir(cache_dir: str = EVAL3_CACHE_DIR) -> Path:
    p = Path(cache_dir) / "candidates_pool"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_contexts_pool_dir(cache_dir: str = EVAL3_CACHE_DIR) -> Path:
    p = Path(cache_dir) / "contexts_pool"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_strict_contexts_path(cache_dir: str = EVAL3_CACHE_DIR, sys_name: str = "dsv4f_domainrag") -> Path:
    safe_name = sys_name.lower().replace("-", "_")
    return get_contexts_pool_dir(cache_dir) / f"contexts_strict_{safe_name}.jsonl"

def get_final_contexts_path(cache_dir: str = EVAL3_CACHE_DIR, sys_name: str = "dsv4f_domainrag") -> Path:
    safe_name = sys_name.lower().replace("-", "_")
    return get_contexts_pool_dir(cache_dir) / f"contexts_{safe_name}.jsonl"

def save_intermediate_artifacts(
    cache_dir: str,
    sys_name: str,
    qid: str,
    strict_top5: List[Dict],
    contexts: List[str]
) -> None:
    """Mirror eval1's slice persistence: the strict top-7 and the enhanced context are stored separately."""
    if strict_top5 and len(strict_top5) > 0:
        strict_path = get_strict_contexts_path(cache_dir, sys_name)
        strict_item = {"system": sys_name, "qid": qid, "strict_top5": strict_top5}
        with open(strict_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(strict_item, ensure_ascii=False) + "\n")
            f.flush()

    if contexts and len(contexts) > 0:
        ctx_path = get_final_contexts_path(cache_dir, sys_name)
        ctx_item = {"system": sys_name, "qid": qid, "contexts": contexts}
        with open(ctx_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ctx_item, ensure_ascii=False) + "\n")
            f.flush()

def load_saved_strict_contexts(cache_dir: str = EVAL3_CACHE_DIR, systems: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    records = {}
    target_systems = systems or DEFAULT_SYSTEMS
    for s_name in target_systems:
        p = get_strict_contexts_path(cache_dir, s_name)
        if not p.exists(): continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                records[f"{d.get('system')}_{d.get('qid')}"] = d.get("strict_top5", [])
    return records

def load_saved_final_contexts(cache_dir: str = EVAL3_CACHE_DIR, systems: Optional[List[str]] = None) -> Dict[str, List[str]]:
    records = {}
    target_systems = systems or DEFAULT_SYSTEMS
    for s_name in target_systems:
        p = get_final_contexts_path(cache_dir, s_name)
        if not p.exists(): continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                records[f"{d.get('system')}_{d.get('qid')}"] = d.get("contexts", [])
    return records

# ==============================================================================
# 8. RAGAS scoring and single-metric NaN repair (aligned with eval1.py)
# ==============================================================================
def evaluate_single_ragas_sample(
    sample: Dict[str, Any],
    llm_binding_host: str = EVAL_LLM_API_BASE,
    llm_api_key: str = EVAL_LLM_API_KEY,
    llm_model: str = EVAL_LLM_MODEL,
    embed_binding_host: str = EVAL_EMBED_API_BASE,
    embed_api_key: str = EVAL_EMBED_API_KEY,
    embed_model: str = EVAL_EMBED_MODEL,
    max_retries: int = EVAL_MAX_RETRIES,
) -> Dict[str, float]:
    """RAGAS scoring for one item, in closed-book mode (context metrics skipped) or RAG mode."""
    gen_ans = str(sample.get("generated_answer", "")).strip()
    is_closed = sample.get("is_closed_book", False)

    if not gen_ans or gen_ans.startswith("ERROR:"):
        return {
            "AnswerCorrectness": 0.0,
            "Faithfulness": float("nan") if is_closed else 0.0,
            "ContextPrecision": float("nan") if is_closed else 0.0,
            "ContextRecall": float("nan") if is_closed else 0.0,
        }

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_correctness, faithfulness, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    for attempt in range(max_retries + 1):
        try:
            base_llm = ChatOpenAI(
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_binding_host,
                temperature=0.0,
                timeout=EVAL_LLM_TIMEOUT,
                reasoning_effort="low",
                extra_body={
                    "thinking": {"type": "enabled", "effort": "low"},
                    "reasoning_effort": "low",
                },
            )
            base_embed = OpenAIEmbeddings(
                model=embed_model,
                api_key=embed_api_key,
                base_url=embed_binding_host,
                check_embedding_ctx_length=False,
            )
            eval_llm = LangchainLLMWrapper(base_llm)
            eval_embeddings = LangchainEmbeddingsWrapper(base_embed)

            if is_closed:
                dataset_dict = {
                    "question": [sample["question"]],
                    "answer": [sample["generated_answer"]],
                    "ground_truth": [sample["reference_answer"]],
                }
                metrics_to_eval = [answer_correctness]
            else:
                dataset_dict = {
                    "question": [sample["question"]],
                    "answer": [sample["generated_answer"]],
                    "contexts": [sample["contexts"]],
                    "ground_truth": [sample["reference_answer"]],
                }
                metrics_to_eval = [answer_correctness, faithfulness, context_precision, context_recall]

            hf_dataset = Dataset.from_dict(dataset_dict)
            from ragas.run_config import RunConfig
            # An explicit RunConfig raises Ragas' hard-coded 180 s timeout to 360 s and sets max_workers=1
            ragas_rc = RunConfig(timeout=360, max_retries=2, max_workers=1)
            results = evaluate(
                dataset=hf_dataset,
                metrics=metrics_to_eval,
                llm=eval_llm,
                embeddings=eval_embeddings,
                run_config=ragas_rc,
            )
            df_ragas = results.to_pandas()

            ans_c = float(df_ragas.iloc[0].get("answer_correctness", float("nan")))
            f_val = float("nan") if is_closed else float(df_ragas.iloc[0].get("faithfulness", float("nan")))
            cp_val = float("nan") if is_closed else float(df_ragas.iloc[0].get("context_precision", float("nan")))
            cr_val = float("nan") if is_closed else float(df_ragas.iloc[0].get("context_recall", float("nan")))

            return {
                "AnswerCorrectness": ans_c,
                "Faithfulness": f_val,
                "ContextPrecision": cp_val,
                "ContextRecall": cr_val,
            }
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(2.0)
            else:
                print(f"      \u2139\ufe0f RAGAS scoring raised: {exc}")

    return {
        "AnswerCorrectness": float("nan"),
        "Faithfulness": float("nan"),
        "ContextPrecision": float("nan"),
        "ContextRecall": float("nan"),
    }


def evaluate_single_specific_metric(
    sample: Dict[str, Any],
    metric_name: str,
    llm_binding_host: str = EVAL_LLM_API_BASE,
    llm_api_key: str = EVAL_LLM_API_KEY,
    llm_model: str = EVAL_LLM_MODEL,
    embed_binding_host: str = EVAL_EMBED_API_BASE,
    embed_api_key: str = EVAL_EMBED_API_KEY,
    embed_model: str = EVAL_EMBED_MODEL,
) -> Optional[float]:
    """[Single-metric repair] Re-run one NaN metric only; healthy metrics are never touched."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_correctness, faithfulness, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    metric_map = {
        "AnswerCorrectness": answer_correctness,
        "Faithfulness": faithfulness,
        "ContextPrecision": context_precision,
        "ContextRecall": context_recall,
    }
    target_metric = metric_map.get(metric_name)
    if not target_metric:
        return None

    try:
        base_llm = ChatOpenAI(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_binding_host,
            temperature=0.0,
            timeout=EVAL_LLM_TIMEOUT,
            reasoning_effort="low",
            extra_body={
                "thinking": {"type": "enabled", "effort": "low"},
                "reasoning_effort": "low",
            },
        )
        base_embed = OpenAIEmbeddings(
            model=embed_model,
            api_key=embed_api_key,
            base_url=embed_binding_host,
            check_embedding_ctx_length=False,
        )
        eval_llm = LangchainLLMWrapper(base_llm)
        eval_embeddings = LangchainEmbeddingsWrapper(base_embed)

        is_closed = sample.get("is_closed_book", False)
        if is_closed and metric_name != "AnswerCorrectness":
            return float("nan")

        ans_str = str(sample.get("generated_answer", "")).strip()
        if not ans_str or ans_str.startswith("ERROR:"):
            return 0.0

        ctxs = sample.get("contexts", [])
        if not is_closed and metric_name in ("Faithfulness", "ContextPrecision", "ContextRecall") and not ctxs:
            return 0.0

        dataset_dict = {
            "question": [sample["question"]],
            "answer": [ans_str],
            "ground_truth": [sample["reference_answer"]],
        }
        if not is_closed:
            dataset_dict["contexts"] = [ctxs]

        hf_dataset = Dataset.from_dict(dataset_dict)
        from ragas.run_config import RunConfig
        ragas_rc = RunConfig(timeout=360, max_retries=2, max_workers=1)
        results = evaluate(
            dataset=hf_dataset,
            metrics=[target_metric],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=ragas_rc,
        )
        df_res = results.to_pandas()
        col_name = target_metric.name
        val = float(df_res.iloc[0].get(col_name))
        return val if not math.isnan(val) else None
    except Exception as exc:
        print(f"         \u26a0\ufe0f Re-running metric [{metric_name}] raised: {exc}")
        return None

# ==============================================================================
# 9. Two-layer validation and graded re-run diagnosis (same core logic as eval1.py)
# ==============================================================================
def validate_record_integrity(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validation rules:
    1. Fields: is_success=True, answer non-empty without error, finish_reason != 'length', AnswerCorrectness not NaN
    2. Punctuation: the answer ends with a valid terminal mark, which catches a cut-off stream
    """
    if not record.get("is_success"):
        return False, f"Not marked successful: {record.get('error_msg')}"

    ans = str(record.get("generated_answer", "")).strip()
    if not ans or ans.startswith("ERROR:"):
        return False, "Answer is empty or an ERROR string"

    if record.get("finish_reason") == "length":
        return False, "Truncated at the API token limit (finish_reason=length)"

    ans_corr = record.get("AnswerCorrectness")
    if ans_corr is None or (isinstance(ans_corr, float) and math.isnan(ans_corr)):
        return False, "Primary metric AnswerCorrectness missing or NaN"

    if not record.get("is_closed_book", False):
        faith = record.get("Faithfulness")
        if faith is None or (isinstance(faith, float) and math.isnan(faith)):
            return False, "Faithfulness missing or NaN under the RAG condition"

    normal_endings = ("。", "！", "？", ".", "!", "?", "”", "’", "）", ")", "]", "】", "%")
    if not any(ans.endswith(c) for c in normal_endings):
        return False, f"Answer does not end with valid punctuation: '{ans[-1]}'"

    return True, "Checks passed"


def decide_rerun_level_by_fields(
    raw_record: Optional[Dict[str, Any]],
    saved_candidates: Optional[List[Dict]],
    saved_contexts: Optional[List[str]],
    is_closed_book: bool,
) -> str:
    """
    The same three re-run levels as eval1.py:
    - 'LEVEL_3_FULL_PIPELINE': a brand-new item, no record at all, or no candidates -> run the whole pipeline from scratch;
    - 'LEVEL_2B_RERANK_ONLY': candidates intact but reranking lost -> reuse the top 50 and redo the Qwen3 rerank plus generation;
    - 'LEVEL_2A_GENERATE_ONLY': evidence and context intact, only the answer lost -> reuse the context and redo generation plus grading.
    """
    # No record, no candidate pool, no history: this must be a full LEVEL_3 re-run
    if not raw_record:
        return "LEVEL_3_FULL_PIPELINE"

    if is_closed_book:
        # Closed book has no retrieval path, so any damage means regenerating
        return "LEVEL_2A_GENERATE_ONLY"

    # Under the RAG conditions an absent or empty top-50 pool forces a full LEVEL_3 re-run
    if not saved_candidates or len(saved_candidates) < TOP_K_RERANK:
        return "LEVEL_3_FULL_PIPELINE"

    err_str = str(raw_record.get("error_msg") or "")
    ans = str(raw_record.get("generated_answer") or "").strip()
    is_succ = raw_record.get("is_success")

    # 1. Reranking lost (error_msg records a rerank failure)
    is_rerank_damaged = "rerank" in err_str.lower()
    if is_rerank_damaged:
        return "LEVEL_2B_RERANK_ONLY"

    # 2. Generation lost (empty answer, truncation, stream cut before punctuation)
    normal_endings = ("。", "！", "？", ".", "!", "?", "”", "’", "）", ")", "]", "】", "%")
    is_generation_damaged = (
        not is_succ
        or "empty content" in err_str
        or "502" in err_str
        or not ans
        or ans.startswith("ERROR:")
        or not any(ans.endswith(c) for c in normal_endings)
    )

    if is_generation_damaged:
        if saved_contexts and len(saved_contexts) >= 1:
            return "LEVEL_2A_GENERATE_ONLY"
        return "LEVEL_2B_RERANK_ONLY"

    return "LEVEL_3_FULL_PIPELINE"


def get_checkpoint_path(cache_dir: str = EVAL3_CACHE_DIR) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "eval3_checkpoint_records.jsonl"


def load_all_checkpoint_records(cache_dir: str = EVAL3_CACHE_DIR) -> Dict[str, Dict[str, Any]]:
    ckpt_path = get_checkpoint_path(cache_dir)
    records = {}
    if not ckpt_path.exists():
        return records
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            key = f"{d.get('system')}_{d.get('qid')}"
            records[key] = d
    return records


def build_skip_cache(all_records: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Only records passing both validation layers enter the skip set."""
    skip_dict = {}
    for key, d in all_records.items():
        is_valid, _ = validate_record_integrity(d)
        if is_valid:
            skip_dict[key] = d
    print(f"\u2139\ufe0f Skip set built: {len(skip_dict)} fully healthy record(s) can be reused instantly")
    return skip_dict


def update_or_append_checkpoint_record(cache_dir: str, record: Dict[str, Any]):
    """Atomic row replacement, as in eval1.py: stale rows are overwritten in place and new
    rows are appended, so the checkpoint never holds duplicates."""
    ckpt_path = get_checkpoint_path(cache_dir)
    if not ckpt_path.exists():
        with open(ckpt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    temp_path = ckpt_path.with_suffix(".tmp")
    target_key = f"{record.get('system')}_{record.get('qid')}"
    replaced = False

    with open(ckpt_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line_str = line.strip()
            if not line_str: continue
            d = json.loads(line_str)
            curr_key = f"{d.get('system')}_{d.get('qid')}"
            if curr_key == target_key:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                replaced = True
            else:
                f_out.write(line_str + "\n")
        if not replaced:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_out.flush()

    os.replace(temp_path, ckpt_path)


def repair_nan_checkpoint_records(cache_dir: str = EVAL3_CACHE_DIR) -> int:
    """Streaming single-metric repair as in eval1.py: NaN metrics are patched in place, under 1 MB."""
    ckpt_path = get_checkpoint_path(cache_dir)
    if not ckpt_path.exists():
        return 0

    temp_path = ckpt_path.with_suffix(".tmp")
    repaired_count = 0

    with open(ckpt_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            has_ans = (
                bool(d.get("is_success"))
                and bool(d.get("generated_answer"))
                and not str(d.get("generated_answer")).startswith("ERROR:")
                and d.get("finish_reason") != "length"
            )
            
            if has_ans:
                target_keys = ["AnswerCorrectness"] if d.get("is_closed_book") else ["AnswerCorrectness", "Faithfulness", "ContextPrecision", "ContextRecall"]
                for k in target_keys:
                    val = d.get(k)
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        print(f"   \U0001f527 [{d.get('system')}_{d.get('qid')}] metric [{k}] is NaN; re-running it ...")
                        new_val = evaluate_single_specific_metric(d, metric_name=k)
                        if new_val is not None:
                            d[k] = new_val
                            repaired_count += 1
                            print(f"      \u2705 [{k}] repaired: {new_val}")
            else:
                # Rows whose answer is damaged or missing cannot be repaired by a single metric;
                pass

            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
            f_out.flush()

    os.replace(temp_path, ckpt_path)
    if repaired_count > 0:
        print(f"\u2705 In-place NaN repair finished: {repaired_count} metric value(s) repaired.")
    return repaired_count

# ==============================================================================
# 10. Experiment 3 main engine
# ==============================================================================
def run_eval3_experiment(
    systems_to_evaluate: List[str] = DEFAULT_SYSTEMS,
    candidates_pool_path: str = DEFAULT_CANDIDATES_DENSE_PATH,
    cache_dir: str = EVAL3_CACHE_DIR,
    output_dir: str = EVAL3_OUTPUT_DIR,
):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_all_qa_datasets(SELECTED_DIR)
    dense_candidates_pool = load_dense_candidates_pool(candidates_pool_path)

    # 1. Repair NaN metrics already stored in the checkpoint
    repair_nan_checkpoint_records(cache_dir)
    all_raw_ckpts = load_all_checkpoint_records(cache_dir)
    skip_cache = build_skip_cache(all_raw_ckpts)

    # Load any previously persisted evidence context
    saved_strict_pool = load_saved_strict_contexts(cache_dir, systems_to_evaluate)
    saved_final_pool = load_saved_final_contexts(cache_dir, systems_to_evaluate)

    reranker = Qwen3RerankerAPI()
    dense_retriever_instance = None
    all_details = []

    actual_systems = resolve_active_systems(systems_to_evaluate)

    print(f"\n================================================================================")
    print(f"\U0001f680 Running Experiment 3: RAG versus native models, and the value of the knowledge base")
    print(f"   Evaluation items: {len(dataset)}")
    print(f"   Conditions: {actual_systems}")
    print(f"   Script: {Path(__file__).resolve()}")
    print(f"   Retrieval: reuse Experiment 1's Dense top-50 pool; items missing from the checkpoint are re-searched live")
    print(f"   Reranker: {QWEN3_RERANK_MODEL} (API: {QWEN3_RERANK_URL})")
    print(f"   Judge model: {EVAL_LLM_MODEL} (API: {EVAL_LLM_API_BASE})")
    print(f"================================================================================\n")

    # ---- Route preview before any API budget is spent ----
    print("\U0001f50e Resume-range preview (check this before spending API budget):")
    for s_name in actual_systems:
        s_closed = "ClosedBook" in s_name
        exist_cnt = sum(1 for qa in dataset if f"{s_name}_{qa['qid']}" in all_raw_ckpts)
        skip_cnt = sum(1 for qa in dataset if f"{s_name}_{qa['qid']}" in skip_cache)
        run_cnt = len(dataset) - skip_cnt
        print(f"   [{s_name}] {exist_cnt} record(s) in the checkpoint | {skip_cnt} skip instantly | {run_cnt} to run now")
        if (not s_closed) and exist_cnt > 0 and skip_cnt == 0:
            print(f"   \u26a0\ufe0f WARNING: none of the {exist_cnt} existing [{s_name}] records can be skipped; this would re-run everything.")
            print(f"      To re-run only the low-scoring items, stop now and keep just those rows in the checkpoint.")
    print("")

    for sys_name in actual_systems:
        print(f"\n---> [evaluating condition]: {sys_name}")

        is_closed_book = "ClosedBook" in sys_name
        # Branch on the model configuration
        if "Model3" in sys_name:
            gen_model = GEN_MODEL3_MODEL
            gen_base = GEN_MODEL3_API_BASE
            gen_key = GEN_MODEL3_API_KEY
        elif "DSV4.1F" in sys_name:
            gen_model = GEN_DS41F_MODEL
            gen_base = GEN_DS41F_API_BASE
            gen_key = GEN_DS41F_API_KEY
        else:
            gen_model = GEN_DS4F_MODEL
            gen_base = GEN_DS4F_API_BASE
            gen_key = GEN_DS4F_API_KEY

        for i, qa in enumerate(dataset):
            qid = qa["qid"]
            ckpt_key = f"{sys_name}_{qid}"

            # ----------------------------------------------------
            # Step 1: consult the skip set
            # Any record that passes both validation layers is skipped instantly, with no
            # re-retrieval and no regeneration; only records absent from the checkpoint are re-run.
            # ----------------------------------------------------
            if ckpt_key in skip_cache:
                saved = skip_cache[ckpt_key]
                all_details.append(saved)
                if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                    print(f"   [{sys_name}] {i + 1}/{len(dataset)} done (skipped instantly)")
                continue

            # ----------------------------------------------------
            # Step 2: graded re-run diagnosis
            # ----------------------------------------------------
            raw_hist = all_raw_ckpts.get(ckpt_key)
            # Reuse the Dense top-50 pool from Experiment 1. Only when a record is missing or
            # damaged is the item re-searched live in the LEVEL_3 branch below.
            cand_50 = dense_candidates_pool.get(qid, [])
            cached_ctx = saved_final_pool.get(ckpt_key)
            rerun_lvl = decide_rerun_level_by_fields(raw_hist, cand_50, cached_ctx, is_closed_book)

            t0 = time.perf_counter()
            eval_contexts = []
            enhanced_docs = []
            strict_top5 = []
            rerank_latency_ms = 0.0

            if not is_closed_book:
                # [Level A] evidence intact, only the answer lost -> reuse the top 7, skip retrieval and reranking
                if rerun_lvl == "LEVEL_2A_GENERATE_ONLY" and cached_ctx:
                    print(f"   \u26a1 [{ckpt_key}] LEVEL_2A: reusing the stored top-7 evidence, skipping rerank and retrieval ...")
                    strict_top5 = saved_strict_pool.get(ckpt_key, [])
                    # RAGAS scores against the raw reranked chunks
                    eval_contexts = [d.get("text", "") or d.get("content", "") for d in strict_top5 if (d.get("text") or d.get("content"))]
                    if not eval_contexts:
                        eval_contexts = cached_ctx
                    # Rebuild the enhanced context for the generator
                    enhanced_docs = apply_context_enhancement(strict_top5)

                # [Level B] candidates exist, only reranking lost -> reuse the top 50 and rerun Qwen3
                elif rerun_lvl == "LEVEL_2B_RERANK_ONLY" and cand_50:
                    print(f"   \u26a1 [{ckpt_key}] LEVEL_2B: reusing the stored top-50 pool and re-running only the Qwen3 rerank ...")
                    strict_top5, rerank_latency_ms, is_succ_rerank, err_rerank = reranker.rerank(
                        query=qa["question"], candidates=cand_50, top_k=TOP_K_RERANK
                    )
                    eval_contexts = [d.get("text", "") or d.get("content", "") for d in strict_top5 if (d.get("text") or d.get("content"))]
                    enhanced_docs = apply_context_enhancement(strict_top5)
                    save_intermediate_artifacts(cache_dir, sys_name, qid, strict_top5, eval_contexts)

                # [Level C / LEVEL_3] no record, no candidates -> run the full online retrieve-rerank path
                else:
                    print(f"   \U0001f504 [{ckpt_key}] LEVEL_3_FULL_PIPELINE: running live Dense retrieval plus the Qwen3 rerank ...")
                    if dense_retriever_instance is None:
                        try:
                            from rag_core.general_RAG import DenseRetriever
                            dense_dir = str(BASE_WORKSPACE / "rag_core" / "cache" / "dense")
                            if not Path(dense_dir).exists():
                                dense_dir = str(ROOT_DIR / "cache" / "dense")
                            dense_retriever_instance = DenseRetriever(index_dir=dense_dir)
                        except Exception as e:
                            print(f"      \u26a0\ufe0f Initialising DenseRetriever raised: {e}")

                    if dense_retriever_instance is not None:
                        # 1. Prefer the O(1) static expansion; on a miss the LLM generates one and it is used
                        from rag_core.general_RAG import get_expanded_query
                        search_q = get_expanded_query(qid, qa["question"])
                        cand_50 = dense_retriever_instance.retrieve(search_q, top_k=TOP_K_RETRIEVE)
                        dense_candidates_pool[qid] = cand_50
                        # 2. Persist this run's candidate pool separately, with the same labelling as Experiment 1
                        try:
                            with open(candidates_pool_path, "a", encoding="utf-8") as f_cand:
                                f_cand.write(json.dumps({"system": "dense", "qid": qid, "candidates": cand_50}, ensure_ascii=False) + "\n")
                        except Exception:
                            pass

                    # 3. Run the Qwen3 rerank on the fly
                    if cand_50:
                        strict_top5, rerank_latency_ms, is_succ_rerank, err_rerank = reranker.rerank(
                            query=qa["question"], candidates=cand_50, top_k=TOP_K_RERANK
                        )
                    else:
                        strict_top5 = []
                        print(f"      \u274c [{ckpt_key}] WARNING: Dense retrieval failed and the candidate pool is still empty.")

                    # 4. Enhance and persist the slices; both the slices and the RAGAS context come from the raw reranked chunks
                    eval_contexts = [d.get("text", "") or d.get("content", "") for d in strict_top5 if (d.get("text") or d.get("content"))]
                    enhanced_docs = apply_context_enhancement(strict_top5)
                    save_intermediate_artifacts(cache_dir, sys_name, qid, strict_top5, eval_contexts)

            # Generate: open book uses enhanced_docs with metadata injected, closed book uses the bare prompt
            if is_closed_book:
                gen_ans, finish_reason = generate_closed_book_answer(
                    question=qa["question"],
                    model_name=gen_model,
                    api_base=gen_base,
                    api_key=gen_key,
                    temperature=GEN_TEMPERATURE,
                )
            else:
                gen_ans, finish_reason = generate_rag_answer(
                    question=qa["question"],
                    enhanced_docs=enhanced_docs,
                    model_name=gen_model,
                    api_base=gen_base,
                    api_key=gen_key,
                    temperature=GEN_TEMPERATURE,
                )

            total_latency_ms = (time.perf_counter() - t0) * 1000.0
            is_success = not gen_ans.startswith("ERROR:")

            # Retrieval metrics: record the Qwen3 top-7 ordering quality under the RAG conditions
            if not is_closed_book and strict_top5:
                ret_metrics = compute_retrieval_metrics(qa["gold_paragraph"], strict_top5, k=TOP_K_RERANK)
            else:
                ret_metrics = {f"Recall@{TOP_K_RERANK}": 0.0, f"MRR@{TOP_K_RERANK}": 0.0, f"nDCG@{TOP_K_RERANK}": 0.0}

            sample_record = {
                "qid": qid,
                "task_type": qa["task_type"],
                "topic": qa["topic"],
                "system": sys_name,
                "is_closed_book": is_closed_book,
                "generator_model": gen_model,
                "question": qa["question"],
                "reference_answer": qa["reference_answer"],
                "gold_paragraph": qa["gold_paragraph"],
                "generated_answer": gen_ans,
                "contexts": eval_contexts,  # RAGAS scores against the raw reranked chunks
                "finish_reason": finish_reason,
                "is_truncated": (finish_reason == "length"),
                "latency_ms": round(total_latency_ms, 2),
                "rerank_latency_ms": round(rerank_latency_ms, 2),
                "is_success": is_success,
                "is_empty": (len(eval_contexts) == 0) if not is_closed_book else False,
                "error_msg": None if is_success else gen_ans,
                **ret_metrics,
            }

            # ----------------------------------------------------
            # Step 3: score with the independent judge (RAGAS)
            # ----------------------------------------------------
            ragas_metrics = evaluate_single_ragas_sample(sample_record)
            sample_record.update(ragas_metrics)

            # ----------------------------------------------------
            # Step 4: atomic checkpoint update
            # ----------------------------------------------------
            update_or_append_checkpoint_record(cache_dir, sample_record)
            skip_cache[ckpt_key] = sample_record
            all_details.append(sample_record)

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                print(f"   [{sys_name}] {i + 1}/{len(dataset)} done (AnswerCorrectness: {sample_record.get('AnswerCorrectness')})")

    # ==============================================================================
    # 11. Report aggregation and table export
    # ==============================================================================
    df_details = pd.DataFrame(all_details)
    detail_csv = Path(output_dir) / "eval3_all_qa_metrics.csv"
    df_details.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    print(f"\n\u2705 Experiment 3 per-item detail written to: {detail_csv}")

    # 1. Summary comparison table
    summary_rows = []
    for sys_name in actual_systems:
        sub = df_details[df_details["system"] == sys_name]
        if sub.empty: continue

        ans_corr_mean = sub["AnswerCorrectness"].dropna().mean()
        faith_mean = sub["Faithfulness"].dropna().mean() if not sub["Faithfulness"].dropna().empty else float("nan")
        ctx_rec_mean = sub["ContextRecall"].dropna().mean() if not sub["ContextRecall"].dropna().empty else float("nan")
        ctx_prec_mean = sub["ContextPrecision"].dropna().mean() if not sub["ContextPrecision"].dropna().empty else float("nan")

        p50_lat = sub["latency_ms"].quantile(0.50)
        p95_lat = sub["latency_ms"].quantile(0.95)
        succ_rate = (sub["is_success"].sum() / len(sub)) * 100.0

        # Fully-correct rate (>= 0.75) and graded correctness (0/1/2 normalised)
        fully_correct_rate = (sub["AnswerCorrectness"] >= 0.75).sum() / len(sub) * 100.0
        graded_correctness = sub["AnswerCorrectness"].apply(lambda s: 2 if s >= 0.75 else (1 if s >= 0.40 else 0)).sum() / (2 * len(sub))

        summary_rows.append({
            "Condition": sys_name,
            "AnswerCorrectness": round(ans_corr_mean, 4),
            "Fully Correct Rate (%)": round(fully_correct_rate, 2),
            "Graded Correctness": round(graded_correctness, 4),
            "Faithfulness": round(faith_mean, 4) if not math.isnan(faith_mean) else "n/a",
            "ContextRecall": round(ctx_rec_mean, 4) if not math.isnan(ctx_rec_mean) else "n/a",
            "ContextPrecision": round(ctx_prec_mean, 4) if not math.isnan(ctx_prec_mean) else "n/a",
            "Latency P50 (ms)": round(p50_lat, 2),
            "Latency P95 (ms)": round(p95_lat, 2),
            "Success Rate (%)": round(succ_rate, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = Path(output_dir) / "eval3_summary_table.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\u2705 Experiment 3 summary table written to: {summary_csv}")

    # 2. Stratified tables by question type and by domain topic
    strat_rows = []
    for sys_name in actual_systems:
        sub_sys = df_details[df_details["system"] == sys_name]
        if sub_sys.empty: continue
        # By question type
        for t_type, g_type in sub_sys.groupby("task_type"):
            strat_rows.append({
                "Condition": sys_name,
                "Stratification": "Task Type",
                "Category": t_type,
                "N": len(g_type),
                "AnswerCorrectness": round(g_type["AnswerCorrectness"].dropna().mean(), 4),
                "Fully Correct Rate (%)": round((g_type["AnswerCorrectness"] >= 0.75).sum() / len(g_type) * 100.0, 2),
            })
        # By domain topic
        for top, g_top in sub_sys.groupby("topic"):
            strat_rows.append({
                "Condition": sys_name,
                "Stratification": "Domain Topic",
                "Category": top,
                "N": len(g_top),
                "AnswerCorrectness": round(g_top["AnswerCorrectness"].dropna().mean(), 4),
                "Fully Correct Rate (%)": round((g_top["AnswerCorrectness"] >= 0.75).sum() / len(g_top) * 100.0, 2),
            })

    df_strat = pd.DataFrame(strat_rows)
    strat_csv = Path(output_dir) / "eval3_stratified_table.csv"
    df_strat.to_csv(strat_csv, index=False, encoding="utf-8-sig")
    print(f"\u2705 Experiment 3 stratified table written to: {strat_csv}")

    print("\n" + "="*80)
    print("\U0001f4ca Experiment 3 core result matrix")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: RAG versus native models")
    parser.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, help="Conditions to evaluate")
    parser.add_argument("--candidates-pool", default=DEFAULT_CANDIDATES_DENSE_PATH, help="Path of the Dense top-50 candidate pool")
    parser.add_argument("--cache-dir", default=EVAL3_CACHE_DIR, help="Checkpoint directory")
    parser.add_argument("--output-dir", default=EVAL3_OUTPUT_DIR, help="Report output directory")
    args = parser.parse_args()

    run_eval3_experiment(
        systems_to_evaluate=args.systems,
        candidates_pool_path=args.candidates_pool,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
