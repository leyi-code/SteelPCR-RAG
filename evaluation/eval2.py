# torch must be imported on the very first line, before pandas/numpy/spacy, so the
# correct Windows C/CUDA runtime is locked in and the pandas-first import no longer makes
import torch

import os
import sys
import json
import time
import re
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import requests

# ==============================================================================
# 0. Cross-directory imports and automatic root-path resolution
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from steelpcr_config import env, env_float, env_int  # noqa: E402

# ==============================================================================
# ==================== Experiment 2: global paths and core configuration ====================
# (Changing the values in this block retargets the dataset, the cache slices, the API endpoints and the model hyper-parameters.)
# ==============================================================================

# ---- 1. Dataset and output paths ----
SELECTED_DIR = str(ROOT_DIR / "QA dataset")                      # Question-answer pairs (355 items)
EVAL_OUTPUT_DIR = str(ROOT_DIR / "eval_results" / "results2")  # Experiment 2 report output: eval_results/results2/
# Experiment 2 keeps its checkpoints and intermediate slices in their own directory
EVAL_CACHE_DIR = str(SCRIPT_DIR / "cache" / "cache2")          # Experiment 2 cache: evaluation/cache/cache2/

# ---- 2. Assets reused from Experiment 1 (slices 1 and 2A under evaluation/cache/cache1) ----
_EVAL1_CACHE_DIR = SCRIPT_DIR / "cache" / "cache1" if (SCRIPT_DIR / "cache" / "cache1").exists() else (SCRIPT_DIR / "cache")

_CANDIDATES_DEFAULT = _EVAL1_CACHE_DIR / "candidates_pool" / "candidates_hybrid.jsonl"
if not _CANDIDATES_DEFAULT.exists():
    _CANDIDATES_DEFAULT = _EVAL1_CACHE_DIR / "hybrid_top50_candidates_pool.jsonl"

_STRICT_TOP5_DEFAULT = _EVAL1_CACHE_DIR / "contexts_pool" / "contexts_strict_hybrid.jsonl"
_EVAL1_CKPT_DEFAULT = _EVAL1_CACHE_DIR / "eval1_checkpoint_records.jsonl"
_EVAL2_CKPT_DEFAULT = Path(EVAL_CACHE_DIR) / "eval2_checkpoint_records.jsonl"

CANDIDATES_POOL_PATH = str(_CANDIDATES_DEFAULT)   # Slice 1: the Hybrid top-50 candidate pool produced by Experiment 1
STRICT_TOP5_POOL_PATH = str(_STRICT_TOP5_DEFAULT) # Slice 2A: the five chunks BGE reranked from that pool in Experiment 1
EVAL1_CHECKPOINT_PATH = str(_EVAL1_CKPT_DEFAULT)  # Experiment 1 checkpoint; its end-to-end answers and RAGAS scores are reused
EVAL2_CHECKPOINT_PATH = str(_EVAL2_CKPT_DEFAULT)  # Experiment 2 checkpoint, written per item into cache2

# ---- 3. Controlled variables (values reported in SI Text S11) ----
TOP_K_RETRIEVE = 50                               # Candidate budget for the initial search (Candidate K = 50)
TOP_K_RERANK = 5                                  # Chunks passed to the generator (Context K = 5)
DEFAULT_RERANKER_SYSTEMS = ["BGE-reranker-v2-m3", "Cohere-Rank-4-Pro", "Qwen3-Reranker-8B"]

# ---- 4. Reranker API services ----
# 4.1 BGE-reranker-v2-m3 (an OpenAI-compatible /rerank endpoint)
BGE_RERANK_URL = env("STEELPCR_RERANK_API_URL", "your api url")
BGE_RERANK_KEY = env("STEELPCR_RERANK_API_KEY", "your api key")
BGE_RERANK_MODEL = env("STEELPCR_RERANK_API_MODEL", "BAAI/bge-reranker-v2-m3")
BGE_BATCH_SIZE = env_int("STEELPCR_BGE_RERANK_BATCH_SIZE", 25)  # The standard interface caps a batch at 25

# 4.2 Cohere Rank 4 Pro / rerank-v3.5
COHERE_RERANK_URL = env("STEELPCR_COHERE_RERANK_URL", "your api url")
COHERE_RERANK_KEY = env("STEELPCR_COHERE_RERANK_KEY", "your api key")
COHERE_RERANK_MODEL = env("STEELPCR_COHERE_RERANK_MODEL", "cohere/rerank-4-pro")
COHERE_BATCH_SIZE = env_int("STEELPCR_COHERE_RERANK_BATCH_SIZE", 50)

# 4.3 Qwen3-Reranker-8B (a compatible endpoint, standard or vendor-native request format)
# The same API key as the embedding service can be reused
QWEN3_RERANK_URL = env("STEELPCR_QWEN3_RERANK_URL", "your api url")
QWEN3_RERANK_KEY = env("STEELPCR_QWEN3_RERANK_KEY", env("STEELPCR_RERANK_API_KEY", "your api key"))
QWEN3_RERANK_MODEL = env("STEELPCR_QWEN3_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
QWEN3_BATCH_SIZE = env_int("STEELPCR_QWEN3_RERANK_BATCH_SIZE", 25)

RERANK_API_TIMEOUT = env_float("STEELPCR_RERANK_API_TIMEOUT", 60.0)
RERANK_MAX_RETRIES = env_int("STEELPCR_RERANK_MAX_RETRIES", 1)

# ---- 5. Generation model (same model as Experiment 1) ----
GEN_LLM_API_BASE = env("STEELPCR_GEN_API_BASE", "https://api.deepseek.com")
GEN_LLM_API_KEY = env("STEELPCR_GEN_API_KEY", "your api key")
GEN_LLM_MODEL = env("STEELPCR_GEN_MODEL", "deepSeek-flash")
GEN_TEMPERATURE = env_float("STEELPCR_GEN_TEMPERATURE", 0.1)
GEN_MAX_TOKENS = env_int("STEELPCR_GEN_MAX_TOKENS", 32000)
GEN_TIMEOUT = env_float("STEELPCR_GEN_TIMEOUT", 120.0)
GEN_MAX_RETRIES = env_int("STEELPCR_GEN_MAX_RETRIES", 1)

# ---- 6. Judge model and evaluation embedding (same as Experiment 1) ----
EVAL_LLM_API_BASE = env("STEELPCR_EVAL_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
EVAL_LLM_API_KEY = env("STEELPCR_EVAL_API_KEY", "your api key")
EVAL_LLM_MODEL = env("STEELPCR_EVAL_MODEL", "glm-5.3-flash")
EVAL_LLM_TIMEOUT = env_float("STEELPCR_EVAL_TIMEOUT", 120.0)
EVAL_MAX_RETRIES = env_int("STEELPCR_EVAL_MAX_RETRIES", 1)

EVAL_EMBED_API_BASE = env("STEELPCR_EVAL_EMBED_API_BASE", "your api url")
EVAL_EMBED_API_KEY = env("STEELPCR_EVAL_EMBED_API_KEY", env("STEELPCR_EMBED_API_KEY", "your api key"))
EVAL_EMBED_MODEL = env("STEELPCR_EVAL_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

# ==============================================================================
# 1. Topic-name normalisation. Keys and values stay in Chinese: they must match the dataset's `category` field.
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

def normalize_topic(category_str: str) -> str:
    category_str = category_str.strip()
    if category_str in STANDARD_TOPIC_MAP:
        return STANDARD_TOPIC_MAP[category_str]
    clean_cat = re.sub(r"（.*?）|\(.*?\)", "", category_str).strip()
    for key, val in STANDARD_TOPIC_MAP.items():
        if clean_cat.startswith(key):
            return val
    return clean_cat or "Other"

def load_selected_dataset(selected_dir: str = SELECTED_DIR) -> List[Dict[str, Any]]:
    """Load the 355 benchmark question-answer pairs."""
    dataset = []
    base_path = Path(selected_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Selected dataset directory not found: {selected_dir}")

    jsonl_files = sorted(base_path.glob("*/*.jsonl"))
    qid_counter = 1

    for filepath in jsonl_files:
        if "paragraphs" in filepath.parts:
            continue

        task_type = filepath.parent.name
        file_topic = filepath.stem

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_category = data.get("category", file_topic)
                topic = normalize_topic(raw_category)
                question = data.get("question", "").strip()
                answer = data.get("answer", "").strip()
                paragraph = data.get("paragraph", "").strip()

                qid = f"qa_{qid_counter:04d}"
                qid_counter += 1

                dataset.append({
                    "qid": qid,
                    "task_type": task_type,
                    "topic": topic,
                    "raw_category": raw_category,
                    "question": question,
                    "reference_answer": answer,
                    "gold_paragraph": paragraph,
                    "source_file": filepath.name,
                })

    if any(not d["gold_paragraph"] for d in dataset):
        print(f"[data] {sum(1 for d in dataset if not d['gold_paragraph'])}/{len(dataset)} items have no gold "
              f"evidence text: reranking metrics need it (see Data availability in the README)")
    return dataset

# ==============================================================================
# 2. Evidence matching and reranking metrics (nDCG@5, Reranked Evidence Recall@5, MRR@5)
# ==============================================================================
def normalize_text_for_match(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\s\W_]+", "", text.lower())

def is_evidence_match(gold_paragraph: str, retrieved_doc: Dict[str, Any]) -> Tuple[bool, int]:
    """
    Decide whether a chunk matches the gold paragraph; returns (hit, relevance grade 0/1/2).
    0: not relevant, 1: partially relevant, 2: fully and directly supportive
    """
    retrieved_text = retrieved_doc.get("text", "") or retrieved_doc.get("content", "")
    if not gold_paragraph or not retrieved_text:
        return False, 0

    norm_gold = normalize_text_for_match(gold_paragraph)
    norm_ret = normalize_text_for_match(retrieved_text)

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

def compute_rerank_metrics(
    gold_paragraph: str,
    reranked_docs: List[Dict[str, Any]],
    k: int = TOP_K_RERANK
) -> Dict[str, Optional[float]]:
    """
    Compute the reranking metrics:
    - nDCG@5 (primary; three-level graded relevance)
    - Reranked Evidence Recall@5 (primary; whether the gold evidence is inside the top 5)
    - MRR@5 (secondary)
    """
    if not (gold_paragraph or "").strip():
        return {f"nDCG@{k}": None, f"Reranked Evidence Recall@{k}": None, f"MRR@{k}": None}
    top_k_docs = reranked_docs[:k]
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
    
    idcg = (2**2 - 1) / np.log2(1 + 1)  # 3.0
    ndcg_k = min(1.0, dcg / idcg) if idcg > 0 else 0.0

    return {
        "Reranked Evidence Recall@5": recall_k,
        "MRR@5": mrr_k,
        "nDCG@5": ndcg_k,
    }

# ==============================================================================
# 3. Reranker abstraction and one client per provider (requests-based, to avoid an ASN1 crash)
# ==============================================================================
class BaseReranker:
    """Common base interface for every reranker."""
    def __init__(self, name: str):
        self.name = name

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: int = TOP_K_RERANK) -> Tuple[List[Dict], float, bool, Optional[str]]:
        """
        Rerank interface:
        Returns: (the reordered top_k items, rerank_latency_ms, is_success, error_msg)
        """
        raise NotImplementedError

class BGEReranker(BaseReranker):
    """BGE-reranker-v2-m3 through an OpenAI-compatible rerank API (at most 25 items per batch)."""
    def __init__(
        self,
        url: str = BGE_RERANK_URL,
        api_key: str = BGE_RERANK_KEY,
        model: str = BGE_RERANK_MODEL,
        batch_size: int = BGE_BATCH_SIZE,
        timeout: float = RERANK_API_TIMEOUT,
        max_retries: int = RERANK_MAX_RETRIES,
    ):
        super().__init__("BGE-reranker-v2-m3")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: int = TOP_K_RERANK) -> Tuple[List[Dict], float, bool, Optional[str]]:
        if not candidates:
            return [], 0.0, True, None

        t0 = time.perf_counter()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }

        scored_candidates = []
        total_candidates = len(candidates)
        is_success = True
        error_msg = None

        for start_idx in range(0, total_candidates, self.batch_size):
            batch = candidates[start_idx : start_idx + self.batch_size]
            docs = [c.get("text", "") or c.get("content", "") for c in batch]
            payload = {
                "model": self.model,
                "query": query,
                "documents": docs,
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
                        last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
                        if attempt < self.max_retries:
                            time.sleep(2.0)
                except Exception as exc:
                    last_err = str(exc)
                    if attempt < self.max_retries:
                        time.sleep(2.0)

            if data and "results" in data:
                for item in data.get("results", []):
                    rel_idx = item.get("index")
                    score = float(item.get("relevance_score", 0.0))
                    if 0 <= rel_idx < len(batch):
                        orig_item = dict(batch[rel_idx], _rerank_score=score)
                        scored_candidates.append(orig_item)
            else:
                is_success = False
                error_msg = f"BGE batch [{start_idx}:{start_idx+len(batch)}] failed: {last_err}"
                for c in batch:
                    scored_candidates.append(dict(c, _rerank_score=0.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        scored_candidates.sort(key=lambda c: c.get("_rerank_score", 0.0), reverse=True)
        return scored_candidates[:top_k], latency_ms, is_success, error_msg

class CohereReranker(BaseReranker):
    """Cohere Rank 4 Pro / rerank-v3.5 through the official Cohere API or a proxy endpoint."""
    def __init__(
        self,
        url: str = COHERE_RERANK_URL,
        api_key: str = COHERE_RERANK_KEY,
        model: str = COHERE_RERANK_MODEL,
        batch_size: int = COHERE_BATCH_SIZE,
        timeout: float = RERANK_API_TIMEOUT,
        max_retries: int = RERANK_MAX_RETRIES,
    ):
        super().__init__("Cohere-Rank-4-Pro")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: int = TOP_K_RERANK) -> Tuple[List[Dict], float, bool, Optional[str]]:
        if not candidates:
            return [], 0.0, True, None

        t0 = time.perf_counter()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }

        scored_candidates = []
        total_candidates = len(candidates)
        is_success = True
        error_msg = None

        for start_idx in range(0, total_candidates, self.batch_size):
            batch = candidates[start_idx : start_idx + self.batch_size]
            docs = [c.get("text", "") or c.get("content", "") for c in batch]
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
                        last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
                        if attempt < self.max_retries:
                            time.sleep(2.0)
                except Exception as exc:
                    last_err = str(exc)
                    if attempt < self.max_retries:
                        time.sleep(2.0)

            if data and "results" in data:
                for item in data.get("results", []):
                    rel_idx = item.get("index")
                    score = float(item.get("relevance_score", 0.0))
                    if 0 <= rel_idx < len(batch):
                        orig_item = dict(batch[rel_idx], _rerank_score=score)
                        scored_candidates.append(orig_item)
            else:
                is_success = False
                error_msg = f"Cohere batch [{start_idx}:{start_idx+len(batch)}] failed: {last_err}"
                for c in batch:
                    scored_candidates.append(dict(c, _rerank_score=0.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        scored_candidates.sort(key=lambda c: c.get("_rerank_score", 0.0), reverse=True)
        return scored_candidates[:top_k], latency_ms, is_success, error_msg

class Qwen3Reranker(BaseReranker):
    """Qwen3-Reranker-8B through a compatible endpoint (standard and vendor-native request formats supported)."""
    def __init__(
        self,
        url: str = QWEN3_RERANK_URL,
        api_key: str = QWEN3_RERANK_KEY,
        model: str = QWEN3_RERANK_MODEL,
        batch_size: int = QWEN3_BATCH_SIZE,
        timeout: float = RERANK_API_TIMEOUT,
        max_retries: int = RERANK_MAX_RETRIES,
    ):
        super().__init__("Qwen3-Reranker-8B")
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

        scored_candidates = []
        total_candidates = len(candidates)
        is_success = True
        error_msg = None

        for start_idx in range(0, total_candidates, self.batch_size):
            batch = candidates[start_idx : start_idx + self.batch_size]
            docs = [c.get("text", "") or c.get("content", "") for c in batch]

            if self.uses_native_rerank_format:
                payload = {
                    "model": self.model,
                    "input": {
                        "query": query,
                        "documents": docs,
                    },
                    "parameters": {
                        "top_n": len(batch),
                        "return_documents": False,
                    }
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
                        last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
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
                        scored_candidates.append(orig_item)
            else:
                is_success = False
                error_msg = f"Qwen3 batch [{start_idx}:{start_idx+len(batch)}] failed: {last_err}"
                for c in batch:
                    scored_candidates.append(dict(c, _rerank_score=0.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        scored_candidates.sort(key=lambda c: c.get("_rerank_score", 0.0), reverse=True)
        return scored_candidates[:top_k], latency_ms, is_success, error_msg

def build_reranker_instance(system_name: str) -> BaseReranker:
    """Build a reranker instance from its system name."""
    sys_lower = system_name.lower().replace("_", "-")
    if "bge" in sys_lower:
        return BGEReranker()
    elif "cohere" in sys_lower or "rank-4" in sys_lower:
        return CohereReranker()
    elif "qwen" in sys_lower:
        return Qwen3Reranker()
    else:
        raise ValueError(f"Unknown reranker system name: {system_name}")

# ==============================================================================
# 4. End-to-end answer generation and context assembly
# ==============================================================================
def generate_rag_answer(
    question: str,
    retrieved_records: Sequence[Dict[str, Any]],
    api_base: str = GEN_LLM_API_BASE,
    api_key: str = GEN_LLM_API_KEY,
    model: str = GEN_LLM_MODEL,
    max_retries: int = GEN_MAX_RETRIES,
) -> Tuple[str, str]:
    """Open book: call general_RAG.generate_answer directly, so this experiment shares one prompt
    definition and one context-assembly rule with Experiment 1."""
    from openai import OpenAI
    from rag_core.general_RAG import generate_answer

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=GEN_TIMEOUT)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            ans, finish_reason = generate_answer(
                query=question,
                retrieved_records=list(retrieved_records),
                client=client,
                model=model,
                temperature=GEN_TEMPERATURE,
                max_tokens=GEN_MAX_TOKENS,
            )
            if ans:
                return ans, finish_reason
            last_err = "The generation API returned empty content"
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(2.0)

    return f"ERROR: generation failed ({last_err})", "error"

# ==============================================================================
# 5. RAGAS scoring and single-metric NaN repair
# ==============================================================================
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
    """
    [Single-metric re-evaluation]
    Re-run exactly one NaN metric; no other metric is recomputed.
    If the re-run also fails, None is returned and the original value is kept, never 0.0.
    """
    contexts = sample.get("contexts") or []
    if not isinstance(contexts, list):
        contexts = [str(contexts)]

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            faithfulness,
            context_precision,
            context_recall,
        )
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

        base_llm = ChatOpenAI(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_binding_host,
            temperature=0.0,
            timeout=EVAL_LLM_TIMEOUT,
        )
        base_embed = OpenAIEmbeddings(
            model=embed_model,
            api_key=embed_api_key,
            base_url=embed_binding_host,
            check_embedding_ctx_length=False,
        )
        eval_llm = LangchainLLMWrapper(base_llm)
        eval_embeddings = LangchainEmbeddingsWrapper(base_embed)

        dataset_dict = {
            "question": [sample["question"]],
            "answer": [sample["generated_answer"]],
            "contexts": [contexts],
            "ground_truth": [sample["reference_answer"]],
        }
        hf_dataset = Dataset.from_dict(dataset_dict)

        results = evaluate(
            dataset=hf_dataset,
            metrics=[target_metric],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        df_res = results.to_pandas()
        col_name = target_metric.name
        val = float(df_res.iloc[0].get(col_name))
        return val if not math.isnan(val) else None
    except Exception as exc:
        # Degrade gracefully when ragas is not installed and fall back to the semantic rule scorer
        try:
            print(f"         [error] Re-evaluating metric [{metric_name}] raised: {exc}")
        except Exception:
            pass

        # Compute this single metric with the fallback rule
        gen_ans = str(sample.get("generated_answer", "")).strip()
        ref_ans = str(sample.get("reference_answer", "")).strip()
        gold_para = str(sample.get("gold_paragraph", "")).strip()

        if metric_name == "AnswerCorrectness":
            set_gen = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", gen_ans))
            set_ref = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", ref_ans))
            inter = len(set_gen & set_ref)
            union = len(set_gen | set_ref)
            return round(inter / union if union > 0 else 0.0, 4)
        elif metric_name == "Faithfulness":
            set_gen = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", gen_ans))
            all_ctx_text = " ".join(contexts)
            set_ctx = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", all_ctx_text))
            return round((len(set_gen & set_ctx) / len(set_gen)) if len(set_gen) > 0 else 0.0, 4)
        elif metric_name == "ContextRecall":
            all_ctx_text = " ".join(contexts)
            set_ctx = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", all_ctx_text))
            set_gold = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", gold_para))
            return round((len(set_gold & set_ctx) / len(set_gold)) if len(set_gold) > 0 else 0.0, 4)
        elif metric_name == "ContextPrecision":
            matched_blocks = 0
            for c in contexts:
                norm_c = normalize_text_for_match(c)
                norm_g = normalize_text_for_match(gold_para)
                if norm_c and norm_g and (norm_c[:60] in norm_g or norm_g[:60] in norm_c):
                    matched_blocks += 1
            return round(matched_blocks / len(contexts) if len(contexts) > 0 else 0.0, 4)

        return None

def _note_rule_fallback(sample: Dict[str, Any]) -> None:
    """Record every item whose metrics come from the fallback rule instead of RAGAS.

    The fallback keeps a run alive when the judge endpoint fails, but the two scorers must never be
    mixed silently: the affected qids are written to a sidecar file next to the checkpoint.
    """
    qid = str(sample.get("qid") or "").strip()
    if not qid:
        return
    path = Path(EVAL_CACHE_DIR) / "rule_fallback_qids.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        seen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if qid not in seen:
            seen.append(qid)
            path.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"      [scoring] fallback rule used for {len(seen)} item(s) so far; recorded in {path.name}")
    except Exception:
        pass


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
    """Full RAGAS evaluation for one item."""
    gen_ans = str(sample.get("generated_answer", "")).strip()
    if not gen_ans or gen_ans.startswith("ERROR:"):
        return {
            "AnswerCorrectness": 0.0,
            "Faithfulness": 0.0,
            "ContextPrecision": 0.0,
            "ContextRecall": 0.0,
        }

    for attempt in range(max_retries + 1):
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_correctness,
                faithfulness,
                context_precision,
                context_recall,
            )
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            base_llm = ChatOpenAI(
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_binding_host,
                temperature=0.0,
                timeout=EVAL_LLM_TIMEOUT,
            )
            base_embed = OpenAIEmbeddings(
                model=embed_model,
                api_key=embed_api_key,
                base_url=embed_binding_host,
                check_embedding_ctx_length=False,
            )

            eval_llm = LangchainLLMWrapper(base_llm)
            eval_embeddings = LangchainEmbeddingsWrapper(base_embed)

            dataset_dict = {
                "question": [sample["question"]],
                "answer": [sample["generated_answer"]],
                "contexts": [sample["contexts"]],
                "ground_truth": [sample["reference_answer"]],
            }
            hf_dataset = Dataset.from_dict(dataset_dict)

            # All four metrics in one pass; if one await times out, Ragas returns NaN for it
            results = evaluate(
                dataset=hf_dataset,
                metrics=[answer_correctness, faithfulness, context_precision, context_recall],
                llm=eval_llm,
                embeddings=eval_embeddings,
            )
            df_ragas = results.to_pandas()

            raw_metrics = {
                "AnswerCorrectness": float(df_ragas.iloc[0].get("answer_correctness", float("nan"))),
                "Faithfulness": float(df_ragas.iloc[0].get("faithfulness", float("nan"))),
                "ContextPrecision": float(df_ragas.iloc[0].get("context_precision", float("nan"))),
                "ContextRecall": float(df_ragas.iloc[0].get("context_recall", float("nan"))),
            }

            # Same in-place single-metric repair as Experiment 1:
            # a four-metric run occasionally leaves one metric as NaN (TimeoutError in Job[0])
            # when the judge endpoint stalls. Do not discard the whole record; re-run that metric.
            target_metric_map = {
                "AnswerCorrectness": answer_correctness,
                "Faithfulness": faithfulness,
                "ContextPrecision": context_precision,
                "ContextRecall": context_recall,
            }
            for m_key, m_val in raw_metrics.items():
                if math.isnan(m_val) or m_val is None:
                    try:
                        print(f"      [single-metric repair] [{m_key}] timed out / is NaN; re-running it now ...")
                    except Exception:
                        pass
                    fixed_val = evaluate_single_specific_metric(
                        sample,
                        metric_name=m_key,
                        llm_binding_host=llm_binding_host,
                        llm_api_key=llm_api_key,
                        llm_model=llm_model,
                        embed_binding_host=embed_binding_host,
                        embed_api_key=embed_api_key,
                        embed_model=embed_model,
                    )
                    if fixed_val is not None:
                        raw_metrics[m_key] = fixed_val
                        try:
                            print(f"         [repair ok] [{m_key}]: {fixed_val}")
                        except Exception:
                            pass

            return raw_metrics
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(2.0)
            else:
                try:
                    print(f"      [RAGAS fallback] Ragas API error or not installed; scoring with the semantic rule fallback: {exc}")
                except Exception:
                    pass

    # Fallback scoring rule
    _note_rule_fallback(sample)
    ref_ans = sample.get("reference_answer", "")
    contexts = sample.get("contexts", [])
    gold_para = sample.get("gold_paragraph", "")

    set_gen = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", gen_ans))
    set_ref = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", ref_ans))
    inter = len(set_gen & set_ref)
    union = len(set_gen | set_ref)
    ans_corr = inter / union if union > 0 else 0.0

    all_ctx_text = " ".join(contexts)
    set_ctx = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", all_ctx_text))
    faith = (len(set_gen & set_ctx) / len(set_gen)) if len(set_gen) > 0 else 0.0

    set_gold = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", gold_para))
    ctx_rec = (len(set_gold & set_ctx) / len(set_gold)) if len(set_gold) > 0 else 0.0

    matched_blocks = 0
    for c in contexts:
        norm_c = normalize_text_for_match(c)
        norm_g = normalize_text_for_match(gold_para)
        if norm_c and norm_g and (norm_c[:60] in norm_g or norm_g[:60] in norm_c):
            matched_blocks += 1
    ctx_prec = matched_blocks / len(contexts) if len(contexts) > 0 else 0.0

    return {
        "AnswerCorrectness": round(ans_corr, 4),
        "Faithfulness": round(faith, 4),
        "ContextPrecision": round(ctx_prec, 4),
        "ContextRecall": round(ctx_rec, 4),
    }

# ==============================================================================
# 6. Checkpoint validation, atomic overwrite and simplified re-run
# ==============================================================================
def get_checkpoint_path(cache_dir: str = EVAL_CACHE_DIR) -> Path:
    return Path(cache_dir) / "eval2_checkpoint_records.jsonl"

def validate_record_integrity(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Two-layer validation (fields first, punctuation second):
    1. Fields: is_success=True, answer non-empty without error, no NaN among the four metrics
    2. Punctuation: the answer ends with a valid terminal punctuation mark
    """
    if not record.get("is_success"):
        return False, f"Not marked successful: {record.get('error_msg')}"

    ans = str(record.get("generated_answer", "")).strip()
    if not ans or ans.startswith("ERROR:"):
        return False, "Answer is empty or an error string"

    finish_reason = record.get("finish_reason")
    if finish_reason == "length":
        return False, "Truncated at the API token limit (finish_reason=length)"

    target_keys = ["AnswerCorrectness", "Faithfulness", "ContextPrecision", "ContextRecall"]
    for k in target_keys:
        v = record.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False, f"Metric missing or NaN: {k}"

    normal_endings = ("。", "！", "？", ".", "!", "?", "”", "’", "）", ")", "]", "】", "%")
    if not any(ans.endswith(c) for c in normal_endings):
        return False, f"Answer does not end with valid punctuation: '{ans[-1]}'"

    return True, "Checks passed"

def repair_nan_in_eval2_checkpoint(cache_dir: str = EVAL_CACHE_DIR) -> int:
    """
    [Streaming single-metric repair]
    Find which of the four metrics is NaN and re-run exactly those;
    a failed re-run keeps the original value, never 0.0.
    """
    ckpt_path = get_checkpoint_path(cache_dir)
    if not ckpt_path.exists():
        return 0

    temp_path = ckpt_path.with_suffix(".tmp")
    repaired_count = 0
    target_metric_keys = ["AnswerCorrectness", "Faithfulness", "ContextPrecision", "ContextRecall"]

    with open(ckpt_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            has_valid_answer = bool(d.get("is_success")) and bool(d.get("generated_answer")) and not str(d.get("generated_answer")).startswith("ERROR:")
            has_contexts = "contexts" in d and isinstance(d.get("contexts"), list) and len(d.get("contexts")) > 0

            if has_valid_answer and has_contexts:
                for metric_key in target_metric_keys:
                    val = d.get(metric_key)
                    if (val is None) or (isinstance(val, float) and math.isnan(val)):
                        try:
                            print(f"   [NaN repair] [{d.get('system')}_{d.get('qid')}] metric [{metric_key}] is NaN; re-running it ...")
                        except Exception:
                            pass
                        new_val = evaluate_single_specific_metric(d, metric_name=metric_key)
                        if new_val is not None:
                            d[metric_key] = new_val
                            repaired_count += 1
                            try:
                                print(f"      [repair ok] [{metric_key}]: {new_val}")
                            except Exception:
                                pass
                        else:
                            try:
                                print(f"      [incomplete] [{metric_key}] re-run failed; the original value is kept.")
                            except Exception:
                                pass

            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
            f_out.flush()

    os.replace(temp_path, ckpt_path)
    if repaired_count > 0:
        try:
            print(f"[done] Checkpoint NaN repair finished: {repaired_count} metric value(s) repaired.")
        except Exception:
            pass
    return repaired_count

def load_all_raw_checkpoint_records(cache_dir: str = EVAL_CACHE_DIR) -> Dict[str, Dict[str, Any]]:
    """Read every raw record from the checkpoint, including failed rows, for diagnosis."""
    ckpt_path = get_checkpoint_path(cache_dir)
    records = {}
    if not ckpt_path.exists():
        return records
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                key = f"{d.get('system')}_{d.get('qid')}"
                records[key] = d
    except Exception as e:
        print(f"[warning] Failed to read the raw checkpoint: {e}")
    return records

def build_skip_cache(all_records: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    [Building the skip set]
    Two-layer validation: fields first, punctuation second.
    Only records confirmed healthy end to end enter the skip set; everything else must be re-run.
    """
    skip_dict = {}
    for key, d in all_records.items():
        is_valid, _ = validate_record_integrity(d)
        if is_valid:
            skip_dict[key] = d
    print(f"[skip set] built: {len(skip_dict)} fully healthy record(s) will be skipped.")
    return skip_dict

def append_checkpoint_record(cache_dir: str, record: Dict[str, Any]) -> None:
    """
    [Atomic in-place upsert]
    - if a record for the same (system, qid) already exists (a historical error or pre-retry row), it is overwritten in place;
    - otherwise the record is appended at the end;
    - the on-disk .jsonl therefore always holds exactly one clean row per item, with stale rows removed.
    """
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
            if not line_str:
                continue
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

# ==============================================================================
# 7. Persisting the intermediate slices (one top-5 evidence file per reranker)
# ==============================================================================
def get_contexts_pool_dir(cache_dir: str = EVAL_CACHE_DIR) -> Path:
    p = Path(cache_dir) / "contexts_pool"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_strict_contexts_path(cache_dir: str = EVAL_CACHE_DIR, sys_name: str = "cohere") -> Path:
    """Path of the strict top-5 reranked chunks for one reranker (exactly five, kept as an archived asset)."""
    safe_name = sys_name.lower().replace("-", "_")
    return get_contexts_pool_dir(cache_dir) / f"contexts_strict_{safe_name}.jsonl"

def save_intermediate_top5_contexts(
    cache_dir: str,
    sys_name: str,
    qid: str,
    strict_top5: List[Dict],
    contexts: List[str]
) -> None:
    """
    [Persist the top-5 evidence as soon as reranking finishes]
    Whether or not generation or RAGAS later fails or times out, the five chunks chosen by reranking are persisted immediately:
    1. appended to contexts_pool/contexts_strict_{system}.jsonl (the full top-5 dict including scores)
    """
    if strict_top5 and len(strict_top5) > 0:
        strict_path = get_strict_contexts_path(cache_dir, sys_name)
        strict_item = {
            "system": sys_name,
            "qid": qid,
            "strict_top5": strict_top5,
            "contexts": contexts,
        }
        with open(strict_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(strict_item, ensure_ascii=False) + "\n")
            f.flush()

# ==============================================================================
# 8. Loading the assets reused from Experiment 1 (slice 1: top-50 pool; the BGE records)
# ==============================================================================
def load_candidates_pool(pool_path: str = CANDIDATES_POOL_PATH) -> Dict[str, List[Dict]]:
    """Read the 50 raw candidates of all 355 items from slice 1."""
    p = Path(pool_path)
    if not p.exists():
        alt_paths = [
            ROOT_DIR / "cache" / "hybrid_top50_candidates_pool.jsonl",
            ROOT_DIR / "cache" / "candidates_pool" / "candidates_hybrid.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "candidates_pool" / "candidates_hybrid.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "cache1" / "candidates_pool" / "candidates_hybrid.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "hybrid_top50_candidates_pool.jsonl",
        ]
        for alt in alt_paths:
            if alt.exists():
                p = alt
                break

    if not p.exists():
        raise FileNotFoundError(
            f"[missing file] candidate pool not found: {pool_path}. "
            f"Run the Hybrid retrieval of Experiment 1 first so that slice 1 exists."
        )

    pool = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get("qid")
            candidates = data.get("candidates", [])
            if qid:
                pool[qid] = candidates

    print(f"[candidate pool] loaded the Hybrid top-50 initial chunks for {len(pool)} items from {p} (no re-retrieval)")
    return pool

def load_strict_top5_pool(pool_path: str = STRICT_TOP5_POOL_PATH) -> Dict[str, List[Dict]]:
    """
    Read the five paragraphs BGE reranked, from slice 2A of Experiment 1 (contexts_strict_hybrid.jsonl).
    These five chunks are neither merged nor expanded, so nDCG@5, Reranked Evidence Recall@5 and MRR@5 can be computed exactly.
    """
    p = Path(pool_path)
    if not p.exists():
        alt_paths = [
            ROOT_DIR / "cache" / "contexts_pool" / "contexts_strict_hybrid.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "contexts_pool" / "contexts_strict_hybrid.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "cache1" / "contexts_pool" / "contexts_strict_hybrid.jsonl",
        ]
        for alt in alt_paths:
            if alt.exists():
                p = alt
                break

    pool = {}
    if not p.exists():
        print(f"[note] strict top-5 slice not found: {pool_path}; falling back to reranking the candidate pool or to the Experiment 1 checkpoint.")
        return pool

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get("qid")
            top5 = data.get("strict_top5", [])
            if qid and top5:
                pool[qid] = top5

    print(f"[strict top-5 assets] loaded the strict BGE top-5 evidence for {len(pool)} items from {p.name}")
    return pool

def load_bge_records_from_eval1(eval1_ckpt_path: str = EVAL1_CHECKPOINT_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Reuse the 355 end-to-end answers and RAGAS scores for system=='Hybrid' from the Experiment 1 checkpoint.
    """
    p = Path(eval1_ckpt_path)
    if not p.exists():
        alt_paths = [
            ROOT_DIR / "cache" / "eval1_checkpoint_records.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "eval1_checkpoint_records.jsonl",
            ROOT_DIR / "evaluation" / "cache" / "cache1" / "eval1_checkpoint_records.jsonl",
        ]
        for alt in alt_paths:
            if alt.exists():
                p = alt
                break

    bge_records = {}
    if not p.exists():
        print(f"[note] Experiment 1 checkpoint not found at {eval1_ckpt_path}; BGE will be re-run as an ordinary reranker.")
        return bge_records

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if str(d.get("system", "")).lower() == "hybrid":
                qid = d.get("qid")
                item = dict(d)
                item["system"] = "BGE-reranker-v2-m3"
                item["rerank_latency_ms"] = item.get("latency_ms", 0.0)
                bge_records[qid] = item

    print(f"[asset reuse] reused {len(bge_records)} Hybrid (BGE) end-to-end records from Experiment 1")
    return bge_records

# ==============================================================================
# 8. Experiment 2 main engine and the three report tables
# ==============================================================================
def run_evaluation_experiment_2(
    dataset: List[Dict[str, Any]],
    rerankers_to_evaluate: Optional[List[str]] = None,
    output_dir: str = EVAL_OUTPUT_DIR,
    cache_dir: str = EVAL_CACHE_DIR,
    sample_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Experiment 2: compare rerankers on the same fixed Hybrid top-50 candidate set, in ordering and end-to-end QA quality.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if sample_limit is not None and sample_limit > 0:
        dataset = dataset[:sample_limit]
        print(f"[note] sample limit for this run: {sample_limit}")

    if rerankers_to_evaluate is None:
        rerankers_to_evaluate = DEFAULT_RERANKER_SYSTEMS

    # 1. Scan the checkpoint and repair NaN metrics
    repair_nan_in_eval2_checkpoint(cache_dir)
    all_raw_checkpoint_records = load_all_raw_checkpoint_records(cache_dir)

    # 2. Build the skip set (fields then punctuation; only healthy records may be skipped)
    skip_cache = build_skip_cache(all_raw_checkpoint_records)

    # 3. Load the assets that must be reused (top-50 pool, strict top-5 slice, Experiment 1 BGE records)
    candidates_pool = {}
    try:
        candidates_pool = load_candidates_pool(CANDIDATES_POOL_PATH)
    except Exception as e:
        print(f"[note] candidate pool not ready ({e}); if the checkpoint already covers every item the tables can still be built.")

    strict_top5_pool = load_strict_top5_pool(STRICT_TOP5_POOL_PATH)
    eval1_bge_records = load_bge_records_from_eval1(EVAL1_CHECKPOINT_PATH)

    print(f"================================================================================")
    print(f"[start] Experiment 2: reranker comparison on the fixed Hybrid top-50 candidate pool")
    print(f"   Evaluation items: {len(dataset)}")
    print(f"   Rerankers compared: {rerankers_to_evaluate}")
    print(f"   Context top-K: {TOP_K_RERANK}")
    print(f"   Checkpoint: {get_checkpoint_path(cache_dir)}")
    print(f"   Report output: {output_dir}")
    print(f"================================================================================")

    for sys_name in rerankers_to_evaluate:
        print(f"\n---> [evaluating reranker]: {sys_name}")

        is_bge_sys = "bge" in sys_name.lower()
        reranker_instance = None
        if not is_bge_sys or len(eval1_bge_records) == 0:
            reranker_instance = build_reranker_instance(sys_name)

        for i, qa in enumerate(dataset):
            qid = qa["qid"]
            task_type = qa["task_type"]
            topic = qa["topic"]
            question = qa["question"]
            ref_ans = qa["reference_answer"]
            gold_para = qa["gold_paragraph"]

            ckpt_key = f"{sys_name}_{qid}"

            # ==================== Step 1: consult the skip set ====================
            if ckpt_key in skip_cache:
                if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                    print(f"   [{sys_name}] {i + 1}/{len(dataset)} done (restored from checkpoint)")
                continue

            # ==================== Step 2: for BGE, reuse the matching assets ====================
            if is_bge_sys:
                reused_item = eval1_bge_records.get(qid, {})
                reused_top5 = strict_top5_pool.get(qid, [])
                
                # 2.1 Reranking metrics: score the five strict chunks (unmerged) to recompute the graded metrics exactly
                if reused_top5:
                    ret_metrics = compute_rerank_metrics(gold_para, reused_top5, k=TOP_K_RERANK)
                elif reused_item:
                    # Fallback: reuse the Experiment 1 record directly
                    ret_metrics = {
                        "nDCG@5": reused_item.get("nDCG@5", 0.0),
                        "Reranked Evidence Recall@5": reused_item.get("Reranked Evidence Recall@5", reused_item.get("Recall@5", 0.0)),
                        "MRR@5": reused_item.get("MRR@5", 0.0),
                    }
                else:
                    ret_metrics = {"nDCG@5": 0.0, "Reranked Evidence Recall@5": 0.0, "MRR@5": 0.0}

                # 2.2 Generation and RAGAS: reuse the exact same answer text and scores from Experiment 1
                if reused_item:
                    record = {
                        "qid": qid,
                        "task_type": task_type,
                        "topic": topic,
                        "system": sys_name,
                        "question": question,
                        "reference_answer": ref_ans,
                        "gold_paragraph": gold_para,
                        "generated_answer": reused_item.get("generated_answer", ""),
                        "contexts": reused_item.get("contexts", []),
                        "nDCG@5": ret_metrics["nDCG@5"],
                        "Reranked Evidence Recall@5": ret_metrics["Reranked Evidence Recall@5"],
                        "MRR@5": ret_metrics["MRR@5"],
                        "AnswerCorrectness": reused_item.get("AnswerCorrectness", 0.0),
                        "Faithfulness": reused_item.get("Faithfulness", 0.0),
                        "ContextPrecision": reused_item.get("ContextPrecision", 0.0),
                        "ContextRecall": reused_item.get("ContextRecall", 0.0),
                        "rerank_latency_ms": reused_item.get("rerank_latency_ms", reused_item.get("latency_ms", 0.0)),
                        "finish_reason": reused_item.get("finish_reason", "stop"),
                        "is_success": reused_item.get("is_success", True),
                        "error_msg": reused_item.get("error_msg", None),
                    }
                    append_checkpoint_record(cache_dir, record)
                    is_valid, _ = validate_record_integrity(record)
                    if is_valid:
                        skip_cache[ckpt_key] = record
                    if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                        print(f"   [{sys_name}] {i + 1}/{len(dataset)} done (assets aligned and reused)")
                    continue

            # ==================== Step 3: re-run path (simplified: reuse the top-50 data) ====================
            # Experiment 2 always reuses Experiment 1's top-50 candidates, so no recall fallback is needed
            candidates = candidates_pool.get(qid, [])
            if not candidates:
                print(f"[warning] item {qid} has no candidates; skipping reranking.")
                reranked_docs, latency_ms, is_success, err = [], 0.0, False, "no candidates for this item"
            else:
                reranked_docs, latency_ms, is_success, err = reranker_instance.rerank(
                    query=question,
                    candidates=candidates,
                    top_k=TOP_K_RERANK
                )

            contexts = [d.get("text", "") or d.get("content", "") for d in reranked_docs if (d.get("text") or d.get("content"))]

            # 3.1 Engineering safeguard: persist the top-5 evidence the moment reranking returns
            # Whatever happens to generation or the judge API afterwards, the top-5 evidence that already cost API calls is kept
            if is_success and reranked_docs:
                save_intermediate_top5_contexts(
                    cache_dir=cache_dir,
                    sys_name=sys_name,
                    qid=qid,
                    strict_top5=reranked_docs,
                    contexts=contexts,
                )

            # 3.2 Retrieval metrics (nDCG@5, Reranked Evidence Recall@5, MRR@5)
            ret_metrics = compute_rerank_metrics(gold_para, reranked_docs, k=TOP_K_RERANK)

            # 3.3 Short-circuit: if reranking failed, do not call the generator or RAGAS at all
            # The failed row is checkpointed immediately so empty or invalid context is never billed to the LLM
            if not is_success or not contexts:
                gen_answer = f"ERROR: reranking failed ({err})"
                finish_reason = "error"
                ragas_metrics = {
                    "AnswerCorrectness": 0.0,
                    "Faithfulness": 0.0,
                    "ContextPrecision": 0.0,
                    "ContextRecall": 0.0,
                }
            else:
                # The generator is only called once reranking succeeded and produced a valid top-5 context
                gen_answer, finish_reason = generate_rag_answer(question=question, retrieved_records=reranked_docs)
                gen_succ = bool(gen_answer) and not gen_answer.startswith("ERROR:")
                
                if not gen_succ:
                    is_success = False
                    err = f"generation failed: {gen_answer}"
                    ragas_metrics = {
                        "AnswerCorrectness": 0.0,
                        "Faithfulness": 0.0,
                        "ContextPrecision": 0.0,
                        "ContextRecall": 0.0,
                    }
                else:
                    # 3.4 RAGAS is only called when the generated answer is usable
                    eval_sample = {
                        "question": question,
                        "generated_answer": gen_answer,
                        "contexts": contexts,
                        "reference_answer": ref_ans,
                        "gold_paragraph": gold_para,
                    }
                    ragas_metrics = evaluate_single_ragas_sample(eval_sample)

            record = {
                "qid": qid,
                "task_type": task_type,
                "topic": topic,
                "system": sys_name,
                "question": question,
                "reference_answer": ref_ans,
                "gold_paragraph": gold_para,
                "generated_answer": gen_answer,
                "contexts": contexts,
                "nDCG@5": ret_metrics["nDCG@5"],
                "Reranked Evidence Recall@5": ret_metrics["Reranked Evidence Recall@5"],
                "MRR@5": ret_metrics["MRR@5"],
                "AnswerCorrectness": ragas_metrics["AnswerCorrectness"],
                "Faithfulness": ragas_metrics["Faithfulness"],
                "ContextPrecision": ragas_metrics["ContextPrecision"],
                "ContextRecall": ragas_metrics["ContextRecall"],
                "rerank_latency_ms": round(latency_ms, 2),
                "finish_reason": finish_reason,
                "is_success": is_success,
                "error_msg": err,
            }

            # Atomic in-place overwrite of the checkpoint row
            append_checkpoint_record(cache_dir, record)

            # A fully healthy row is added to the skip set
            is_valid, _ = validate_record_integrity(record)
            if is_valid:
                skip_cache[ckpt_key] = record

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                print(f"   [{sys_name}] {i + 1}/{len(dataset)} items evaluated and checkpointed")

    # ==============================================================================
    # 9. Read the full checkpoint and build the three report tables
    # ==============================================================================
    final_records = []
    ckpt_path = get_checkpoint_path(cache_dir)
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_records.append(json.loads(line))

    df_details = pd.DataFrame(final_records)

    # 1. Per-item table: 355 items x 3 rerankers
    detail_csv_path = Path(output_dir) / "eval2_all_qa_metrics.csv"
    df_details.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[saved] Experiment 2 per-item table written to: {detail_csv_path}")

    # 2. Summary table
    summary_rows = []
    for sys_name in rerankers_to_evaluate:
        sub = df_details[df_details["system"] == sys_name]
        if len(sub) == 0:
            continue

        p50_latency = sub["rerank_latency_ms"].quantile(0.50)
        p95_latency = sub["rerank_latency_ms"].quantile(0.95)
        success_rate = (sub["is_success"].sum() / len(sub)) * 100.0

        summary_rows.append({
            "Reranker system": sys_name,
            "nDCG@5": round(sub["nDCG@5"].mean(), 4),
            "Reranked Evidence Recall@5": round(sub["Reranked Evidence Recall@5"].mean(), 4),
            "MRR@5": round(sub["MRR@5"].mean(), 4),
            "AnswerCorrectness": round(sub["AnswerCorrectness"].mean(), 4),
            "Faithfulness": round(sub["Faithfulness"].mean(), 4),
            "ContextPrecision": round(sub["ContextPrecision"].mean(), 4),
            "ContextRecall": round(sub["ContextRecall"].mean(), 4),
            "Rerank Latency P50 (ms)": round(p50_latency, 2),
            "Rerank Latency P95 (ms)": round(p95_latency, 2),
            "Rerank Success Rate (%)": round(success_rate, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = Path(output_dir) / "eval2_summary_table.csv"
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"[saved] Experiment 2 summary table written to: {summary_csv_path}")

    # 3. Stratified table by question type and by domain topic
    stratified_rows = []
    for sys_name in rerankers_to_evaluate:
        for t_type in df_details["task_type"].unique():
            sub = df_details[(df_details["system"] == sys_name) & (df_details["task_type"] == t_type)]
            if len(sub) == 0:
                continue
            stratified_rows.append({
                "Reranker system": sys_name,
                "Stratification": "Task Type",
                "Category": t_type,
                "N": len(sub),
                "nDCG@5": round(sub["nDCG@5"].mean(), 4),
                "Reranked Evidence Recall@5": round(sub["Reranked Evidence Recall@5"].mean(), 4),
                "MRR@5": round(sub["MRR@5"].mean(), 4),
                "AnswerCorrectness": round(sub["AnswerCorrectness"].mean(), 4),
                "Faithfulness": round(sub["Faithfulness"].mean(), 4),
            })

    for sys_name in rerankers_to_evaluate:
        for topic in df_details["topic"].unique():
            sub = df_details[(df_details["system"] == sys_name) & (df_details["topic"] == topic)]
            if len(sub) == 0:
                continue
            stratified_rows.append({
                "Reranker system": sys_name,
                "Stratification": "Domain Topic",
                "Category": topic,
                "N": len(sub),
                "nDCG@5": round(sub["nDCG@5"].mean(), 4),
                "Reranked Evidence Recall@5": round(sub["Reranked Evidence Recall@5"].mean(), 4),
                "MRR@5": round(sub["MRR@5"].mean(), 4),
                "AnswerCorrectness": round(sub["AnswerCorrectness"].mean(), 4),
                "Faithfulness": round(sub["Faithfulness"].mean(), 4),
            })

    df_stratified = pd.DataFrame(stratified_rows)
    stratified_csv_path = Path(output_dir) / "eval2_stratified_table.csv"
    df_stratified.to_csv(stratified_csv_path, index=False, encoding="utf-8-sig")
    print(f"[saved] Experiment 2 stratified table written to: {stratified_csv_path}")

    print("\n" + "="*95)
    print("Experiment 2: end-to-end and ordering metrics by reranker")
    print("="*95)
    print(df_summary.to_string(index=False))
    print("="*95 + "\n")

    return df_details, df_summary, df_stratified

# ==============================================================================
# 10. Command line entry point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 2: reranker comparison (eval2.py)")
    parser.add_argument("--selected-dir", type=str, default=SELECTED_DIR, help="Root directory of the \"QA dataset\" folder")
    parser.add_argument("--output-dir", type=str, default=EVAL_OUTPUT_DIR, help="Report output directory")
    parser.add_argument("--cache-dir", type=str, default=EVAL_CACHE_DIR, help="Checkpoint and intermediate-slice cache directory")
    parser.add_argument("--rerankers", nargs="+", default=DEFAULT_RERANKER_SYSTEMS, help="Rerankers to compare")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of evaluated items (smoke testing)")
    args = parser.parse_args()

    dataset = load_selected_dataset(args.selected_dir)
    print(f"[dataset] loaded {len(dataset)} benchmark items from {args.selected_dir}")

    run_evaluation_experiment_2(
        dataset=dataset,
        rerankers_to_evaluate=args.rerankers,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        sample_limit=args.sample_limit,
    )

if __name__ == "__main__":
    main()
