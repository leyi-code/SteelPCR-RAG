# torch must be imported on the very first line, before pandas/numpy/spacy, so the
# correct Windows C/CUDA runtime is locked in and the pandas-first import no longer makes
import torch

import os
import sys
import json
import time
import re
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

# ==============================================================================
# 0. Cross-directory imports and automatic root-path resolution
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from steelpcr_config import env, env_float, env_int  # noqa: E402

# ==============================================================================
# ==================== Experiment 1: global paths and core configuration ====================
# (Changing the values in this block retargets the dataset, the index directories, the model parameters and the evaluation output.)
# ==============================================================================

# ---- 1. Dataset and output paths ----
SELECTED_DIR = str(ROOT_DIR / "QA dataset")                 # Question-answer pairs (355 items)
EVAL_OUTPUT_DIR = str(ROOT_DIR / "eval_results" / "results1")  # Final report output: eval_results/results1/
EVAL_CACHE_DIR = str(SCRIPT_DIR / "cache"/"cache1")                # Intermediate slices and resume checkpoints: evaluation/cache/cache1/
DEFAULT_SYSTEMS = ["Dense", "Sparse", "Hybrid", "LightRAG-mix"]  # Retrieval paradigms compared by default

# ---- 2. Index locations for the classic paradigms (Dense / Sparse / Hybrid) ----
# Read from rag_core/cache when it exists, otherwise fall back to ROOT_DIR/cache.
_RAG_CORE_CACHE = ROOT_DIR / "rag_core" / "cache"
_DEFAULT_CACHE = _RAG_CORE_CACHE if _RAG_CORE_CACHE.exists() else (ROOT_DIR / "cache")

DENSE_INDEX_DIR = env("STEELPCR_DENSE_INDEX_DIR", str(_DEFAULT_CACHE / "dense"))       # Dense: FAISS-HNSW index directory
SPARSE_INDEX_DIR = env("STEELPCR_SPARSE_INDEX_DIR", str(_DEFAULT_CACHE / "sparse"))     # Sparse: BM25 index directory
EMBEDDING_CACHE_DIR = str(_DEFAULT_CACHE / "embeddings") # Embedding cache directory

# ---- 3. Controlled variables (values reported in SI Text S10) ----
TOP_K_RETRIEVE = 50                              # Candidate budget for the initial search (Candidate K = 50)
TOP_K_RERANK = 5                                 # Chunks passed to the generator (Context K = 5)
ENABLE_CONTEXT_ENHANCEMENT = True                # Enable formula pull-in and adjacent-paragraph merging

# ---- 4. Reranker, called over HTTP (avoids the local PyTorch WinError 1114 issue) ----
RERANK_API_URL = env("STEELPCR_RERANK_API_URL", "your api url") # Remote or local /rerank endpoint
RERANK_API_KEY = env("STEELPCR_RERANK_API_KEY", "your api key")  # API credential
RERANK_API_MODEL = env("STEELPCR_RERANK_API_MODEL", "BAAI/bge-reranker-v2-m3")     # Reranker model name
RERANK_API_TIMEOUT = env_float("STEELPCR_RERANK_API_TIMEOUT", 60.0)
RERANK_MAX_RETRIES = env_int("STEELPCR_RERANK_MAX_RETRIES", 1)  # One retry, i.e. at most two requests in total

# Local cross-encoder fallback (used when the API is disabled)

# ---- 5. LightRAG-mix storage backends (PostgreSQL + Qdrant) ----
LIGHTRAG_WORKING_DIR = str(_DEFAULT_CACHE / "lightRAG")
LIGHTRAG_STORAGE_WORKSPACE = env("STEELPCR_LIGHTRAG_WORKSPACE", "graphrag_cache")
LIGHTRAG_POSTGRES_CONFIG = {
    "host": env("STEELPCR_PG_HOST", "127.0.0.1"),
    "port": env_int("STEELPCR_PG_PORT", 5432),
    "user": env("STEELPCR_PG_USER", "postgres"),
    "password": env("STEELPCR_PG_PASSWORD", ""),
    "database": env("STEELPCR_PG_DATABASE", "lightrag"),
    "workspace": LIGHTRAG_STORAGE_WORKSPACE,
    "max_connections": env_int("STEELPCR_PG_MAX_CONNECTIONS", 50),
}
LIGHTRAG_QDRANT_CONFIG = {
    "url": env("STEELPCR_QDRANT_URL", "http://127.0.0.1:6333"),
    "api_key": env("STEELPCR_QDRANT_API_KEY", ""),
    "workspace": LIGHTRAG_STORAGE_WORKSPACE,
}

# ---- 6. Generation model (produces the end-to-end answer) ----
GEN_LLM_API_BASE = env("STEELPCR_GEN_API_BASE", "https://api.deepseek.com")
GEN_LLM_API_KEY = env("STEELPCR_GEN_API_KEY", "your api key")
GEN_LLM_MODEL = env("STEELPCR_GEN_MODEL", "deepSeek-flash")
GEN_TEMPERATURE = env_float("STEELPCR_GEN_TEMPERATURE", 0.1)
GEN_MAX_TOKENS = env_int("STEELPCR_GEN_MAX_TOKENS", 32000)
GEN_TIMEOUT = env_float("STEELPCR_GEN_TIMEOUT", 120.0)
GEN_MAX_RETRIES = env_int("STEELPCR_GEN_MAX_RETRIES", 1)  # One retry, i.e. at most two requests in total

# ---- 7. Judge model for the RAGAS metrics (AnswerCorrectness / Faithfulness / ...) ----
EVAL_LLM_API_BASE = env("STEELPCR_EVAL_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
EVAL_LLM_API_KEY = env("STEELPCR_EVAL_API_KEY", "your api key")
EVAL_LLM_MODEL = env("STEELPCR_EVAL_MODEL", "glm-5.3-flash")
EVAL_LLM_TIMEOUT = env_float("STEELPCR_EVAL_TIMEOUT", 120.0)
EVAL_MAX_RETRIES = env_int("STEELPCR_EVAL_MAX_RETRIES", 1)  # One retry, i.e. at most two requests in total

EVAL_EMBED_API_BASE = env("STEELPCR_EVAL_EMBED_API_BASE", "your api url")
EVAL_EMBED_API_KEY = env("STEELPCR_EVAL_EMBED_API_KEY", env("STEELPCR_EMBED_API_KEY", "your api key"))
EVAL_EMBED_MODEL = env("STEELPCR_EVAL_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
EVAL_EMBED_DIM = env_int("STEELPCR_EVAL_EMBED_DIM", 4096)


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


# ==============================================================================
# 2. Loading the evaluation dataset
# ==============================================================================
def load_selected_dataset(selected_dir: str = SELECTED_DIR) -> List[Dict[str, Any]]:
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
              f"evidence text: retrieval metrics need it (see Data availability in the README)")
    return dataset


# ==============================================================================
# 3. Evidence similarity and retrieval-hit matching
# ==============================================================================
def normalize_text_for_match(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\s\W_]+", "", text.lower())

def is_evidence_match(gold_paragraph: str, retrieved_doc: Dict[str, Any]) -> Tuple[bool, int]:
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


# ==============================================================================
# 4. Retrieval metrics (Recall@5, MRR@5, nDCG@5)
# ==============================================================================
def compute_retrieval_metrics(
    gold_paragraph: str,
    retrieved_docs: List[Dict[str, Any]],
    k: int = TOP_K_RERANK
) -> Dict[str, Optional[float]]:
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
    
    idcg = (2**2 - 1) / np.log2(1 + 1) # 3.0
    ndcg_k = min(1.0, dcg / idcg) if idcg > 0 else 0.0

    return {
        f"Recall@{k}": recall_k,
        f"MRR@{k}": mrr_k,
        f"nDCG@{k}": ndcg_k,
    }


# ==============================================================================
# 5. Reranker API adapter (requests-based; batches candidates over 25 and retries once)
# ==============================================================================
class APIRerankerWithRetry:
    """Call a Cohere-compatible /rerank API, retrying once on a network timeout.
    Candidates are split into batches of at most 25 to respect the server-side limit.
    """
    def __init__(
        self,
        url: str = RERANK_API_URL,
        api_key: str = RERANK_API_KEY,
        model: str = RERANK_API_MODEL,
        timeout: float = RERANK_API_TIMEOUT,
        max_retries: int = RERANK_MAX_RETRIES,
        batch_size: int = 25,
    ):
        self.url, self.api_key, self.model = url, api_key, model
        self.timeout, self.max_retries = timeout, max_retries
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: Optional[int] = None) -> List[Dict]:
        import requests

        if not candidates:
            return []
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        scored_candidates = []
        total_candidates = len(candidates)

        # Batch the request by batch_size (at most 25) so the API limit is never exceeded
        for start_idx in range(0, total_candidates, self.batch_size):
            batch = candidates[start_idx : start_idx + self.batch_size]
            payload = {
                "model": self.model,
                "query": query,
                "documents": [c.get("text", "") for c in batch],
                "top_n": len(batch),
            }

            data = None
            last_error = None
            total_attempts = self.max_retries + 1
            for attempt in range(total_attempts):
                try:
                    response = requests.post(
                        self.url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout
                    )
                    if response.status_code == 200:
                        data = response.json()
                        break
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        if attempt < self.max_retries:
                            wait_sec = 2.0
                            print(f"      \u26a0\ufe0f Rerank API returned status {response.status_code} (retrying once) in {wait_sec:.1f}s ...")
                            time.sleep(wait_sec)
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        wait_sec = 2.0
                        print(f"      \u26a0\ufe0f Rerank API timeout/error (retrying once) in {wait_sec:.1f}s: {exc}")
                        time.sleep(wait_sec)

            if data is not None and "results" in data:
                for item in data.get("results", []):
                    rel_idx = item.get("index")
                    score = float(item.get("relevance_score", 0.0))
                    if 0 <= rel_idx < len(batch):
                        orig_item = dict(batch[rel_idx], _rerank_score=score)
                        scored_candidates.append(orig_item)
            else:
                print(f"      \u274c Rerank API batch [{start_idx}:{start_idx+len(batch)}] failed: {last_error}; keeping the original order.")
                for c in batch:
                    scored_candidates.append(dict(c, _rerank_score=0.0))

        # Sort globally by descending score and keep the top-K
        scored_candidates.sort(key=lambda c: c.get("_rerank_score", 0.0), reverse=True)
        return scored_candidates[:top_k] if top_k is not None else scored_candidates


# ==============================================================================
# 6. Per-question RAGAS evaluation (one retry)
# ==============================================================================
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
    gen_ans = str(sample.get("generated_answer", "")).strip()
    if not gen_ans or gen_ans.startswith("ERROR:"):
        return {
            "AnswerCorrectness": 0.0,
            "Faithfulness": 0.0,
            "ContextPrecision": 0.0,
            "ContextRecall": 0.0,
        }

    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
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

            results = evaluate(
                dataset=hf_dataset,
                metrics=[answer_correctness, faithfulness, context_precision, context_recall],
                llm=eval_llm,
                embeddings=eval_embeddings,
            )
            df_ragas = results.to_pandas()

            return {
                "AnswerCorrectness": float(df_ragas.iloc[0].get("answer_correctness", 0.0)),
                "Faithfulness": float(df_ragas.iloc[0].get("faithfulness", 0.0)),
                "ContextPrecision": float(df_ragas.iloc[0].get("context_precision", 0.0)),
                "ContextRecall": float(df_ragas.iloc[0].get("context_recall", 0.0)),
            }
        except Exception as exc:
            if attempt < max_retries:
                wait_sec = 2.0
                print(f"      \u26a0\ufe0f Ragas evaluation timeout/error (retrying once) in {wait_sec:.1f}s: {exc}")
                time.sleep(wait_sec)
            else:
                print(f"      \u2139\ufe0f Ragas still unavailable after one retry; falling back to the semantic-rule scorer: {exc}")

    # Fallback rule
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
    Request exactly one metric that came back NaN; no other metric is touched.
    Returns the float on success, or None on failure/timeout, so the original value is
    """
    import math
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

    try:
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

        # Evaluate this single metric only; the call is small enough that it rarely times out
        results = evaluate(
            dataset=hf_dataset,
            metrics=[target_metric],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        df_res = results.to_pandas()
        
        # Extract the matching column (Ragas uses lowercase snake_case internally)
        col_name = target_metric.name
        val = float(df_res.iloc[0].get(col_name))
        return val if not math.isnan(val) else None
    except Exception as exc:
        print(f"         \u26a0\ufe0f Re-running metric [{metric_name}] raised: {exc}")
        return None
    
# ==============================================================================
# 7. System adapters: one interface over the four retrieval systems, exposing intermediate slices
# ==============================================================================
class BaseSystemAdapter:
    name: str
    def retrieve_and_generate(self, question: str) -> Dict[str, Any]:
        raise NotImplementedError


class GeneralRAGAdapter(BaseSystemAdapter):
    """Adapter for the Dense / Sparse / Hybrid paradigms."""
    def __init__(
        self,
        mode: str = "hybrid",
        dense_dir: str = DENSE_INDEX_DIR,
        sparse_dir: str = SPARSE_INDEX_DIR,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_rerank: int = TOP_K_RERANK,
        enable_context_enhancement: bool = ENABLE_CONTEXT_ENHANCEMENT,
    ):
        from rag_core.general_RAG import (
            DenseRetriever,
            SparseRetriever,
            HybridRetriever,
            build_rag_context,
            generate_answer,
            enhance_context,
        )

        self.name = mode.capitalize()
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.enable_context_enhancement = enable_context_enhancement

        if mode.lower() == "dense":
            self.base_retriever = DenseRetriever(index_dir=dense_dir)
        elif mode.lower() == "sparse":
            self.base_retriever = SparseRetriever(index_dir=sparse_dir)
        elif mode.lower() == "hybrid":
            self.base_retriever = HybridRetriever(dense_dir=dense_dir, sparse_dir=sparse_dir)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        self.reranker = APIRerankerWithRetry()
        self.build_rag_context = build_rag_context
        self.generate_answer = generate_answer
        self.context_enhance = enhance_context

    def retrieve_and_generate(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        candidates = []
        try:
            # 1. Candidate retrieval (slice 1: freeze the top-50 candidates immediately, whatever happens downstream)
            candidates = self.base_retriever.retrieve(question, top_k=self.top_k_retrieve)
            
            # 2. Strict reranking (slice 2A: exactly five chunks from the API, no expansion or merging)
            strict_top5 = self.reranker.rerank(question, candidates, top_k=self.top_k_rerank)
            if not strict_top5:
                raise RuntimeError("Rerank API call failed and returned no usable ranking")

            # 3. Context enhancement and merging for the generator (slice 2B: formula pull-in and neighbour merging)
            if self.enable_context_enhancement:
                all_recs = getattr(self.base_retriever, "records", None)
                if all_recs is None:
                    all_recs = getattr(self.base_retriever, "dense_records", None)
                
                final_docs = self.context_enhance(
                    retrieved_records=strict_top5,
                    all_records=all_recs,
                    expand_window=1,
                    fill_internal_gaps=True,
                )
            else:
                final_docs = strict_top5

            latency_ms = (time.perf_counter() - t0) * 1000.0

            is_empty = (len(final_docs) == 0)
            contexts = [doc.get("text", "") for doc in final_docs if doc.get("text")]

            # 4. Answer generation (one retry, with the max_tokens ceiling lifted)
            from openai import OpenAI
            gen_client = OpenAI(
                api_key=GEN_LLM_API_KEY,
                base_url=GEN_LLM_API_BASE,
                timeout=GEN_TIMEOUT,
            )

            generated_ans = ""
            finish_reason = "unknown"
            total_attempts = GEN_MAX_RETRIES + 1
            for attempt in range(total_attempts):
                try:
                    generated_ans, finish_reason = self.generate_answer(
                        question,
                        final_docs,
                        client=gen_client,
                        model=GEN_LLM_MODEL,
                        temperature=GEN_TEMPERATURE,
                        max_tokens=GEN_MAX_TOKENS,
                    )
                    break
                except Exception as exc:
                    if attempt < GEN_MAX_RETRIES:
                        wait_sec = 2.0
                        print(f"      \u26a0\ufe0f Generation timeout/error (retrying once) in {wait_sec:.1f}s: {exc}")
                        time.sleep(wait_sec)
                    else:
                        raise

            is_truncated = (finish_reason == "length")        

            return {
                "candidates_top50": candidates,  # Intermediate retriever output, reused by Experiment 2
                "strict_top5": strict_top5,      # Exactly five reranked chunks, used for retrieval-only scoring and Experiment 2 alignment
                "retrieved_docs": final_docs,    # Final chunks after reranking and context enhancement
                "contexts": contexts,            # Context actually sent to the model, reused by Experiment 3
                "generated_answer": generated_ans,
                "finish_reason": finish_reason,
                "is_truncated": is_truncated,
                "latency_ms": latency_ms,
                "is_success": True,
                "is_empty": is_empty,
                "error_msg": None,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "candidates_top50": candidates,  # The 50 candidates survive even if a later stage fails
                "retrieved_docs": [],
                "contexts": [],
                "generated_answer": f"ERROR: {str(e)}",
                "latency_ms": latency_ms,
                "is_success": False,
                "is_empty": True,
                "error_msg": str(e),
            }


class LightRAGMixAdapter(BaseSystemAdapter):
    """Adapter for LightRAG-mix (joint graph and vector retrieval)."""
    def __init__(self, top_k_retrieve: int = TOP_K_RETRIEVE, top_k_rerank: int = TOP_K_RERANK):
        from rag_core.graphRAG import create_rag
        from lightrag.lightrag import QueryParam

        self.name = "LightRAG-mix"
        self.rag = create_rag()
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # Disable LightRAG's internal reranking (enable_rerank=False) so it acts purely as a candidate recaller
        # This avoids a second rerank call inside LightRAG and an out-of-memory failure on the local GPU
        self.param = QueryParam(
            mode="mix",
            enable_rerank=False,
            top_k=self.top_k_retrieve,
            chunk_top_k=self.top_k_retrieve,
            response_type="Multiple Paragraphs",
        )

        # The same external rerank API then selects the top five for all four systems, in a single call
        self.reranker = APIRerankerWithRetry()

    async def _init_storage(self):
        await self.rag.initialize_storages()

    def init_sync(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self._init_storage())

    def retrieve_and_generate(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # 1. Graph-only recall: aquery_data returns 50 candidate chunks, without reranking
            structured_data = loop.run_until_complete(
                self.rag.aquery_data(question, param=self.param)
            )
            
            raw_chunks = structured_data.get("data", {}).get("chunks", []) if isinstance(structured_data, dict) else []
            candidates = [{"text": c.get("content", ""), "block_id": c.get("id", "")} for c in raw_chunks if isinstance(c, dict)]

            # 2. Shared reranking: the same external reranker picks the top five, in one call
            reranked_docs = self.reranker.rerank(question, candidates, top_k=self.top_k_rerank)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            is_empty = (len(reranked_docs) == 0)
            contexts = [doc.get("text", "") for doc in reranked_docs if doc.get("text")]

            # 3. Shared generation: the same client and model as Dense/Hybrid, with the max_tokens ceiling lifted
            from openai import OpenAI
            gen_client = OpenAI(
                api_key=GEN_LLM_API_KEY,
                base_url=GEN_LLM_API_BASE,
                timeout=GEN_TIMEOUT,
            )

            from rag_core.general_RAG import generate_answer
            generated_ans = ""
            finish_reason = "unknown"
            total_attempts = GEN_MAX_RETRIES + 1
            for attempt in range(total_attempts):
                try:
                    generated_ans, finish_reason = generate_answer(
                        question,
                        reranked_docs,
                        client=gen_client,
                        model=GEN_LLM_MODEL,
                        temperature=GEN_TEMPERATURE,
                        max_tokens=GEN_MAX_TOKENS,
                    )
                    break
                except Exception as exc:
                    if attempt < GEN_MAX_RETRIES:
                        wait_sec = 2.0
                        print(f"      \u26a0\ufe0f Generation timeout/error (retrying once) in {wait_sec:.1f}s: {exc}")
                        time.sleep(wait_sec)
                    else:
                        raise

            is_truncated = (finish_reason == "length")

            return {
                "candidates_top50": candidates,     # Graph-recall candidates, kept as an intermediate slice
                "strict_top5": reranked_docs,       # The five chunks returned by the reranker
                "retrieved_docs": reranked_docs,    # Top-5 evidence after reranking
                "contexts": contexts,               # Final context, kept as an intermediate slice
                "generated_answer": str(generated_ans),
                "finish_reason": finish_reason,  
                "is_truncated": is_truncated,
                "latency_ms": latency_ms,
                "is_success": True,
                "is_empty": is_empty,
                "error_msg": None,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "candidates_top50": [],
                "retrieved_docs": [],
                "contexts": [],
                "generated_answer": f"ERROR: {str(e)}",
                "latency_ms": latency_ms,
                "is_success": False,
                "is_empty": True,
                "error_msg": str(e),
            }

def validate_record_integrity(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Field checks first, punctuation second:
    1. Fields: is_success=True, answer non-empty without error, finish_reason != 'length', no NaN among the four metrics
    2. Punctuation: the answer ends with a valid terminal punctuation mark
    """
    import math

    # ==================== Step 1: field checks ====================
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

    # ==================== Step 2: punctuation check ====================
    normal_endings = ("。", "！", "？", ".", "!", "?", "”", "’", "）", ")", "]", "】", "%")
    if not any(ans.endswith(c) for c in normal_endings):
        return False, f"Answer does not end with valid punctuation: '{ans[-1]}'"

    return True, "Checks passed"

# ==============================================================================
# 8. Resume checkpoints and intermediate-slice cache (kept under evaluation/cache)
# ==============================================================================
def get_checkpoint_path(cache_dir: str = EVAL_CACHE_DIR) -> Path:
    return Path(cache_dir) / "eval1_checkpoint_records.jsonl"

# 845: def get_candidates_pool_path(cache_dir: str = EVAL_CACHE_DIR) -> Path:
#     """Intermediate slice 1: the top-50 candidate pool, reused by Experiment 2."""
#     return Path(cache_dir) / "hybrid_top50_candidates_pool.jsonl"

def get_final_contexts_path(cache_dir: str = EVAL_CACHE_DIR) -> Path:
    """Intermediate slice 2: the final top-5 context, reused by Experiment 3."""
    return Path(cache_dir) / "final_top5_contexts_pool.jsonl"

def repair_nan_checkpoint_records(cache_dir: str = EVAL_CACHE_DIR) -> int:
    """
    [Streaming single-metric repair]
    Find which of the four metrics is NaN and re-run exactly those, one by one:
    - only the NaN metrics are re-run; healthy metrics are never recomputed;
    - a successful re-run overwrites that field;
    - a failed re-run leaves the original record untouched, never writing 0.0;
    - the whole pass streams one line at a time and stays under 1 MB of memory.
    """
    import math
    ckpt_path = get_checkpoint_path(cache_dir)
    if not ckpt_path.exists():
        return 0

    temp_path = ckpt_path.with_suffix(".tmp")
    repaired_fields_count = 0
    target_metric_keys = ["AnswerCorrectness", "Faithfulness", "ContextPrecision", "ContextRecall"]

    with open(ckpt_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            has_valid_answer = bool(d.get("is_success")) and bool(d.get("generated_answer")) and not str(d.get("generated_answer")).startswith("ERROR:")
            
            if has_valid_answer:
                # Visit each metric in turn and re-run every NaN independently
                for metric_key in target_metric_keys:
                    val = d.get(metric_key)
                    is_nan = (val is None) or (isinstance(val, float) and math.isnan(val))
                    
                    if is_nan:
                        print(f"   \U0001f527 [{d.get('system')}_{d.get('qid')}] metric [{metric_key}] is NaN; re-running it on its own ...")
                        new_val = evaluate_single_specific_metric(d, metric_name=metric_key)
                        
                        if new_val is not None:
                            d[metric_key] = new_val
                            repaired_fields_count += 1
                            print(f"      \u2705 [{metric_key}] repaired: {new_val}")
                        else:
                            print(f"      \u26a0\ufe0f [{metric_key}] re-run failed; the original value is kept unchanged.")

            # Write the records back to a temporary file line by line
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
            f_out.flush()

    os.replace(temp_path, ckpt_path)
    if repaired_fields_count > 0:
        print(f"\u2705 In-place NaN repair finished: {repaired_fields_count} metric value(s) repaired.")
    return repaired_fields_count

def load_all_raw_checkpoint_records(cache_dir: str = EVAL_CACHE_DIR) -> Dict[str, Dict[str, Any]]:
    """Read every raw record, including failed rows, so the error type can be diagnosed."""
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
        print(f"\u26a0\ufe0f Failed to read the raw checkpoint: {e}")
    return records


def build_skip_cache(all_records: Dict[str, Dict[str, Any]], tainted_qids: set) -> Dict[str, Dict[str, Any]]:
    """
    [Building the skip set]
    Two-layer validation (fields, then punctuation) also rejects records that slipped through with corrupt data.
    Only records confirmed healthy end to end enter this skip set.
    """
    skip_dict = {}
    for key, d in all_records.items():
        qid = d.get("qid")
        if qid in tainted_qids:
            continue
        is_valid, _ = validate_record_integrity(d)
        if is_valid:
            skip_dict[key] = d
    print(f"\u2139\ufe0f Skip set built: {len(skip_dict)} fully healthy record(s) will be skipped.")
    return skip_dict


def decide_rerun_level_by_fields(
    raw_record: Optional[Dict[str, Any]],
    saved_candidates: Optional[List[Dict]],
    saved_contexts: Optional[List[str]],
    qid: str,
    tainted_qids: set,
) -> str:
    """
    [Classifying how each remaining item must be re-run]
    Every item outside the skip set is re-run. The stored field state decides where to restart:
    - 'LEVEL_2A_GENERATE_ONLY': evidence intact, only the answer was lost (empty/502/stream cut) -> reuse the top-5 evidence and redo generation plus grading;
    - 'LEVEL_2B_RERANK_ONLY': candidates retrieved but reranking failed (for example a 502 that slipped through) -> reuse the top-50 candidates and redo reranking plus generation;
    - 'LEVEL_3_FULL_PIPELINE': no candidates at all (new or badly damaged item) -> run the full retrieve-rerank-generate path from scratch.
    """
    if not raw_record:
        return "LEVEL_3_FULL_PIPELINE"

    err_str = str(raw_record.get("error_msg") or "")
    ans = str(raw_record.get("generated_answer") or "").strip()
    is_succ = raw_record.get("is_success")

    # 1. Did the reranking stage fail? (a 502 that slipped through, or an error_msg recording a rerank failure)
    is_rerank_damaged = (qid in tainted_qids) or ("Rerank" in err_str)
    if is_rerank_damaged:
        if saved_candidates and len(saved_candidates) >= TOP_K_RERANK:
            return "LEVEL_2B_RERANK_ONLY"
        return "LEVEL_3_FULL_PIPELINE"

    # 2. Did the generation stage fail? (empty answer, 502, or a stream cut mid-sentence with no punctuation)
    normal_endings = ("。", "！", "？", ".", "!", "?", "”", "’", "）", ")", "]", "】", "%")
    is_generation_damaged = (
        not is_succ
        or "empty answer" in err_str
        or "502" in err_str
        or not ans
        or ans.startswith("ERROR:")
        or not any(ans.endswith(c) for c in normal_endings)
    )

    if is_generation_damaged:
        if saved_contexts and len(saved_contexts) == TOP_K_RERANK:
            return "LEVEL_2A_GENERATE_ONLY"
        if saved_candidates and len(saved_candidates) >= TOP_K_RERANK:
            return "LEVEL_2B_RERANK_ONLY"
        return "LEVEL_3_FULL_PIPELINE"

    return "LEVEL_3_FULL_PIPELINE"

def get_candidates_pool_dir(cache_dir: str = EVAL_CACHE_DIR) -> Path:
    p = Path(cache_dir) / "candidates_pool"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_candidates_pool_path(cache_dir: str = EVAL_CACHE_DIR, sys_name: str = "hybrid") -> Path:
    """Path of the top-50 candidate pool for one system."""
    safe_name = sys_name.lower().replace("-", "_")
    return get_candidates_pool_dir(cache_dir) / f"candidates_{safe_name}.jsonl"

def get_contexts_pool_dir(cache_dir: str = EVAL_CACHE_DIR) -> Path:
    p = Path(cache_dir) / "contexts_pool"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_final_contexts_path(cache_dir: str = EVAL_CACHE_DIR, sys_name: str = "hybrid") -> Path:
    """Path of the final top-5 context for one system (after enhancement)."""
    safe_name = sys_name.lower().replace("-", "_")
    return get_contexts_pool_dir(cache_dir) / f"contexts_{safe_name}.jsonl"

def get_strict_contexts_path(cache_dir: str = EVAL_CACHE_DIR, sys_name: str = "hybrid") -> Path:
    """Path of the strict top-5 reranked chunks for one system (exactly five, unmerged, used by Experiment 2 and for baseline alignment)."""
    safe_name = sys_name.lower().replace("-", "_")
    return get_contexts_pool_dir(cache_dir) / f"contexts_strict_{safe_name}.jsonl"

def load_saved_strict_contexts_pool(cache_dir: str = EVAL_CACHE_DIR, systems: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """Read every system's strict top-5 evidence into a dict keyed by f'{system}_{qid}'."""
    records = {}
    target_systems = systems or DEFAULT_SYSTEMS
    for s_name in target_systems:
        ctx_path = get_strict_contexts_path(cache_dir, s_name)
        if not ctx_path.exists():
            continue
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    key = f"{d.get('system')}_{d.get('qid')}"
                    if d.get("strict_top5"):
                        records[key] = d["strict_top5"]
        except Exception as e:
            print(f"\u26a0\ufe0f Failed to read the strict context pool ({ctx_path.name}): {e}")
    return records

def load_saved_candidates_pool(cache_dir: str = EVAL_CACHE_DIR, systems: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """Read every system's candidate pool into a dict keyed by f'{system}_{qid}'."""
    records = {}
    target_systems = systems or DEFAULT_SYSTEMS
    for s_name in target_systems:
        pool_path = get_candidates_pool_path(cache_dir, s_name)
        if not pool_path.exists():
            continue
        try:
            with open(pool_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    key = f"{d.get('system')}_{d.get('qid')}"
                    if d.get("candidates"):
                        records[key] = d["candidates"]
        except Exception as e:
            print(f"\u26a0\ufe0f Failed to read the candidate pool ({pool_path.name}): {e}")
    return records

def load_saved_contexts_pool(cache_dir: str = EVAL_CACHE_DIR, systems: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Read every system's context pool into a dict keyed by f'{system}_{qid}'."""
    records = {}
    target_systems = systems or DEFAULT_SYSTEMS
    for s_name in target_systems:
        ctx_path = get_final_contexts_path(cache_dir, s_name)
        if not ctx_path.exists():
            continue
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    key = f"{d.get('system')}_{d.get('qid')}"
                    if d.get("contexts"):
                        records[key] = d["contexts"]
        except Exception as e:
            print(f"\u26a0\ufe0f Failed to read the context pool ({ctx_path.name}): {e}")
    return records

def append_checkpoint_record(cache_dir: str, record: Dict[str, Any]) -> None:
    """
    [Atomic in-place upsert]
    - if a record for the same (system, qid) already exists (a historical 502 or error row), it is overwritten in place;
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
            
            # Same item found: replace the stale row with the new successful record
            if curr_key == target_key:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                replaced = True
            else:
                f_out.write(line_str + "\n")

        # A brand-new item (never recorded) is appended at the end
        if not replaced:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_out.flush()

    # Atomic replace so the file is never left half-written
    os.replace(temp_path, ckpt_path)

def save_intermediate_artifacts(
    cache_dir: str,
    sys_name: str,
    qid: str,
    candidates: List[Dict],
    strict_top5: List[Dict],
    contexts: List[str]
) -> None:
    """
    Persist the intermediate slices, one file per system under candidates_pool/ and contexts_pool/:
    - slice 1: candidates_pool/candidates_{system}.jsonl, written as soon as recall finishes so reranking can resume without re-searching;
    - slice 2A: contexts_pool/contexts_strict_{system}.jsonl, the five strict chunks without merging, reused verbatim by Experiment 2;
    - slice 2B: contexts_pool/contexts_{system}.jsonl, the enhanced and merged context that Experiment 3 answers from.
    """
    # 1. Candidate pool: one jsonl per system
    if candidates and len(candidates) > 0:
        pool_path = get_candidates_pool_path(cache_dir, sys_name)
        item = {
            "system": sys_name,
            "qid": qid,
            "candidates": candidates,
        }
        with open(pool_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()

    # 2A. Strict top-5 chunks: one jsonl per system
    if strict_top5 and len(strict_top5) > 0:
        strict_path = get_strict_contexts_path(cache_dir, sys_name)
        strict_item = {
            "system": sys_name,
            "qid": qid,
            "strict_top5": strict_top5,
        }
        with open(strict_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(strict_item, ensure_ascii=False) + "\n")
            f.flush()

    # 2B. Final enhanced context: one jsonl per system
    if contexts and len(contexts) > 0:
        ctx_path = get_final_contexts_path(cache_dir, sys_name)
        ctx_item = {
            "system": sys_name,
            "qid": qid,
            "contexts": contexts,
        }
        with open(ctx_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ctx_item, ensure_ascii=False) + "\n")
            f.flush()


# ==============================================================================
# 9. Main evaluation flow and report tables
# ==============================================================================
def run_evaluation_experiment_1(
    dataset: List[Dict[str, Any]],
    systems_to_evaluate: List[str] = None,
    output_dir: str = EVAL_OUTPUT_DIR,
    cache_dir: str = EVAL_CACHE_DIR,
    sample_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Experiment 1 and score every metric (reranker API, one retry, intermediate slices, resumable checkpoints).
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if sample_limit is not None and sample_limit > 0:
        dataset = dataset[:sample_limit]
        print(f"\u2139\ufe0f Sample limit for this run: {sample_limit}")

    if systems_to_evaluate is None:
        systems_to_evaluate = DEFAULT_SYSTEMS

    # 1. Stream the checkpoint first and repair any NaN metric that already has an answer
    repair_nan_checkpoint_records(cache_dir)

    # 2. Read every raw record for the field-level diagnosis
    all_raw_checkpoint_records = load_all_raw_checkpoint_records(cache_dir)

    # 3. Build the skip set: only records passing the field and punctuation checks, with no error, may be skipped
    TAINTED_RERANK_QIDS = {}
    skip_cache = build_skip_cache(all_raw_checkpoint_records, TAINTED_RERANK_QIDS)

    # 4. Preload the persisted intermediate slices (50-candidate pool and 5-chunk context pool)
    saved_candidates_pool = load_saved_candidates_pool(cache_dir)
    saved_contexts_pool = load_saved_contexts_pool(cache_dir)

    print(f"================================================================================")
    print(f"\U0001f680 Running Experiment 1: comparing retrieval paradigms")
    print(f"   Evaluation items: {len(dataset)}")
    print(f"   Systems compared: {systems_to_evaluate}")
    print(f"   Top-K: candidate={TOP_K_RETRIEVE}, context={TOP_K_RERANK}")
    print(f"   Retry policy: one retry on timeout or error")
    print(f"   Checkpoint and slice cache: {cache_dir}")
    print(f"   Report output: {output_dir}")
    print(f"================================================================================")

    all_details = []

    for sys_name in systems_to_evaluate:
        print(f"\n---> [evaluating system]: {sys_name}")
        adapter = None
        try:
            if sys_name.lower() == "dense":
                adapter = GeneralRAGAdapter(mode="dense")
            elif sys_name.lower() == "sparse":
                adapter = GeneralRAGAdapter(mode="sparse")
            elif sys_name.lower() == "hybrid":
                adapter = GeneralRAGAdapter(mode="hybrid")
            elif sys_name.lower() in ("lightrag", "lightrag-mix"):
                adapter = LightRAGMixAdapter()
                adapter.init_sync()
        except Exception as e:
            print(f"\u26a0\ufe0f Initialising system {sys_name} reported: {e}; recording the failure and continuing ...")

        for i, qa in enumerate(dataset):
            qid = qa["qid"]
            task_type = qa["task_type"]
            topic = qa["topic"]
            question = qa["question"]
            ref_ans = qa["reference_answer"]
            gold_para = qa["gold_paragraph"]

            ckpt_key = f"{sys_name}_{qid}"

            # ==================== Step 1: consult the skip set ====================
            # Only healthy items in the skip set may be skipped instantly
            if ckpt_key in skip_cache:
                saved_record = skip_cache[ckpt_key]
                all_details.append(saved_record)
                if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                    print(f"   [{sys_name}] {i + 1}/{len(dataset)} done (restored from checkpoint)")
                continue

            # ==================== Step 2: everything outside the skip set is re-run ====================
            # The stored field state decides whether to restart from scratch or midway
            raw_history_record = all_raw_checkpoint_records.get(ckpt_key)
            cached_candidates = saved_candidates_pool.get(ckpt_key)
            cached_contexts = saved_contexts_pool.get(ckpt_key)

            rerun_level = decide_rerun_level_by_fields(
                raw_record=raw_history_record,
                saved_candidates=cached_candidates,
                saved_contexts=cached_contexts,
                qid=qid,
                tainted_qids=TAINTED_RERANK_QIDS,
            )

            res = None

            # [Level A] evidence intact, only the answer was lost -> reuse the top-5 evidence and redo generation plus grading
            if rerun_level == "LEVEL_2A_GENERATE_ONLY":
                print(f"   \u26a1 [{ckpt_key}] diagnosed as a generation-stage failure: reusing the stored top-5 evidence and redoing generation plus grading ...")
                t0 = time.perf_counter()
                existing_contexts = cached_contexts
                final_docs = [{"text": c} for c in existing_contexts]
                
                from openai import OpenAI
                gen_client = OpenAI(
                    api_key=GEN_LLM_API_KEY,
                    base_url=GEN_LLM_API_BASE,
                    timeout=GEN_TIMEOUT,
                )
                from rag_core.general_RAG import generate_answer
                
                generated_ans = ""
                finish_reason = "unknown"
                for attempt in range(GEN_MAX_RETRIES + 1):
                    try:
                        generated_ans, finish_reason = generate_answer(
                            question,
                            final_docs,
                            client=gen_client,
                            model=GEN_LLM_MODEL,
                            temperature=GEN_TEMPERATURE,
                            max_tokens=GEN_MAX_TOKENS,
                        )
                        break
                    except Exception as exc:
                        if attempt < GEN_MAX_RETRIES:
                            time.sleep(2.0)
                        else:
                            generated_ans = f"ERROR: {str(exc)}"

                latency_ms = (time.perf_counter() - t0) * 1000.0
                is_succ = not generated_ans.startswith("ERROR:")
                res = {
                    "candidates_top50": [],
                    "retrieved_docs": final_docs,
                    "contexts": existing_contexts,
                    "generated_answer": generated_ans,
                    "finish_reason": finish_reason,
                    "is_truncated": (finish_reason == "length"),
                    "latency_ms": latency_ms,
                    "is_success": is_succ,
                    "is_empty": (len(final_docs) == 0),
                    "error_msg": None if is_succ else generated_ans,
                }

            # [Level B] candidates retrieved but reranking failed -> reuse the top-50 candidates and redo reranking plus generation
            elif rerun_level == "LEVEL_2B_RERANK_ONLY":
                print(f"   \u26a1 [{ckpt_key}] diagnosed as a reranking-stage failure: reusing the stored top-50 candidates and redoing the reranker call plus generation ...")
                t0 = time.perf_counter()
                candidates = cached_candidates
                
                reranked = adapter.reranker.rerank(question, candidates, top_k=TOP_K_RERANK)
                if not reranked:
                    res = {
                        "candidates_top50": candidates,
                        "strict_top5": [],
                        "retrieved_docs": [],
                        "contexts": [],
                        "generated_answer": "ERROR: rerank API failed",
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                        "is_success": False,
                        "is_empty": True,
                        "error_msg": "rerank API failed",
                    }
                else:
                    strict_top5 = reranked

                    # Only Dense/Sparse/Hybrid expand and merge the context; LightRAG-mix keeps its evidence untouched
                    if sys_name.lower() not in ("lightrag", "lightrag-mix") and getattr(adapter, "enable_context_enhancement", False):
                        all_recs = getattr(adapter.base_retriever, "records", None) or getattr(adapter.base_retriever, "dense_records", None)
                        final_docs = adapter.context_enhance(
                            retrieved_records=strict_top5,
                            all_records=all_recs,
                            expand_window=1,
                            fill_internal_gaps=True,
                        )
                    else:
                        final_docs = strict_top5

                    contexts = [doc.get("text", "") for doc in final_docs if doc.get("text")]
                    from openai import OpenAI
                    gen_client = OpenAI(
                        api_key=GEN_LLM_API_KEY,
                        base_url=GEN_LLM_API_BASE,
                        timeout=GEN_TIMEOUT,
                    )
                    from rag_core.general_RAG import generate_answer
                    generated_ans = ""
                    finish_reason = "unknown"
                    for attempt in range(GEN_MAX_RETRIES + 1):
                        try:
                            generated_ans, finish_reason = generate_answer(
                                question,
                                final_docs,
                                client=gen_client,
                                model=GEN_LLM_MODEL,
                                temperature=GEN_TEMPERATURE,
                                max_tokens=GEN_MAX_TOKENS,
                            )
                            break
                        except Exception as exc:
                            if attempt < GEN_MAX_RETRIES:
                                time.sleep(2.0)
                            else:
                                generated_ans = f"ERROR: {str(exc)}"
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    is_succ = not generated_ans.startswith("ERROR:")
                    res = {
                        "candidates_top50": candidates,
                        "strict_top5": strict_top5,
                        "retrieved_docs": final_docs,
                        "contexts": contexts,
                        "generated_answer": generated_ans,
                        "finish_reason": finish_reason,
                        "is_truncated": (finish_reason == "length"),
                        "latency_ms": latency_ms,
                        "is_success": is_succ,
                        "is_empty": False,
                        "error_msg": None if is_succ else generated_ans,
                    }

            # [Level C] no candidates (new or badly damaged item) -> run the full online retrieve-rerank-generate path
            else:
                print(f"   \U0001f504 [{ckpt_key}] diagnosed as a fresh or fully missing item: running the whole retrieve-rerank flow ...")
                if adapter is not None:
                    res = adapter.retrieve_and_generate(question)
                else:
                    res = {
                        "candidates_top50": [],
                        "retrieved_docs": [],
                        "contexts": [],
                        "generated_answer": "ERROR: Adapter not initialized",
                        "latency_ms": 0.0,
                        "is_success": False,
                        "is_empty": True,
                        "error_msg": "Adapter not initialized",
                    }

            # Persist the intermediate slices; the candidate pool, the strict top-5 and the final
            save_intermediate_artifacts(
                cache_dir=cache_dir,
                sys_name=sys_name,
                qid=qid,
                candidates=res.get("candidates_top50", []),
                strict_top5=res.get("strict_top5", []),
                contexts=res.get("contexts", [])
            )

            # 3. Retrieval metrics (Recall@5, MRR@5, nDCG@5)
            ret_metrics = compute_retrieval_metrics(gold_para, res["retrieved_docs"], k=TOP_K_RERANK)

            sample_item = {
                "qid": qid,
                "task_type": task_type,
                "topic": topic,
                "system": sys_name,
                "question": question,
                "reference_answer": ref_ans,
                "gold_paragraph": gold_para,
                "generated_answer": res["generated_answer"],
                "contexts": res["contexts"],
                "latency_ms": res["latency_ms"],
                "is_success": res["is_success"],
                "is_empty": res["is_empty"],
                "error_msg": res.get("error_msg", None),
                **ret_metrics,
            }

            # 4. Per-question RAGAS metrics (with one retry)
            ragas_metric = evaluate_single_ragas_sample(sample_item)
            sample_item.update(ragas_metric)

            # 5. Persist to the checkpoint under evaluation/cache immediately
            append_checkpoint_record(cache_dir, sample_item)
            skip_cache[ckpt_key] = sample_item
            all_details.append(sample_item)

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                print(f"   [{sys_name}] {i + 1}/{len(dataset)} items evaluated and checkpointed")

    df_details = pd.DataFrame(all_details)

    # Export the per-item metric table (all metrics, question type and topic)
    detail_csv_path = os.path.join(output_dir, "eval1_all_qa_metrics.csv")
    df_details.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n\u2705 Per-item metric table saved to: {detail_csv_path}")

    # Aggregate into the Experiment 1 summary table
    summary_rows = []
    for sys_name in systems_to_evaluate:
        sub = df_details[df_details["system"] == sys_name]
        if len(sub) == 0:
            continue

        p50_latency = sub["latency_ms"].quantile(0.50)
        p95_latency = sub["latency_ms"].quantile(0.95)
        success_rate = (sub["is_success"].sum() / len(sub)) * 100.0
        empty_rate = (sub["is_empty"].sum() / len(sub)) * 100.0

        summary_rows.append({
            "Retrieval method": sys_name,
            "nDCG@5": round(sub["nDCG@5"].mean(), 4),
            "Recall@5": round(sub["Recall@5"].mean(), 4),
            "MRR@5": round(sub["MRR@5"].mean(), 4),
            "AnswerCorrectness": round(sub["AnswerCorrectness"].mean(), 4),
            "Faithfulness": round(sub["Faithfulness"].mean(), 4),
            "ContextPrecision": round(sub["ContextPrecision"].mean(), 4),
            "ContextRecall": round(sub["ContextRecall"].mean(), 4),
            "Online Latency P50 (ms)": round(p50_latency, 2),
            "Online Latency P95 (ms)": round(p95_latency, 2),
            "Success Rate (%)": round(success_rate, 2),
            "Empty Rate (%)": round(empty_rate, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "eval1_summary_table.csv")
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\u2705 Experiment 1 summary table saved to: {summary_csv_path}")

    # Stratified analysis by question type and by domain topic
    stratified_rows = []
    for sys_name in systems_to_evaluate:
        for t_type in df_details["task_type"].unique():
            sub = df_details[(df_details["system"] == sys_name) & (df_details["task_type"] == t_type)]
            if len(sub) == 0:
                continue
            stratified_rows.append({
                "Retrieval method": sys_name,
                "Stratification": "Task Type",
                "Category": t_type,
                "N": len(sub),
                "nDCG@5": round(sub["nDCG@5"].mean(), 4),
                "Recall@5": round(sub["Recall@5"].mean(), 4),
                "MRR@5": round(sub["MRR@5"].mean(), 4),
                "AnswerCorrectness": round(sub["AnswerCorrectness"].mean(), 4),
                "Faithfulness": round(sub["Faithfulness"].mean(), 4),
            })

    for sys_name in systems_to_evaluate:
        for topic in df_details["topic"].unique():
            sub = df_details[(df_details["system"] == sys_name) & (df_details["topic"] == topic)]
            if len(sub) == 0:
                continue
            stratified_rows.append({
                "Retrieval method": sys_name,
                "Stratification": "Domain Topic",
                "Category": topic,
                "N": len(sub),
                "nDCG@5": round(sub["nDCG@5"].mean(), 4),
                "Recall@5": round(sub["Recall@5"].mean(), 4),
                "MRR@5": round(sub["MRR@5"].mean(), 4),
                "AnswerCorrectness": round(sub["AnswerCorrectness"].mean(), 4),
                "Faithfulness": round(sub["Faithfulness"].mean(), 4),
            })

    df_stratified = pd.DataFrame(stratified_rows)
    stratified_csv_path = os.path.join(output_dir, "eval1_stratified_table.csv")
    df_stratified.to_csv(stratified_csv_path, index=False, encoding="utf-8-sig")
    print(f"\u2705 Stratified analysis table saved to: {stratified_csv_path}")

    print("\n" + "="*95)
    print("\U0001f4ca Experiment 1: end-to-end and retrieval metrics by paradigm")
    print("="*95)
    print(df_summary.to_string(index=False))
    print("="*95 + "\n")

    return df_details, df_summary, df_stratified


# ==============================================================================
# 10. Command line entry point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 1: retrieval-paradigm comparison (eval1.py)")
    parser.add_argument("--selected-dir", type=str, default=SELECTED_DIR, help="Root directory of the \"QA dataset\" folder")
    parser.add_argument("--output-dir", type=str, default=EVAL_OUTPUT_DIR, help="Report output directory")
    parser.add_argument("--cache-dir", type=str, default=EVAL_CACHE_DIR, help="Checkpoint and intermediate-slice cache directory")
    parser.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, help="Retrieval systems to evaluate")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of evaluated items (smoke testing, e.g. 5 or 10)")
    args = parser.parse_args()

    dataset = load_selected_dataset(args.selected_dir)
    print(f"\u2705 Loaded {len(dataset)} evaluation items from {args.selected_dir}")

    run_evaluation_experiment_1(
        dataset=dataset,
        systems_to_evaluate=args.systems,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
