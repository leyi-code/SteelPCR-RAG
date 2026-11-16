# Import torch first so the Windows C runtime initialises the PyTorch CUDA context before spaCy; this avoids the WinError 1114 crash seen when spaCy is imported first.
import torch

import json
import math
import os
import re
import pickle
import time
import numpy as np
from pathlib import Path
import faiss
import jieba
import spacy
from typing import Any, Dict, List, Optional, Protocol, Sequence,Tuple,Union
from sentence_transformers import CrossEncoder
import sys

from openai import OpenAI

# Allow this module to be used both as a package import
# (``from rag_core.general_RAG import ...``) and as a standalone script
# (``python rag_core/general_RAG.py``).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from steelpcr_config import env  # noqa: E402

# ==================== Configuration ====================
EMBEDDING_API_BASE = env("STEELPCR_EMBED_API_BASE", "your api url")
EMBEDDING_API_KEY = env("STEELPCR_EMBED_API_KEY", "your api key")
EMBEDDING_MODEL_API = "Qwen3-Embedding-8B"
# Embedding dimensionality; the API response must match this value.
EMBEDDING_DIMENSION = 4096
EMBEDDING_API_BATCH_SIZE = 100
EMBEDDING_API_MAX_RETRIES = 3
EMBEDDING_API_DELAY = 0.05
# Timeout for a single embedding API request, in seconds.
EMBEDDING_API_TIMEOUT = 120.0

DENSE_DIR = "./cache/dense"
SPARSE_DIR = "./cache/sparse"

TOP_K_RETRIEVE = 50
TOP_K_RERANK = 5
DEVICE = "cuda"

HNSW_EF_SEARCH = 128

# Reranking. A retriever only performs the initial search; combining it with a reranker is done through the retrieval wrapper.
RERANKER_MODEL = "./model/bge-reranker-v2-m3"
RERANKER_BATCH_SIZE = 16
# The API reranker uses an explicit Cohere-compatible /rerank URL; the path and provider protocol are never guessed.
RERANK_API_URL = env("STEELPCR_RERANK_API_URL", "")  # e.g. "https://your-endpoint/v1/rerank"
RERANK_API_KEY = env("STEELPCR_RERANK_API_KEY", "your api key")  # supply through the environment; never commit a real key
RERANK_API_MODEL = env("STEELPCR_RERANK_API_MODEL", "")
RERANK_API_TIMEOUT = 30.0
RERANK_API_MAX_RETRIES = 3

# Generation model (OpenAI-compatible Chat Completions API).
# Endpoint, model version and credential are centralised here so the experiment stays reproducible.
GENERATION_API_BASE = env("STEELPCR_GEN_API_BASE", "https://api.deepseek.com")
GENERATION_API_KEY = env("STEELPCR_GEN_API_KEY", "your api key")
GENERATION_MODEL = env("STEELPCR_GEN_MODEL", "deepSeek-flash")
GENERATION_API_TIMEOUT = 120.0
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 32000

# Query rewriting and bilingual terminology expansion (OpenAI-compatible API)
EXPANSION_API_BASE = env("STEELPCR_EXPANSION_API_BASE", GENERATION_API_BASE)
EXPANSION_API_KEY = env("STEELPCR_EXPANSION_API_KEY", GENERATION_API_KEY)
EXPANSION_MODEL = env("STEELPCR_EXPANSION_MODEL", "deepSeek-flash")
EXPANSION_API_TIMEOUT = 30.0

# Bound the context sent to the generator so it stays inside the model window.
GENERATION_CONTEXT_MAX_CHARS_PER_DOC = 3000
GENERATION_CONTEXT_MAX_CHARS_TOTAL = 12000

# ==============================================================

embedding_client = OpenAI(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_API_BASE,
    timeout=EMBEDDING_API_TIMEOUT,
)


def get_generation_client() -> OpenAI:
    """Create the generation client lazily so building an index does not require a generation service."""
    if not GENERATION_API_KEY or GENERATION_API_KEY == "your api key":
        raise ValueError("STEELPCR_GEN_API_KEY is not set; copy .env.example to .env and fill it in.")
    return OpenAI(
        api_key=GENERATION_API_KEY,
        base_url=GENERATION_API_BASE,
        timeout=GENERATION_API_TIMEOUT,
    )


print("[init] Loading the spaCy English model ...")
_nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def tokenize_for_bm25(text: str) -> List[str]:
    if not text:
        return []
    if re.search(r'[\u4e00-\u9fff]', text):
        tokens = list(jieba.cut(text))
        return [t.strip() for t in tokens if t.strip() and not re.match(r'^[\s\W]+$', t.strip())]
    doc = _nlp_en(text.lower())
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop:
            continue
        lemma = token.lemma_.strip()
        if lemma and len(lemma) > 1:
            tokens.append(lemma)
        elif token.norm_.strip() and len(token.norm_.strip()) > 1:
            tokens.append(token.norm_.strip())
    return tokens


def embed_via_api(texts: List[str], batch_size: int = EMBEDDING_API_BATCH_SIZE) -> np.ndarray:
    """Encode texts in batches through the OpenAI-compatible embedding API with a per-request timeout."""
    if not texts:
        return np.empty((0, 0), dtype="float32")

    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(EMBEDDING_API_MAX_RETRIES):
            try:
                response = embedding_client.embeddings.create(
                    model=EMBEDDING_MODEL_API,
                    input=batch,
                    encoding_format="float",
                    timeout=EMBEDDING_API_TIMEOUT,
                    dimensions=EMBEDDING_DIMENSION
                )
                batch_embeddings = [item.embedding for item in response.data]
                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        f"Unexpected number of vectors returned: expected {len(batch)}, got {len(batch_embeddings)}"
                    )
                invalid_dimensions = [
                    len(embedding)
                    for embedding in batch_embeddings
                    if len(embedding) != EMBEDDING_DIMENSION
                ]
                if invalid_dimensions:
                    raise RuntimeError(
                        f"Unexpected vector dimensionality: expected {EMBEDDING_DIMENSION}, "
                        f"got {invalid_dimensions[0]}"
                    )
                all_embeddings.extend(batch_embeddings)
                end_i = min(i + batch_size, total)
                print(f"  [embed API] {end_i}/{total} done")
                if end_i < total:
                    time.sleep(EMBEDDING_API_DELAY)
                break
            except Exception as exc:
                print(
                    f"  [API Error] batch {i}-{i + len(batch)} "
                    f"retry {attempt + 1} (timeout={EMBEDDING_API_TIMEOUT}s): {exc}"
                )
                if attempt == EMBEDDING_API_MAX_RETRIES - 1:
                    raise RuntimeError(f"Embedding API call failed: {exc}") from exc
                time.sleep(2 ** attempt)

    embeddings = np.asarray(all_embeddings, dtype="float32")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != total
        or embeddings.shape[1] != EMBEDDING_DIMENSION
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}, "
            f"expected ({total}, {EMBEDDING_DIMENSION})"
        )
    faiss.normalize_L2(embeddings)
    return embeddings


def _load_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ==================== Stage 3: composable retrieval and reranking ====================
class Reranker(Protocol):
    """Pluggable reranker: returns copies of the records annotated with ``_rerank_score``."""
    def rerank(self, query: str, candidates: Sequence[Dict], top_k: Optional[int] = None) -> List[Dict]: ...


class LocalCrossEncoderReranker:
    """Local cross-encoder reranker; the model is loaded lazily on the first rerank call."""
    def __init__(self, model: str = RERANKER_MODEL, device: str = DEVICE,
                 batch_size: int = RERANKER_BATCH_SIZE):
        self.model_name, self.device, self.batch_size = model, device, batch_size
        self._model: Optional[CrossEncoder] = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            print(f"[local-reranker] loading model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: Optional[int] = None) -> List[Dict]:
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must not be negative")
        if not candidates:
            return []
        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = self._get_model().predict(pairs, batch_size=self.batch_size)
        result = [dict(c, _rerank_score=float(s)) for c, s in zip(candidates, scores)]
        result.sort(key=lambda c: c["_rerank_score"], reverse=True)
        return result[:top_k] if top_k is not None else result


class APIReranker:
    """Call an explicitly configured Cohere-compatible /rerank API and map the returned indices back to candidates."""
    def __init__(self, url: str = RERANK_API_URL, api_key: str = RERANK_API_KEY,
                 model: str = RERANK_API_MODEL, timeout: float = RERANK_API_TIMEOUT,
                 max_retries: int = RERANK_API_MAX_RETRIES):
        if not url or not model:
            raise ValueError("APIReranker requires an explicit url and model")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.url, self.api_key, self.model = url, api_key, model
        self.timeout, self.max_retries = timeout, max_retries

    def rerank(self, query: str, candidates: Sequence[Dict], top_k: Optional[int] = None) -> List[Dict]:
        import requests

        if top_k is not None and top_k < 0:
            raise ValueError("top_k must not be negative")
        if not candidates:
            return []
        payload = {"model": self.model, "query": query,
                   "documents": [c.get("text", "") for c in candidates],
                   "top_n": top_k if top_k is not None else len(candidates)}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = None
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    break
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt + 1 < self.max_retries:
                        time.sleep(2 ** attempt)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
        if data is None:
            raise RuntimeError(f"Rerank API call failed after {self.max_retries} attempts: {last_error}") from last_error
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RuntimeError("Rerank API response must contain a results array")

        ranked, seen = [], set()
        for item in data["results"]:
            if not isinstance(item, dict):
                raise RuntimeError("Rerank API results contain a non-object entry")
            idx, score = item.get("index"), item.get("relevance_score")
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(candidates):
                raise RuntimeError(f"Rerank API returned an invalid index: {idx}")
            if idx in seen:
                raise RuntimeError(f"Rerank API returned a duplicate index: {idx}")
            if isinstance(score, bool):
                raise RuntimeError(f"Rerank API returned an invalid score: {score}")
            try:
                score = float(score)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Rerank API returned an invalid score: {score}") from exc
            if not math.isfinite(score):
                raise RuntimeError(f"Rerank API returned a non-finite score: {score}")
            seen.add(idx)
            ranked.append(dict(candidates[idx], _rerank_score=score))
        ranked.sort(key=lambda c: c["_rerank_score"], reverse=True)
        return ranked[:top_k] if top_k is not None else ranked


class DenseRetriever:
    def __init__(self, index_dir: str = DENSE_DIR):
        self.index = faiss.read_index(os.path.join(index_dir, "faiss_hnsw.index"))
        self.records = _load_records(os.path.join(index_dir, "records.jsonl"))
        self.index.hnsw.efSearch = HNSW_EF_SEARCH

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
        scores, indices = self.index.search(embed_via_api([query]), top_k)
        return [
            dict(self.records[i], _dense_score=float(score))
            for score, i in zip(scores[0], indices[0])
            if 0 <= i < len(self.records)
        ]

    def search(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict]:
        return self.retrieve(query, top_k)


class SparseRetriever:
    def __init__(self, index_dir: str = SPARSE_DIR):
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            data = pickle.load(f)
        self.bm25, self.records = data["bm25"], data["records"]

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
        scores = self.bm25.get_scores(tokenize_for_bm25(query))
        return [dict(self.records[i], _bm25_score=float(scores[i]))
                for i in np.argsort(scores)[::-1][:top_k]]

    def search(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict]:
        return self.retrieve(query, top_k)


class HybridRetriever:
    def __init__(self, dense_dir: str = DENSE_DIR, sparse_dir: str = SPARSE_DIR):
        self.dense_index = faiss.read_index(os.path.join(dense_dir, "faiss_hnsw.index"))
        self.dense_records = _load_records(os.path.join(dense_dir, "records.jsonl"))
        self.dense_index.hnsw.efSearch = HNSW_EF_SEARCH
        with open(os.path.join(sparse_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
        _, dense_indices = self.dense_index.search(embed_via_api([query]), top_k)
        sparse_scores = self.bm25.get_scores(tokenize_for_bm25(query))
        rrf_scores = {}
        for rank, idx in enumerate(dense_indices[0]):
            if 0 <= idx < len(self.dense_records):
                rrf_scores[int(idx)] = 1 / (60 + rank + 1)
        for rank, idx in enumerate(np.argsort(sparse_scores)[::-1][:top_k]):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (60 + rank + 1)
        return [
            dict(self.dense_records[i], _rrf_score=float(rrf_scores[i]))
            for i in sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        ]

    def search(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict]:
        return self.retrieve(query, top_k)


class RerankedRetriever:
    """Combine any retriever with any reranker to run retrieve-then-rerank."""
    def __init__(self, retriever: Any, reranker: Reranker):
        self.retriever, self.reranker = retriever, reranker

    def search(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict]:
        candidates = self.retriever.retrieve(query, TOP_K_RETRIEVE)
        return self.reranker.rerank(query, candidates, top_k)


# ==================== Stage 3b: context enhancement and neighbour merging ====================
def _ensure_doc_index(records_pool: Sequence[Dict]) -> Dict[str, List[Dict]]:
    """Group every knowledge-base record by source_file into ordered lists so neighbours can be located by index."""
    doc_map: Dict[str, List[Dict]] = {}
    for r in records_pool:
        doc_key = r.get("source_file") or r.get("doc_title") or r.get("doc_uuid") or "__default__"
        if doc_key not in doc_map:
            doc_map[doc_key] = []
        doc_map[doc_key].append(r)
    return doc_map


def enhance_context(
    retrieved_records: Sequence[Dict],
    all_records: Optional[Sequence[Dict]] = None,
    expand_window: int = 1,
    merge_by_paragraph: bool = True,
    fill_internal_gaps: bool = True,
    max_tokens_or_chars: Optional[int] = None,
) -> List[Dict]:
    """
    Enhance and merge the context around the reranked candidate chunks.

    Solves two problems:
    1. Formula-only blocks carry no natural-language semantics and are almost
       never retrieved directly; once a prose block describing them is hit, this
       module probes its neighbours and pulls the adjacent formula blocks in.
    2. Scattered hits that share a head_paragraph / head_section inside one
       document are stitched back together in original order into a single
       coherent evidence passage.

    Args:
        retrieved_records: the top-K chunks selected by the reranker.
        all_records: every knowledge-base record (e.g. retriever.records or
            retriever.dense_records).
        expand_window: how many neighbouring chunks to include on each side
            (default 1), which captures adjacent formulas or surrounding context.
        merge_by_paragraph: merge hits that share a head_paragraph first, falling
            back to head_section.
        fill_internal_gaps: fill the chunks between two hits of the same
            subsection (for example the formula block sitting between block i
            and block i+2).
        max_tokens_or_chars: optional cap on the length of a single merged block.

    Returns:
        The enhanced, merged evidence blocks, preserving the descending order of
        the reranker scores.
    """
    if not retrieved_records:
        return []

    # 1. Build a fast lookup index over all documents
    pool = all_records if all_records is not None else retrieved_records
    doc_ordered_blocks = _ensure_doc_index(pool)

    block_pos_map: Dict[str, Dict[str, int]] = {}
    for doc_key, b_list in doc_ordered_blocks.items():
        block_pos_map[doc_key] = {}
        for idx, blk in enumerate(b_list):
            b_id = blk.get("block_id")
            if b_id:
                block_pos_map[doc_key][str(b_id)] = idx

    # 2. Group by doc_key
    retrieved_by_doc: Dict[str, List[Dict]] = {}
    for r in retrieved_records:
        doc_key = r.get("source_file") or r.get("doc_title") or r.get("doc_uuid") or "__default__"
        if doc_key not in retrieved_by_doc:
            retrieved_by_doc[doc_key] = []
        retrieved_by_doc[doc_key].append(r)

    enhanced_results: List[Dict] = []

    # 3. Merge and expand context inside each document, by paragraph or subsection
    for doc_key, hits in retrieved_by_doc.items():
        doc_chain = doc_ordered_blocks.get(doc_key, [])
        pos_lookup = block_pos_map.get(doc_key, {})

        # 3.1 Group by (head_paragraph or head_section)
        groups: Dict[str, List[Dict]] = {}
        for h in hits:
            sec_key = h.get("head_paragraph", "").strip() if merge_by_paragraph else ""
            if not sec_key:
                sec_key = h.get("head_section", "").strip() or "__global__"
            
            if sec_key not in groups:
                groups[sec_key] = []
            groups[sec_key].append(h)

        # 3.2 Sort, fill and expand each group at both ends
        for sec_key, group_hits in groups.items():
            valid_indices = []
            for gh in group_hits:
                bid = str(gh.get("block_id", ""))
                if bid in pos_lookup:
                    valid_indices.append(pos_lookup[bid])
            
            # When indices cannot be resolved against the global index, concatenate in group_hits order
            if not valid_indices or not doc_chain:
                combined_text = "\n\n".join(gh.get("text", "").strip() for gh in group_hits if gh.get("text", "").strip())
                best_score = max((gh.get("_rerank_score", gh.get("_rrf_score", gh.get("_dense_score", 0.0))) for gh in group_hits), default=0.0)
                first_h = group_hits[0]
                merged_record = dict(first_h)
                merged_record["text"] = combined_text
                merged_record["_rerank_score"] = best_score
                merged_record["_is_enhanced"] = False
                enhanced_results.append(merged_record)
                continue

            min_idx = min(valid_indices)
            max_idx = max(valid_indices)

            # (A) Fill the gaps inside the group
            if fill_internal_gaps and min_idx < max_idx:
                expanded_indices = set(range(min_idx, max_idx + 1))
            else:
                expanded_indices = set(valid_indices)

            # (B) Expand by expand_window chunks on each side
            curr_top = min_idx
            for _ in range(expand_window):
                prev_idx = curr_top - 1
                if prev_idx >= 0:
                    prev_blk = doc_chain[prev_idx]
                    curr_sec = doc_chain[curr_top].get("head_section", "")
                    prev_sec = prev_blk.get("head_section", "")
                    if curr_sec and prev_sec and curr_sec != prev_sec:
                        break
                    expanded_indices.add(prev_idx)
                    curr_top = prev_idx
                else:
                    break

            curr_bottom = max_idx
            for _ in range(expand_window):
                next_idx = curr_bottom + 1
                if next_idx < len(doc_chain):
                    next_blk = doc_chain[next_idx]
                    curr_sec = doc_chain[curr_bottom].get("head_section", "")
                    next_sec = next_blk.get("head_section", "")
                    if curr_sec and next_sec and curr_sec != next_sec:
                        break
                    expanded_indices.add(next_idx)
                    curr_bottom = next_idx
                else:
                    break

            sorted_indices = sorted(list(expanded_indices))
            included_blocks = [doc_chain[i] for i in sorted_indices]

            text_segments = []
            has_formula = False
            has_table = False
            seen_paragraphs = set()
            merged_ids = []
            for b in included_blocks:
                b_text = b.get("text", "").strip()
                if not b_text:
                    continue
                
                b_type = b.get("block_type", "")
                if b_type in ("formula", "equation") or "formula" in str(b.get("id", "")):
                    has_formula = True
                if b_type == "table":
                    has_table = True
                
                merged_ids.append(b.get("block_id") or b.get("id", ""))

                # 1. Emit the [paragraph] header once per identical head_paragraph
                curr_para = b.get("head_paragraph", "").strip()
                if curr_para and curr_para not in seen_paragraphs:
                    seen_paragraphs.add(curr_para)
                    text_segments.append(f"[段落：{curr_para}]")

                # 2. Structured table enhancement: label the caption and wrap the table in a Markdown code fence
                if b_type == "table":
                    table_title = b.get("title", "").strip()
                    title_prefix = f"【表格：{table_title}】\n" if table_title else "【表格数据】\n"
                    table_block = f"{title_prefix}```markdown\n{b_text}\n```"
                    text_segments.append(table_block)
                else:
                    text_segments.append(b_text)

            merged_text = "\n\n".join(text_segments)
            if max_tokens_or_chars and len(merged_text) > max_tokens_or_chars:
                merged_text = merged_text[:max_tokens_or_chars]

            best_rerank_score = max(
                (gh.get("_rerank_score", gh.get("_rrf_score", gh.get("_dense_score", 0.0))) for gh in group_hits),
                default=0.0
            )

            base_meta = dict(group_hits[0])
            base_meta["text"] = merged_text
            base_meta["_rerank_score"] = best_rerank_score
            base_meta["_is_enhanced"] = True
            base_meta["_has_formula"] = has_formula
            base_meta["_has_table"] = has_table
            base_meta["_merged_block_ids"] = merged_ids
            base_meta["_merged_count"] = len(included_blocks)
            enhanced_results.append(base_meta)

    enhanced_results.sort(
        key=lambda x: x.get("_rerank_score", x.get("_rrf_score", x.get("_dense_score", 0.0))),
        reverse=True
    )
    return enhanced_results


# ==================== Stage 4: generation ====================
def reorder_to_u_shape(records: List[Dict]) -> List[Dict]:
    """Mitigate lost-in-the-middle attention decay by alternating the strongest evidence between the start and the end of the context."""
    if len(records) <= 2:
        return list(records)
    left, right = [], []
    for i, doc in enumerate(records):
        (left if i % 2 == 0 else right).append(doc)
    return left + list(reversed(right))


def build_rag_context(
    records: List[Dict],
    max_chars_per_doc: int = GENERATION_CONTEXT_MAX_CHARS_PER_DOC,
    max_chars_total: int = GENERATION_CONTEXT_MAX_CHARS_TOTAL,
    use_u_shape: bool = True,
) -> str:
    if records and use_u_shape:
        records = reorder_to_u_shape(records)
    """Assemble the retrieved records into a numbered, source-annotated context for the generator."""
    context_parts = []
    used_chars = 0
    for rank, record in enumerate(records, 1):
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        text = text[:max_chars_per_doc]
        if used_chars + len(text) > max_chars_total:
            remaining = max_chars_total - used_chars
            if remaining <= 0:
                break
            text = text[:remaining]
        source = record.get("doc_title") or record.get("source_file") or "未知来源"
        section = record.get("head_section", "")
        location = f"；章节：{section}" if section else ""
        has_table = record.get("_has_table", False)
        type_tag = " (含表格)" if has_table else ""
        context_parts.append(f"[证据{rank}{type_tag}] 来源：{source}{location}\n{text}")
        used_chars += len(text)
        if used_chars >= max_chars_total:
            break
    return "\n\n".join(context_parts)


def generate_answer(
    query: str,
    retrieved_records: List[Dict],
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, str]:
    """Generate an answer from the retrieved evidence via Chat Completions; returns (answer, finish_reason).
    When max_tokens is None the parameter is omitted, letting the model use the platform default budget for reasoning plus the final answer.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    context = build_rag_context(retrieved_records)
    if not context:
        return "知识库中没有检索到可用于回答该问题的证据。", "no_context"
    if client is None:
        client = get_generation_client()

    target_model = model or GENERATION_MODEL
    target_temp = temperature if temperature is not None else GENERATION_TEMPERATURE

    system_prompt = (
        "【系统提示】\n"
        "你是一名专业的钢铁行业减污降碳与清洁生产技术专家。\n"
        "请结合【参考背景资料】以及你的专业工程常识，准确、严谨、直奔要点地回答问题。\n"
        "核心要求：\n"
        "1. 【直奔结论】：回答开头必须直接给出明确的定量结论或核心观点，重点突出关键数值、指标变化与核心机理，避免冗长的背景铺垫；\n"
        "2. 【证据锚定】：以参考背景资料中的事实与数据为主要支撑依据，简要标注证据来源编号即可，严禁为了罗列证据而重复堆砌自辩废话；\n"
        "3. 【常识推导与补全】：若参考资料已涵盖核心事实，严格基于资料准确回答；若资料欠缺部分次要细节（如某种工况或对比项数据未完全提供），请在依据已有资料作答的同时，结合专业工程常识进行合理推导与完整补全，力求给出结论明确、专业可用的技术解答。"
    )
    user_prompt = (
        f"问题：{query.strip()}\n\n"
        f"【参考背景资料】：\n{context}\n\n"
        "请直接给出专业、定量、精炼的回答："
    )

    request_kwargs = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": target_temp,
        "timeout": GENERATION_API_TIMEOUT,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request_kwargs)
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", "unknown") or "unknown"
    answer = choice.message.content
    if not answer:
        raise RuntimeError("The generation API returned an empty answer")
    return answer.strip(), finish_reason

_EXPANDED_QUERY_CACHE: Optional[Dict[str, str]] = None


def expand_query_with_llm(
    question: str,
    api_base: str = EXPANSION_API_BASE,
    api_key: str = EXPANSION_API_KEY,
    model: str = EXPANSION_MODEL,
    timeout: float = EXPANSION_API_TIMEOUT,
) -> str:
    """Bilingual terminology expansion: prompt an LLM to align the Chinese question with the corresponding English technical terms."""
    if not question or not question.strip():
        return ""
    if not api_key:
        return question.strip()

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
    system_prompt = (
        "你是一名材料化工与冶金领域的双语专业学术检索助手。\n"
        "请对用户给出的中文技术问题进行专业术语与国际学术缩写对齐重写，生成适合跨语言向量检索的扩展Query。\n"
        "规则：\n"
        "1. 保留中文核心问题；\n"
        "2. 提取并追加对应的国际标准学术英文名称、专业缩写、化学式或前沿工艺代号（如：全氧高炉 -> Full Oxygen Blast Furnace, FOBF；干法重整 -> Dry Reforming of Methane, DRM；焦炉煤气 -> Coke Oven Gas, COG）；\n"
        "3. 只输出合并后的检索串，禁止解释或回答问题。"
    )
    user_prompt = f"技术问题：{question.strip()}\n扩展Query："

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        expanded = (resp.choices[0].message.content or "").strip()
        return expanded if expanded else question.strip()
    except Exception as e:
        print(f"      \u26a0\ufe0f Bilingual expansion call failed, falling back to the original query: {e}")
        return question.strip()


def get_expanded_query(
    qid: Optional[str],
    original_query: str,
    cache_file: Optional[Union[str, Path]] = None,
    allow_llm_fallback: bool = True,
) -> str:
    """Return the bilingual-expanded query.

    Resolution order:
    1. read the static expansion cache first (O(1), no latency);
    2. on a cache miss, and if LLM generation is allowed, call the model and write the result back to the cache file;
    3. if the call fails or LLM generation is disabled, fall back to the original query.
    """
    global _EXPANDED_QUERY_CACHE
    if not original_query or not original_query.strip():
        return ""

    resolved_path = None
    if _EXPANDED_QUERY_CACHE is None:
        _EXPANDED_QUERY_CACHE = {}
        target_path = (
            Path(cache_file)
            if cache_file
            else (
                Path(__file__).resolve().parent
                / "cache"
                / "query_expansion_355.json"
            )
        )
        if not target_path.exists():
            target_path = Path(__file__).resolve().parent / "query_expansion_355.json"

        resolved_path = target_path
        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    _EXPANDED_QUERY_CACHE = json.load(f)
            except Exception as e:
                print(f"      \u26a0\ufe0f Failed to read the query_expansion_355.json cache: {e}")
                _EXPANDED_QUERY_CACHE = {}
    else:
        resolved_path = (
            Path(cache_file)
            if cache_file
            else (Path(__file__).resolve().parent / "query_expansion_355.json")
        )

    # 1. Cache hit
    if qid and qid in _EXPANDED_QUERY_CACHE:
        return _EXPANDED_QUERY_CACHE[qid]

    # 2. Cache miss: generate with the LLM and persist the result
    if allow_llm_fallback:
        generated_expanded = expand_query_with_llm(original_query)
        if generated_expanded and generated_expanded != original_query.strip():
            if qid:
                _EXPANDED_QUERY_CACHE[qid] = generated_expanded
                # Persist immediately so later runs can reuse it
                if resolved_path:
                    try:
                        resolved_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(resolved_path, "w", encoding="utf-8") as f_out:
                            json.dump(_EXPANDED_QUERY_CACHE, f_out, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"      \u26a0\ufe0f Failed to write the query_expansion cache file: {e}")
            return generated_expanded

    return original_query.strip()

def answer_with_retriever(
    query: str, retriever, top_k: int = TOP_K_RERANK,
    reranker: Optional[Reranker] = None,
    enable_context_enhancement: bool = True,
    expand_window: int = 1,
) -> Dict:
    """Full RAG call: retrieve (with optional reranking) -> enhance and merge context -> assemble the prompt -> generate.

    With ``reranker`` set to None the legacy ``retriever.search`` behaviour is kept; when it is
    provided, ``RerankedRetriever`` fetches candidates first and then reranks them.
    ``enable_context_enhancement`` toggles context expansion and neighbour merging after reranking.
    """
    pipeline = RerankedRetriever(retriever, reranker) if reranker is not None else retriever
    records = pipeline.search(query, top_k=top_k)

    if enable_context_enhancement:
        # Fetch the full record set (preferring retriever.records or retriever.dense_records)
        all_recs = getattr(retriever, "records", None)
        if all_recs is None:
            all_recs = getattr(retriever, "dense_records", None)
        records = enhance_context(
            retrieved_records=records,
            all_records=all_recs,
            expand_window=expand_window,
            fill_internal_gaps=True,
        )

    answer = generate_answer(query, records)
    return {"query": query, "answer": answer, "sources": records}
