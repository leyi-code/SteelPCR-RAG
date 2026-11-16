# -*- coding: utf-8 -*-
"""
SteelPCR-RAG retrieval and generation framework, shared by the four case scripts under probe_case/.

Chain:
  * dense retrieval: Qwen3-Embedding-8B (4096-d) plus FAISS-HNSW, top-50 initial search
  * evidence reranking: Qwen3-Reranker-8B over HTTP, keeping the top 7
  * context organisation: enhance_context (neighbour expansion, paragraph merging, table fencing), identical
    to rag_core/general_RAG.py except that the full-record index is replaced by an on-demand read-only
    document chain, so the semantics are the same and the records file never has to be held in memory
  * metadata annotation, U-shape reordering and context assembly: same as general_RAG.build_rag_context
  * FAISS is opened read-only via mmap
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import requests
from openai import OpenAI

# Make the repository-level configuration module importable when this package
# is executed from anywhere inside the repository.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from steelpcr_config import env, env_bool, env_float, env_int, repo_path  # noqa: E402

# ==================== Configuration (aligned with evaluation/eval3.py) ====================
DENSE_DIR = repo_path("STEELPCR_DENSE_INDEX_DIR", "rag_core/cache/dense")
INDEX_PATH = env("STEELPCR_FAISS_INDEX", os.path.join(DENSE_DIR, "faiss_hnsw.index"))
RECORDS_PATH = env("STEELPCR_RECORDS", os.path.join(DENSE_DIR, "records.jsonl"))

TOP_K_RETRIEVE = env_int("STEELPCR_TOP_K_RETRIEVE", 50)
TOP_K_RERANK = env_int("STEELPCR_TOP_K_RERANK", 7)
HNSW_EF_SEARCH = env_int("STEELPCR_HNSW_EF_SEARCH", 128)
ENABLE_CONTEXT_ENHANCEMENT = env_bool("STEELPCR_CTX_ENHANCEMENT", True)

# ---- Embedding API ----
EMB_API_BASE = env("STEELPCR_EMBED_API_BASE", "your api url")
EMB_API_KEY = env("STEELPCR_EMBED_API_KEY", "your api key")
EMB_MODEL = env("STEELPCR_EMBED_MODEL", "Qwen3-Embedding-8B")
EMB_DIM = env_int("STEELPCR_EMBED_DIM", 4096)

# ---- Reranker API (Qwen3-Reranker-8B) ----
RERANK_URL = env("STEELPCR_QWEN3_RERANK_URL", "your api url")
RERANK_KEY = env("STEELPCR_QWEN3_RERANK_KEY", EMB_API_KEY)
RERANK_MODEL = env("STEELPCR_QWEN3_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
RERANK_BATCH = env_int("STEELPCR_QWEN3_RERANK_BATCH_SIZE", 25)
RERANK_TIMEOUT = env_float("STEELPCR_RERANK_API_TIMEOUT", 60.0)
RERANK_RETRIES = env_int("STEELPCR_RERANK_MAX_RETRIES", 2)

# ---- Generation model ----
GEN_BASE = env("STEELPCR_GEN_STRONG_API_BASE", "https://api.deepseek.com")
GEN_KEY = env("STEELPCR_GEN_STRONG_API_KEY", "your api key")
GEN_MODEL = env("STEELPCR_GEN_STRONG_MODEL", "deepSeek-flash")
GEN_TEMPERATURE = env_float("STEELPCR_GEN_TEMPERATURE", 0.0)
GEN_MAX_TOKENS = env_int("STEELPCR_GEN_MAX_TOKENS", 32000)
GEN_TIMEOUT = env_float("STEELPCR_GEN_TIMEOUT", 600.0)
GEN_RETRIES = env_int("STEELPCR_GEN_MAX_RETRIES", 2)

# ---- Query expansion (bilingual terminology alignment) ----
EXP_BASE = env("STEELPCR_GEN_STRONG_API_BASE", GEN_BASE)
EXP_KEY = env("STEELPCR_EXPANSION_API_KEY", GEN_KEY)
EXP_MODEL = env("STEELPCR_EXPANSION_MODEL", GEN_MODEL)

# ---- Context budget (paper defaults) ----
GENERATION_CONTEXT_MAX_CHARS_PER_DOC = 3000
GENERATION_CONTEXT_MAX_CHARS_TOTAL = 12000

def log(msg: str) -> None:
    print(msg, flush=True)


# ==================== On-demand read-only record store ====================
class LazyRecords:
    """Read records.jsonl at random byte offsets, parsing only the records that are actually used.

    Grouping matches `general_RAG._ensure_doc_index`: records are grouped by
    source_file/doc_title/doc_uuid and keep their original order inside each group.
    """

    def __init__(self, path: str = RECORDS_PATH):
        self.path = path
        self.entries: List[Tuple[str, int, int]] = []  # (doc_key, offset, length)
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._doc_index: Optional[Dict[str, List[int]]] = None
        t0 = time.time()
        with open(path, "rb") as f:
            offset = 0
            for raw in f:
                length = len(raw)
                if raw.strip():
                    key = self._extract_doc_key(raw)
                    self.entries.append((key, offset, length))
                offset += length
        log(f"[LazyRecords] indexed {len(self.entries)} records in {time.time()-t0:.2f}s")

    @staticmethod
    def _extract_doc_key(raw: bytes) -> str:
        m = re.search(rb'"source_file"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            return m.group(1).decode("utf-8", "ignore")
        return "__default__"

    def _read(self, i: int) -> Dict[str, Any]:
        if i in self._cache:
            return self._cache[i]
        _, off, ln = self.entries[i]
        with open(self.path, "rb") as f:
            f.seek(off)
            raw = f.read(ln)
        rec = json.loads(raw.decode("utf-8"))
        self._cache[i] = rec
        return rec

    def get(self, i: int) -> Dict[str, Any]:
        return dict(self._read(i))

    def build_doc_index(self) -> Dict[str, List[int]]:
        if self._doc_index is None:
            idx: Dict[str, List[int]] = {}
            for i, (key, _, _) in enumerate(self.entries):
                idx.setdefault(key, []).append(i)
            self._doc_index = idx
        return self._doc_index

    def doc_chain(self, doc_key: str) -> List[Dict[str, Any]]:
        rows = self.build_doc_index().get(doc_key, [])
        return [self._read(i) for i in rows]


# ==================== Embedding ====================
def _emb_client() -> OpenAI:
    return OpenAI(api_key=EMB_API_KEY, base_url=EMB_API_BASE, timeout=180.0)


def embed_texts(texts: Sequence[str], retries: int = 3) -> np.ndarray:
    if not texts:
        return np.empty((0, EMB_DIM), dtype="float32")
    client = _emb_client()
    last = None
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(
                model=EMB_MODEL, input=list(texts),
                encoding_format="float", dimensions=EMB_DIM)
            vecs = np.asarray([d.embedding for d in resp.data], dtype="float32")
            if vecs.shape[0] != len(texts) or vecs.shape[1] != EMB_DIM:
                raise RuntimeError(f"unexpected embedding shape {vecs.shape}")
            faiss.normalize_L2(vecs)
            return vecs
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Embedding API call failed: {last}")


# ==================== Reranker API ====================
class Qwen3RerankerAPI:
    def __init__(self, url: str = RERANK_URL, api_key: str = RERANK_KEY,
                 model: str = RERANK_MODEL, batch_size: int = RERANK_BATCH,
                 timeout: float = RERANK_TIMEOUT, max_retries: int = RERANK_RETRIES):
        self.url, self.api_key, self.model = url, api_key, model
        self.batch_size, self.timeout, self.max_retries = batch_size, timeout, max_retries
        self.calls = 0

    def rerank(self, query: str, candidates: Sequence[Dict],
               top_k: int = TOP_K_RERANK) -> Tuple[List[Dict], float, Optional[str]]:
        if not candidates:
            return [], 0.0, None
        t0 = time.perf_counter()
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        scored: List[Dict] = []
        err: Optional[str] = None

        for start in range(0, len(candidates), self.batch_size):
            batch = list(candidates[start:start + self.batch_size])
            payload = {"model": self.model, "query": query,
                       "documents": [c.get("text", "") for c in batch],
                       "top_n": len(batch)}
            data, last_err = None, None
            for attempt in range(self.max_retries + 1):
                try:
                    resp = requests.post(self.url, json=payload, headers=headers,
                                         timeout=self.timeout)
                    self.calls += 1
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                    last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                if attempt < self.max_retries:
                    time.sleep(2.0)

            results = []
            if isinstance(data, dict):
                results = data.get("results") or (data.get("output") or {}).get("results") or []
            if results:
                for item in results:
                    idx = item.get("index")
                    if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                        raise RuntimeError(f"rerank API returned an invalid index: {idx}")
                    scored.append(dict(batch[idx],
                                       _rerank_score=float(item.get("relevance_score", 0.0))))
            else:
                err = f"batch [{start}:{start+len(batch)}] failed: {last_err}"
                log(f"      ⚠️ {err}")
                for c in batch:
                    scored.append(dict(c, _rerank_score=0.0))

        scored.sort(key=lambda c: c["_rerank_score"], reverse=True)
        latency = (time.perf_counter() - t0) * 1000.0
        return scored[:top_k], latency, err


# ==================== Context organisation (copied from general_RAG) ====================
def enhance_context(
    retrieved_records: Sequence[Dict],
    records: "LazyRecords",
    expand_window: int = 1,
    merge_by_paragraph: bool = True,
    fill_internal_gaps: bool = True,
    max_tokens_or_chars: Optional[int] = None,
) -> List[Dict]:
    """Expand and merge the reranked candidates by neighbour and by paragraph.

    The logic copies `rag_core.general_RAG.enhance_context` line by line; the only change is that
    `all_records` becomes `LazyRecords`, so an ordered chain is built only for the hit documents.
    """
    if not retrieved_records:
        return []

    def doc_key_of(r: Dict) -> str:
        return r.get("source_file") or r.get("doc_title") or r.get("doc_uuid") or "__default__"

    need_docs = {doc_key_of(r) for r in retrieved_records}
    doc_ordered_blocks = {k: records.doc_chain(k) for k in need_docs}

    block_pos_map: Dict[str, Dict[str, int]] = {}
    for doc_key, b_list in doc_ordered_blocks.items():
        block_pos_map[doc_key] = {}
        for i, blk in enumerate(b_list):
            b_id = blk.get("block_id")
            if b_id:
                block_pos_map[doc_key][str(b_id)] = i

    retrieved_by_doc: Dict[str, List[Dict]] = {}
    for r in retrieved_records:
        retrieved_by_doc.setdefault(doc_key_of(r), []).append(r)

    enhanced_results: List[Dict] = []

    for doc_key, hits in retrieved_by_doc.items():
        doc_chain = doc_ordered_blocks.get(doc_key, [])
        pos_lookup = block_pos_map.get(doc_key, {})

        groups: Dict[str, List[Dict]] = {}
        for h in hits:
            sec_key = h.get("head_paragraph", "").strip() if merge_by_paragraph else ""
            if not sec_key:
                sec_key = h.get("head_section", "").strip() or "__global__"
            groups.setdefault(sec_key, []).append(h)

        for sec_key, group_hits in groups.items():
            valid_indices = []
            for gh in group_hits:
                bid = str(gh.get("block_id", ""))
                if bid in pos_lookup:
                    valid_indices.append(pos_lookup[bid])

            if not valid_indices or not doc_chain:
                combined = "\n\n".join(gh.get("text", "").strip()
                                       for gh in group_hits if gh.get("text", "").strip())
                best = max((gh.get("_rerank_score", 0.0) for gh in group_hits), default=0.0)
                merged = dict(group_hits[0])
                merged["text"] = combined
                merged["_rerank_score"] = best
                merged["_is_enhanced"] = False
                enhanced_results.append(merged)
                continue

            min_idx, max_idx = min(valid_indices), max(valid_indices)

            if fill_internal_gaps and min_idx < max_idx:
                expanded = set(range(min_idx, max_idx + 1))
            else:
                expanded = set(valid_indices)

            cur_top = min_idx
            for _ in range(expand_window):
                prev = cur_top - 1
                if prev >= 0:
                    cur_sec = doc_chain[cur_top].get("head_section", "")
                    prev_sec = doc_chain[prev].get("head_section", "")
                    if cur_sec and prev_sec and cur_sec != prev_sec:
                        break
                    expanded.add(prev)
                    cur_top = prev
                else:
                    break

            cur_bot = max_idx
            for _ in range(expand_window):
                nxt = cur_bot + 1
                if nxt < len(doc_chain):
                    cur_sec = doc_chain[cur_bot].get("head_section", "")
                    nxt_sec = doc_chain[nxt].get("head_section", "")
                    if cur_sec and nxt_sec and cur_sec != nxt_sec:
                        break
                    expanded.add(nxt)
                    cur_bot = nxt
                else:
                    break

            sorted_idx = sorted(expanded)
            included = [doc_chain[i] for i in sorted_idx]

            segments: List[str] = []
            has_formula = has_table = False
            seen_paras = set()
            merged_ids: List[str] = []
            for b in included:
                b_text = (b.get("text") or "").strip()
                if not b_text:
                    continue
                b_type = b.get("block_type", "")
                if b_type in ("formula", "equation") or "formula" in str(b.get("id", "")):
                    has_formula = True
                if b_type == "table":
                    has_table = True
                merged_ids.append(b.get("block_id") or b.get("id", ""))

                cur_para = (b.get("head_paragraph") or "").strip()
                if cur_para and cur_para not in seen_paras:
                    seen_paras.add(cur_para)
                    segments.append(f"[段落：{cur_para}]")

                if b_type == "table":
                    title = (b.get("title") or "").strip()
                    prefix = f"【表格：{title}】\n" if title else "【表格数据】\n"
                    segments.append(f"{prefix}```markdown\n{b_text}\n```")
                else:
                    segments.append(b_text)

            merged_text = "\n\n".join(segments)
            if max_tokens_or_chars and len(merged_text) > max_tokens_or_chars:
                merged_text = merged_text[:max_tokens_or_chars]

            best = max((gh.get("_rerank_score", 0.0) for gh in group_hits), default=0.0)
            base = dict(group_hits[0])
            base.update({
                "text": merged_text,
                "_rerank_score": best,
                "_is_enhanced": True,
                "_has_formula": has_formula,
                "_has_table": has_table,
                "_merged_block_ids": merged_ids,
                "_merged_count": len(included),
            })
            enhanced_results.append(base)

    enhanced_results.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
    return enhanced_results


def reorder_to_u_shape(records: List[Dict]) -> List[Dict]:
    if len(records) <= 2:
        return list(records)
    left, right = [], []
    for i, d in enumerate(records):
        (left if i % 2 == 0 else right).append(d)
    return left + list(reversed(right))


def build_rag_context(
    records: List[Dict],
    max_chars_per_doc: int = GENERATION_CONTEXT_MAX_CHARS_PER_DOC,
    max_chars_total: int = GENERATION_CONTEXT_MAX_CHARS_TOTAL,
    use_u_shape: bool = True,
) -> str:
    if records and use_u_shape:
        records = reorder_to_u_shape(records)
    parts: List[str] = []
    used = 0
    rank = 0
    for record in records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        rank += 1
        text = text[:max_chars_per_doc]
        if used + len(text) > max_chars_total:
            remain = max_chars_total - used
            if remain <= 0:
                break
            text = text[:remain]
        source = record.get("doc_title") or record.get("source_file") or "未知来源"
        section = record.get("head_section", "")
        location = f"；章节：{section}" if section else ""
        tag = " (含表格)" if record.get("_has_table") else ""
        parts.append(f"[证据{rank}{tag}] 来源：{source}{location}\n{text}")
        used += len(text)
        if used >= max_chars_total:
            break
    return "\n\n".join(parts)


# ==================== Generation ====================
# Note: this prompt was written for the present experiment; it is not general_RAG's original prompt.
# Design goal: let the model judge evidence validity and reason like an engineer, while forcing it to answer
# from the evidence wherever the evidence covers the question, and to separate evidence conclusions,
SYSTEM_PROMPT = (
    "【系统提示】\n"
    "你是一名钢铁行业减污降碳与清洁生产技术专家。你的任务是依据给定的【参考背景资料】回答工程问题。\n"
    "\n"
    "【核心原则：先判证据，再作答】\n"
    "你可以自行判断每条参考证据是否真正切题、是否可信、适用于什么条件；也有权基于专业工程常识做推理。\n"
    "但必须遵守下列证据分级规则：\n"
    "1. 【覆盖即照证据】凡参考资料已直接回答的要点（数值、比例、温度、气源、趋势方向、约束条件等），必须严格照证据作答：\n"
    "   - 数值、单位、配比、温度、工况参数一律照原文转述，不得改写、取整、放大或跨工况套用；\n"
    "   - 结论方向（升高/降低/先降后升/存在阈值/无显著变化）必须与证据一致，不得反转或模糊化；\n"
    "   - 每条证据结论都必须绑定其适用条件（原料类型 × 工况参数 × 系统边界），并标注 [证据N]；\n"
    "   - 严禁把单一工况下的结论泛化成普适结论。\n"
    "2. 【未覆盖可推理，但必须标注】凡参考资料未覆盖的要点，可结合工程常识补全，但必须以【外推】显式标注并说明推理依据；\n"
    "   不得把外推内容写成证据结论，也不得为凑完整而编造具体数值。\n"
    "3. 【冲突与权衡必须揭示】若证据之间存在方向性冲突或权衡（例如降碳效果与污染物负荷、直接排放与上游间接排放、\n"
    "   抑制效果与添加比例/温度窗口），必须显式指出：冲突是否存在、各自成立的条件、以及可能的来源（原料差异、工艺参数、系统边界口径）。\n"
    "4. 【证据缺口如实说明】某要点参考资料完全未涉及，就写\"资料未提供\"，不要用常识猜测冒充实证。\n"
    "\n"
    "【输出结构】\n"
    "一、核心结论：先给直接判断（含关键数值与成立条件），3~6 句。\n"
    "二、逐工况/逐要点证据锚定：每条写明「证据结论 + 适用条件 + [证据N]」；工况间结论不一致时并列给出，不要强行合并。\n"
    "三、文献分歧与权衡：列出方向性冲突或权衡关系及其条件依赖；确无则写\"未见直接冲突证据\"。\n"
    "四、证据缺口与外推：哪些要点资料未覆盖、哪些结论属【外推】。\n"
    "五、参考来源清单：把正文真正引用为依据的 [证据N] 按来源文献归并，一篇一行，格式「- 文献标题：k 条（证据N、证据M…）」：\n"
    "   - 文献标题必须逐字复制参考资料证据行中「来源：」之后的文本；该行若含「；章节：」及其后内容，不得复制；不得改写、缩写、翻译或补全标题；\n"
    "   - 同一文献的多条证据合并为一行，k 为去重后的证据编号个数，编号按升序排列；\n"
    "   - 只列正文中真正引用为依据的证据，不列仅转述而未被引用的证据；\n"
    "   - 各行按该文献在正文中首次被引用的顺序排列；\n"
    "   - 本节只做来源归并，不得新增任何结论、数值或判断；正文未引用任何证据时只写「无」。\n"
    "\n"
    "语言精炼、专业、定量，不要复述背景铺垫。"
)

_gen_client: Optional[OpenAI] = None


def gen_client() -> OpenAI:
    global _gen_client
    if _gen_client is None:
        _gen_client = OpenAI(api_key=GEN_KEY, base_url=GEN_BASE, timeout=GEN_TIMEOUT)
    return _gen_client


def chat(system: Optional[str], user: str, temperature: float = GEN_TEMPERATURE,
         max_tokens: int = GEN_MAX_TOKENS, retries: int = GEN_RETRIES) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    last = None
    for attempt in range(retries + 1):
        try:
            resp = gen_client().chat.completions.create(
                model=GEN_MODEL, messages=msgs,
                temperature=temperature, max_tokens=max_tokens)
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            last = f"empty answer finish_reason={getattr(resp.choices[0],'finish_reason',None)}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(3.0)
    raise RuntimeError(f"generation failed: {last}")


def generate_answer(query: str, records: List[Dict]) -> str:
    """Context assembly matches general_RAG.build_rag_context; the prompt is this experiment's graded-evidence version."""
    context = build_rag_context(records)
    if not context:
        return "知识库中没有检索到可用于回答该问题的证据。"
    user_prompt = (
        f"问题：{query.strip()}\n\n"
        f"【参考背景资料】：\n{context}\n\n"
        "请直接给出专业、定量、精炼的回答："
    )
    return chat(SYSTEM_PROMPT, user_prompt)


# ==================== Query expansion (bilingual terminology alignment) ====================
EXPAND_SYSTEM = (
    "你是一名材料化工与冶金领域的双语专业学术检索助手。\n"
    "请对用户给出的中文技术问题进行专业术语与国际学术缩写对齐重写，生成适合跨语言向量检索的扩展Query。\n"
    "规则：\n"
    "1. 保留中文核心问题；\n"
    "2. 提取并追加对应的国际标准学术英文名称、专业缩写、化学式或前沿工艺代号（如：全氧高炉 -> Full Oxygen Blast Furnace, FOBF；干法重整 -> Dry Reforming of Methane, DRM；焦炉煤气 -> Coke Oven Gas, COG）；\n"
    "3. 只输出合并后的检索串，禁止解释或回答问题。"
)


def expand_query(question: str, enable: bool = True) -> str:
    if not enable or not question.strip():
        return question.strip()
    try:
        out = chat(EXPAND_SYSTEM, f"技术问题：{question.strip()}\n扩展Query：",
                   temperature=0.0, max_tokens=1024)
        return out.strip() or question.strip()
    except Exception as exc:  # noqa: BLE001
        log(f"      \u26a0\ufe0f Bilingual expansion failed, falling back to the original query: {exc}")
        return question.strip()


# ==================== Full retrieval chain ====================
class SteelPCRPipeline:
    def __init__(self, enable_expansion: bool = True):
        log("[Pipeline] loading FAISS-HNSW (mmap, read-only) ...")
        t0 = time.time()
        self.index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        self.index.hnsw.efSearch = HNSW_EF_SEARCH
        log(f"[Pipeline] index ready ntotal={self.index.ntotal} d={self.index.d} "
            f"({time.time()-t0:.1f}s)")
        self.records = LazyRecords()
        self.reranker = Qwen3RerankerAPI()
        self.enable_expansion = enable_expansion

    # --- Single round: expand -> dense top 50 -> rerank top 7 -> context enhancement ---
    def retrieve(self, query: str, top_k: int = TOP_K_RERANK,
                 top_n_retrieve: int = TOP_K_RETRIEVE,
                 enable_expansion: Optional[bool] = None) -> Dict[str, Any]:
        t_start = time.perf_counter()
        use_exp = self.enable_expansion if enable_expansion is None else enable_expansion
        expanded = expand_query(query, use_exp)

        t0 = time.perf_counter()
        qv = embed_texts([expanded])
        t_emb = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dist, idxs = self.index.search(qv, top_n_retrieve)
        t_search = (time.perf_counter() - t0) * 1000
        candidates = []
        for d, i in zip(dist[0], idxs[0]):
            if 0 <= i < len(self.records.entries):
                rec = self.records.get(int(i))
                rec["_dense_dist"] = float(d)
                rec["_row"] = int(i)
                candidates.append(rec)

        reranked, t_rerank, err = self.reranker.rerank(expanded, candidates, top_k)

        enhanced = reranked
        if ENABLE_CONTEXT_ENHANCEMENT:
            enhanced = enhance_context(retrieved_records=reranked, records=self.records,
                                       expand_window=1, fill_internal_gaps=True)

        return {
            "query": query,
            "expanded_query": expanded,
            "candidates": candidates,
            "reranked": reranked,
            "enhanced": enhanced,
            "rerank_error": err,
            "latency_ms": {
                "embed": round(t_emb, 1),
                "search": round(t_search, 1),
                "rerank": round(t_rerank, 1),
                "total": round((time.perf_counter() - t_start) * 1000, 1),
            },
        }

    def answer(self, query: str, top_k: int = TOP_K_RERANK,
               enable_expansion: Optional[bool] = None) -> Dict[str, Any]:
        pack = self.retrieve(query, top_k=top_k, enable_expansion=enable_expansion)
        t0 = time.perf_counter()
        ans = generate_answer(query, pack["enhanced"])
        pack["answer"] = ans
        pack["latency_ms"]["generate"] = round((time.perf_counter() - t0) * 1000, 1)
        pack["context"] = build_rag_context(pack["enhanced"])
        return pack


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def dump_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pipe = SteelPCRPipeline()
    pack = pipe.answer("烧结配加生物质炭对烟气SO2和NOx排放有什么影响？")
    log("=" * 70)
    log("expanded query: " + pack["expanded_query"])
    log(f"latency: {pack['latency_ms']}")
    log("reranked top 7:")
    for i, r in enumerate(pack["reranked"], 1):
        log(f"  [{i}] {r['_rerank_score']:.4f}  {r.get('doc_title','')[:60]} | {r.get('head_section','')[:40]}")
    log("=" * 70)
    log("answer:\n" + pack["answer"])
