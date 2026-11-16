# -*- coding: utf-8 -*-
"""Index construction for the SteelPCR-RAG knowledge base.

Reads the structured JSON documents, encodes them through the embedding API with
per-file resume checkpoints, then writes the FAISS-HNSW dense index and the BM25
sparse index. The query path lives in general_RAG.py and is not touched here.

Run:  python rag_core/build_index.py
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from steelpcr_config import env  # noqa: E402

from rag_core.general_RAG import (  # noqa: E402
    DENSE_DIR,
    EMBEDDING_MODEL_API,
    HNSW_EF_SEARCH,
    SPARSE_DIR,
    _load_records,
    embed_via_api,
    tokenize_for_bm25,
)

# ==================== Index-build parameters ====================
# Directory holding one resume checkpoint per source file during batch embedding.
CHECKPOINT_DIR = "./cache/embeddings/checkpoints"
CHECKPOINT_MANIFEST_NAME = "manifest.json"

# INPUT_MODE: "single" processes INPUT_PATH, "folder" processes every JSON under INPUT_DIR.
INPUT_MODE = env("STEELPCR_INPUT_MODE", "folder")
INPUT_PATH = env("STEELPCR_INPUT_PATH", "")
INPUT_DIR = env("STEELPCR_INPUT_DIR", str(Path(__file__).resolve().parents[1] / "QA_generation" / "data"))

EMBED_DIR = "./cache/embeddings"

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200


def load_structured_documents(input_path: str) -> List[Dict]:
    records = []
    files = []
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".json")
            and os.path.isfile(os.path.join(input_path, f))
        )
    else:
        files = [input_path]

    for fp in files:
        print(f"[load] {os.path.basename(fp)}")
        with open(fp, "r", encoding="utf-8") as f:
            doc = json.load(f)

        doc_meta = {
            "doc_title": doc.get("metadata", {}).get("title", ""),
            "doc_uuid": doc.get("uuid", ""),
            "doc_source": doc.get("metadata", {}).get("source_type", ""),
        }

        for para in doc.get("body_text", []):
            text = para.get("text", "").strip()
            if not text:
                continue
            records.append({
                **doc_meta,
                "block_id": para.get("id", ""),
                "block_type": para.get("type", "text"),
                "text": text,
                "head_section": para.get("head_section", ""),
                "head_paragraph": para.get("head_paragraph", ""),
                "refs": para.get("refs", []),
                "source_file": os.path.basename(fp),
            })

        for idx, ft in enumerate(doc.get("figures_and_tables", [])):
            text = ft.get("text", "").strip()
            if not text:
                continue
            records.append({
                **doc_meta,
                "block_id": ft.get("id", f"ft_{idx}"),
                "block_type": ft.get("type", "table"),
                "text": text,
                "head_section": ft.get("head_section", ""),
                "head_paragraph": "",
                "title": ft.get("title", ""),
                "source_file": os.path.basename(fp),
            })

    print(f"[load] extracted {len(records)} text blocks")
    return records


def _file_signature(path: str) -> Dict:
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _checkpoint_paths(output_dir: str):
    checkpoint_dir = CHECKPOINT_DIR
    manifest_path = os.path.join(checkpoint_dir, CHECKPOINT_MANIFEST_NAME)
    return checkpoint_dir, manifest_path


def _load_manifest(manifest_path: str) -> Dict:
    if not os.path.exists(manifest_path):
        return {"files": {}}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        print(f"[checkpoint] manifest unusable, reprocessing: {manifest_path}")
    return {"files": {}}


def _save_manifest(manifest_path: str, manifest: Dict):
    temp_path = manifest_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, manifest_path)


def _checkpoint_file_name(file_path: str) -> str:
    return f"{os.path.basename(file_path)}.npz"


def _load_checkpoint(checkpoint_path: str):
    data = np.load(checkpoint_path, allow_pickle=False)
    embeddings = np.asarray(data["embeddings"], dtype="float32")
    records = json.loads(str(data["records_json"]))
    return embeddings, records


def run_embedding(input_path: str, output_dir: str):
    """Encode a single JSON file, or every JSON file in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    records = load_structured_documents(input_path)
    if not records:
        raise ValueError(f"No valid text block found in {input_path}")

    texts = [r["text"] for r in records]
    print(f"[embed] encoding {len(texts)} text blocks through the API ({EMBEDDING_MODEL_API}) ...")
    embeddings = embed_via_api(texts)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[embed] done, dim={embeddings.shape[1]}, saved to {output_dir}")


def run_embedding_batch(input_dir: str, output_dir: str):
    """Encode JSON files in batch with one checkpoint per file so an interrupted run can resume."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    json_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".json")
        and os.path.isfile(os.path.join(input_dir, f))
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON files in directory: {input_dir}")

    checkpoint_dir, manifest_path = _checkpoint_paths(output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    manifest_files = manifest.setdefault("files", {})
    completed = {}
    failed = []

    print(f"[batch] found {len(json_files)} JSON files")
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        signature = _file_signature(file_path)
        checkpoint_path = os.path.join(checkpoint_dir, _checkpoint_file_name(file_path))
        entry = manifest_files.get(file_name, {})

        if (
            entry.get("status") == "completed"
            and entry.get("signature") == signature
            and entry.get("checkpoint") == os.path.basename(checkpoint_path)
            and os.path.exists(checkpoint_path)
        ):
            try:
                file_embeddings, file_records = _load_checkpoint(checkpoint_path)
                if file_embeddings.shape[0] == len(file_records):
                    completed[file_name] = (file_embeddings, file_records)
                    print(f"[checkpoint] skipping completed file: {file_name}")
                    continue
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                print(f"[checkpoint] cannot read checkpoint, reprocessing {file_name}: {exc}")

        print(f"[batch] processing file: {file_name}")
        try:
            file_records = load_structured_documents(file_path)
            if not file_records:
                print(f"[batch] skipping empty file: {file_name}")
                manifest_files[file_name] = {
                    "status": "completed",
                    "signature": signature,
                    "checkpoint": None,
                    "records": 0,
                }
                _save_manifest(manifest_path, manifest)
                continue

            file_embeddings = embed_via_api([r["text"] for r in file_records])
            if file_embeddings.shape[0] != len(file_records):
                raise RuntimeError(f"Vector count and record count differ for {file_name}")

            temp_path = checkpoint_path + ".tmp"
            with open(temp_path, "wb") as temp_file:
                np.savez_compressed(
                    temp_file,
                    embeddings=file_embeddings,
                    records_json=np.array(json.dumps(file_records, ensure_ascii=False)),
                )
            os.replace(temp_path, checkpoint_path)
            manifest_files[file_name] = {
                "status": "completed",
                "signature": signature,
                "checkpoint": os.path.basename(checkpoint_path),
                "records": len(file_records),
            }
            _save_manifest(manifest_path, manifest)
            completed[file_name] = (file_embeddings, file_records)
            print(f"[checkpoint] saved: {file_name}")
        except Exception as exc:
            failed.append(file_name)
            manifest_files[file_name] = {
                "status": "failed",
                "signature": signature,
                "checkpoint": None,
                "error": str(exc),
            }
            _save_manifest(manifest_path, manifest)
            print(f"[batch] file failed, continuing with the next one: {file_name} -> {exc}")

        if os.path.exists(checkpoint_path + ".tmp"):
            os.remove(checkpoint_path + ".tmp")

    all_embeddings = []
    all_records = []
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        if file_name not in completed:
            continue
        file_embeddings, file_records = completed[file_name]
        all_embeddings.append(file_embeddings)
        all_records.extend(file_records)

    if not all_embeddings:
        raise ValueError(f"No valid text block found in {input_dir}; failed files: {failed}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[batch] merge done, files={len(completed)}, text blocks={len(all_records)}, dim={embeddings.shape[1]}")
    if failed:
        print(f"[batch] {len(failed)} file(s) failed and will be retried on the next run: {', '.join(failed)}")


def build_faiss_index(embed_dir: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    embeddings = np.load(os.path.join(embed_dir, "embeddings.npy"))
    dim = embeddings.shape[1]

    print(f"[dense-index] building the HNSW index, dim={dim}, M={HNSW_M}")
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "faiss_hnsw.index"))
    import shutil
    shutil.copy(
        os.path.join(embed_dir, "records.jsonl"),
        os.path.join(index_dir, "records.jsonl"),
    )
    print(f"[dense-index] done, saved to {index_dir}")


def build_bm25_index(embed_dir: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    records = _load_records(os.path.join(embed_dir, "records.jsonl"))
    texts = [r["text"] for r in records]

    print(f"[sparse-index] building the BM25 index over {len(texts)} entries")
    tokenized = [tokenize_for_bm25(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "records": records}, f)
    print(f"[sparse-index] done, saved to {index_dir}")


def main_embedding():
    print("=" * 60)
    print("Stage 1: generate embeddings through the API")
    print("=" * 60)
    if INPUT_MODE == "single":
        run_embedding(INPUT_PATH, EMBED_DIR)
    elif INPUT_MODE == "folder":
        run_embedding_batch(INPUT_DIR, EMBED_DIR)
    else:
        raise ValueError(f"INPUT_MODE must be 'single' or 'folder', got: {INPUT_MODE}")
    print("\n" + "=" * 60)


def main_indexs():
    print("Stage 2: build the indexes")
    print("=" * 60)
    build_faiss_index(EMBED_DIR, DENSE_DIR)
    build_bm25_index(EMBED_DIR, SPARSE_DIR)
    print("\n" + "=" * 60)



if __name__ == "__main__":
    main_embedding()
