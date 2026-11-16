import sys
from pathlib import Path

# Add the repository root (two levels up from this file) to the import path.
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, List, Dict, Any, TypeVar

import numpy as np

from steelpcr_config import env, env_int

# ===== 1. LightRAG core =====
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# ===== 2. Reranking (local BGE cross-encoder) =====
from sentence_transformers import CrossEncoder

# =============================================
# ===== Configuration =====
# =============================================

# ---- Embedding API ----
EMBED_API_URL = env("STEELPCR_EMBED_API_BASE", "your api url")
EMBED_API_KEY = env("STEELPCR_EMBED_API_KEY", "your api key")
EMBED_MODEL = env("STEELPCR_LIGHTRAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
EMBED_DIM = env_int("STEELPCR_LIGHTRAG_EMBED_DIM", 4096)

# ---- LLM API ----
LLM_API_URL = env("STEELPCR_LIGHTRAG_LLM_API_BASE", "https://api.deepseek.com")
LLM_API_KEY = env("STEELPCR_LIGHTRAG_LLM_API_KEY", "your api key")
LLM_MODEL = env("STEELPCR_LIGHTRAG_LLM_MODEL", "deepSeek-flash")

# ---- Local cross-encoder reranker ----
RERANK_MODEL_PATH = env("STEELPCR_RERANKER_MODEL_PATH", "./model/bge-reranker-v2-m3")

# ---- PostgreSQL + Qdrant storage backends ----
# These values apply to the current graphRAG.py process only. They are injected
# as POSTGRES_*/QDRANT_* environment variables before LightRAG is constructed, so neither Windows nor the PostgreSQL server configuration is modified.
STORAGE_WORKSPACE = env("STEELPCR_LIGHTRAG_WORKSPACE", "graphrag_cache")

POSTGRES_CONFIG = {
    "host": env("STEELPCR_PG_HOST", "127.0.0.1"),
    "port": env_int("STEELPCR_PG_PORT", 5432),
    "user": env("STEELPCR_PG_USER", "postgres"),
    "password": env("STEELPCR_PG_PASSWORD", ""),
    "database": env("STEELPCR_PG_DATABASE", "lightrag"),
    "workspace": STORAGE_WORKSPACE,
    "max_connections": env_int("STEELPCR_PG_MAX_CONNECTIONS", 50),
}

QDRANT_CONFIG = {
    "url": env("STEELPCR_QDRANT_URL", "http://127.0.0.1:6333"),
    "api_key": env("STEELPCR_QDRANT_API_KEY", ""),
    "workspace": STORAGE_WORKSPACE,
}

KV_STORAGE_BACKEND = "PGKVStorage"
VECTOR_STORAGE_BACKEND = "QdrantVectorDBStorage"
GRAPH_STORAGE_BACKEND = "PGTableGraphStorage"
DOC_STATUS_STORAGE_BACKEND = "PGDocStatusStorage"

def configure_storage_environment() -> None:
    """Inject the backend configuration above into the current Python process.

    LightRAG reads its storage configuration from ``os.environ`` at construction
    time; the assignments below never touch persistent Windows variables, the PostgreSQL server configuration, or other projects.
    """
    for key in ("user", "password", "database"):
        value = POSTGRES_CONFIG.get(key)
        if value is None or not str(value).strip():
            raise ValueError(
                f"PostgreSQL setting '{key}' is missing; set the corresponding STEELPCR_PG_* environment variable"
            )

    for key in ("host", "port", "workspace"):
        value = POSTGRES_CONFIG.get(key)
        if value is None or not str(value).strip():
            raise ValueError(
                f"PostgreSQL setting '{key}' is missing; set the corresponding STEELPCR_PG_* environment variable"
            )

    for key in ("url", "workspace"):
        value = QDRANT_CONFIG.get(key)
        if value is None or not str(value).strip():
            raise ValueError(
                f"Qdrant setting '{key}' is missing; set the corresponding STEELPCR_QDRANT_* environment variable"
            )

    workspace = str(STORAGE_WORKSPACE).strip()
    if not workspace or workspace in {".", ".."} or "/" in workspace or "\\" in workspace:
        raise ValueError(
            "STORAGE_WORKSPACE must be a single path segment and may not contain '/', '\\', '.' or '..'"
        )
    if str(POSTGRES_CONFIG["workspace"]).strip() != workspace:
        raise ValueError("POSTGRES_CONFIG['workspace'] must match STORAGE_WORKSPACE")
    if str(QDRANT_CONFIG["workspace"]).strip() != workspace:
        raise ValueError("QDRANT_CONFIG['workspace'] must match STORAGE_WORKSPACE")

    os.environ["POSTGRES_HOST"] = str(POSTGRES_CONFIG["host"])
    os.environ["POSTGRES_PORT"] = str(POSTGRES_CONFIG["port"])
    os.environ["POSTGRES_USER"] = str(POSTGRES_CONFIG["user"])
    os.environ["POSTGRES_PASSWORD"] = str(POSTGRES_CONFIG["password"])
    os.environ["POSTGRES_DATABASE"] = str(POSTGRES_CONFIG["database"])
    os.environ["POSTGRES_WORKSPACE"] = workspace
    if POSTGRES_CONFIG.get("max_connections") is not None:
        os.environ["POSTGRES_MAX_CONNECTIONS"] = str(
            POSTGRES_CONFIG["max_connections"]
        )

    os.environ["QDRANT_URL"] = str(QDRANT_CONFIG["url"])
    os.environ["QDRANT_WORKSPACE"] = workspace
    qdrant_api_key = str(QDRANT_CONFIG.get("api_key") or "").strip()
    if qdrant_api_key:
        os.environ["QDRANT_API_KEY"] = qdrant_api_key
    else:
        # An empty value disables authentication on a local Qdrant instance and avoids inheriting a stale key from another project.
        os.environ.pop("QDRANT_API_KEY", None)

    # The LightRAG instance uses the same workspace so the PostgreSQL and Qdrant partitions stay aligned.
    os.environ["WORKSPACE"] = workspace


# Contains no password or API key; lets the resume manifest detect a backend or model switch.
STORAGE_SIGNATURE = hashlib.sha256(
    json.dumps(
        {
            "kv_storage": KV_STORAGE_BACKEND,
            "vector_storage": VECTOR_STORAGE_BACKEND,
            "graph_storage": GRAPH_STORAGE_BACKEND,
            "doc_status_storage": DOC_STATUS_STORAGE_BACKEND,
            "postgres_host": POSTGRES_CONFIG["host"],
            "postgres_port": POSTGRES_CONFIG["port"],
            "postgres_database": POSTGRES_CONFIG["database"],
            "postgres_workspace": POSTGRES_CONFIG["workspace"],
            "qdrant_url": QDRANT_CONFIG["url"],
            "qdrant_workspace": QDRANT_CONFIG["workspace"],
            "embedding_model": EMBED_MODEL,
            "embedding_dim": EMBED_DIM,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

# ---- Runtime and file handling ----
# "preprocess": build or update the index; "query": retrieval and answer generation only;
# "full": preprocess (embedding + knowledge extraction) and then run the query stage.
RUN_MODE = "query"
# "single": process INPUT_JSON_PATH; "batch": process every JSON file under INPUT_JSON_DIR.
INPUT_MODE = env("STEELPCR_LIGHTRAG_INPUT_MODE", "single")
INPUT_JSON_PATH = env("STEELPCR_INPUT_PATH", "")
INPUT_JSON_DIR = env("STEELPCR_INPUT_DIR", str(Path(__file__).resolve().parents[1] / "QA_generation" / "data"))

# ---- Resume handling and API request settings ----
WORKING_DIR = "./cache/lightRAG"
MANIFEST_FILE_NAME = "file_ingest_manifest.json"
API_TIMEOUT_SECONDS = 200.0
MAX_API_ATTEMPTS = 4
RETRY_INTERVAL_SECONDS = 6.0

# ---- Query settings (used when RUN_MODE = "query") ----
QUERY_TEXT = "热压延工序能耗的计算公式是什么？"

# =============================================
# ===== 3. Embedding function =====
# =============================================

T = TypeVar("T")


async def call_with_fixed_retry(
    operation: Callable[[], Awaitable[T]],
    operation_name: str,
) -> T:
    """Retry transient API failures at a fixed interval, then hand the exception to the file-level resume logic."""
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt == MAX_API_ATTEMPTS:
                raise
            print(
                f"⚠️ {operation_name} failed on attempt {attempt}/{MAX_API_ATTEMPTS}: {exc}; "
                f"retrying in {RETRY_INTERVAL_SECONDS} s"
            )
            await asyncio.sleep(RETRY_INTERVAL_SECONDS)


# =============================================
# ===== Shared OpenAI clients (one connection pool each, avoiding repeated handshakes) =====
# =============================================
_GLOBAL_EMBED_CLIENT = None
_GLOBAL_LLM_CLIENT = None


def get_embed_client():
    global _GLOBAL_EMBED_CLIENT
    if _GLOBAL_EMBED_CLIENT is None:
        import openai
        import httpx
        # No cap on concurrent connections (max_connections=None, max_keepalive_connections=None)
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        http_client = httpx.AsyncClient(limits=limits)
        _GLOBAL_EMBED_CLIENT = openai.AsyncOpenAI(
            base_url=EMBED_API_URL,
            api_key=EMBED_API_KEY,
            timeout=API_TIMEOUT_SECONDS,
            max_retries=0,
            http_client=http_client,
        )
    return _GLOBAL_EMBED_CLIENT


def get_llm_client():
    global _GLOBAL_LLM_CLIENT
    if _GLOBAL_LLM_CLIENT is None:
        import openai
        import httpx
        # No cap on concurrent connections (max_connections=None, max_keepalive_connections=None)
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        http_client = httpx.AsyncClient(limits=limits)
        _GLOBAL_LLM_CLIENT = openai.AsyncOpenAI(
            base_url=LLM_API_URL,
            api_key=LLM_API_KEY,
            timeout=API_TIMEOUT_SECONDS,
            max_retries=0,
            http_client=http_client,
        )
    return _GLOBAL_LLM_CLIENT


async def embedding_func(texts: List[str]) -> np.ndarray:
    async def request_embedding():
        client = get_embed_client()
        return await client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            dimensions=EMBED_DIM,
        )

    response = await call_with_fixed_retry(request_embedding, "Embedding API")
    return np.array([item.embedding for item in response.data], dtype=np.float32)

embedding_component = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    max_token_size=32000,
    func=embedding_func,
    model_name=EMBED_MODEL,
)

# =============================================
# ===== 2. LLM callable (argument firewall; stops enable_cot from reaching the API) =====
# =============================================

# Arguments accepted by the OpenAI API
_OPENAI_API_PARAMS = {
    "temperature", "top_p", "n", "stream", "stop", "max_tokens",
    "presence_penalty", "frequency_penalty", "logit_bias", "user",
    "response_format", "seed", "tools", "tool_choice", "parallel_tool_calls",
}

async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    keyword_extraction: bool = False,
    hashing_kv=None,
    enable_cot: bool = False,
    **kwargs
) -> str:
    """
    Call the LLM API.
    LightRAG passes its own internal parameters; they are accepted explicitly and
    consumed here, and kwargs are filtered through a whitelist so nothing unknown reaches the API.
    """
    import openai

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Whitelist filter: keep only the arguments the OpenAI API understands
    api_kwargs = {k: v for k, v in kwargs.items() if k in _OPENAI_API_PARAMS}

    async def request_completion() -> str:
        client = get_llm_client()
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            **api_kwargs,
        )

        if not response.choices:
            raise RuntimeError("LLM returned no choices")

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "LLM returned empty content "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r}, "
                f"refusal={getattr(message, 'refusal', None)!r})"
            )
        return content

    return await call_with_fixed_retry(request_completion, "LLM API")

# =============================================
# ===== 3. Reranker =====
# =============================================

_reranker_model = None

def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        print(f"Loading reranker: {RERANK_MODEL_PATH} ...")
        _reranker_model = CrossEncoder(RERANK_MODEL_PATH, device="cpu")
        print("done")
    return _reranker_model

async def rerank_func(query: str, documents: List[str], top_n: int = None) -> List[Dict[str, Any]]:
    model = get_reranker()
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if top_n:
        scored_indices = scored_indices[:top_n]
    return [{"index": idx, "relevance_score": float(scores[idx])} for idx in scored_indices]

# =============================================
# ===== 4. Metadata flattening =====
# =============================================

def flatten_metadata(metadata: dict, prefix: str = "doc_") -> dict:
    result = {}
    for key, value in metadata.items():
        new_key = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(flatten_metadata(value, prefix=new_key + "_"))
        else:
            result[new_key] = value
    return result

# =============================================
# ===== 5. Structured-data parsing (raw chunks only) =====
# =============================================

def extract_document_metadata(data: Dict[str, Any], json_path: str) -> Dict[str, Any]:
    """Extract document-level metadata for the resume manifest, independent of whether the document yields any text chunk."""
    doc_meta = flatten_metadata(data.get("metadata", {}), prefix="doc_")
    doc_meta["doc_source_file"] = Path(json_path).name
    if "uuid" in data:
        doc_meta["doc_uuid"] = data["uuid"]
    return doc_meta


def parse_json_to_chunks(json_path: str) -> tuple[str, List[Dict], set, Dict[str, Any]]:
    """Parse one JSON document and return only the unmerged raw chunks plus document-level metadata."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_meta = extract_document_metadata(data, json_path)

    raw_chunks = []
    order_idx = 0

    for item in data.get("body_text", []):
        text = item.get("text", "").strip()
        if not text:
            continue
        chunk = {
            "content": text,
            "block_id": item.get("id", f"p_{order_idx}"),
            "block_type": item.get("type", "text"),
            "head_section": item.get("head_section", ""),
            "head_paragraph": item.get("head_paragraph", ""),
            "refs": item.get("refs", []),
            "chunk_order_index": order_idx,
        }
        chunk.update(doc_meta)
        raw_chunks.append(chunk)
        order_idx += 1

    for item in data.get("figures_and_tables", []):
        text = item.get("text", "").strip()
        if not text:
            continue
        chunk = {
            "content": text,
            "block_id": item.get("id", f"ft_{order_idx}"),
            "block_type": item.get("type", "figure"),
            "head_section": item.get("head_section", ""),
            "title": item.get("title", ""),
            "chunk_order_index": order_idx,
        }
        chunk.update(doc_meta)
        raw_chunks.append(chunk)
        order_idx += 1

    all_field_names = set()
    for chunk in raw_chunks:
        all_field_names.update(chunk.keys())

    full_text = "\n\n".join(chunk["content"] for chunk in raw_chunks)

    return full_text, raw_chunks, all_field_names, doc_meta


# =============================================
# ===== 6. File-level preprocessing resume =====
# =============================================

def get_input_json_paths() -> List[Path]:
    """Return the JSON files to process; batch mode reads only the direct children of the directory."""
    if INPUT_MODE == "single":
        json_path = Path(INPUT_JSON_PATH)
        if not json_path.is_file():
            raise FileNotFoundError(f"Single-file input does not exist: {INPUT_JSON_PATH}")
        return [json_path]

    if INPUT_MODE == "batch":
        input_dir = Path(INPUT_JSON_DIR)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Batch input directory does not exist: {INPUT_JSON_DIR}")
        json_paths = sorted(input_dir.glob("*.json"))
        if not json_paths:
            raise FileNotFoundError(f"No JSON files found in: {INPUT_JSON_DIR}")
        return json_paths

    raise ValueError("INPUT_MODE must be 'single' or 'batch'")


def manifest_path() -> Path:
    return Path(WORKING_DIR) / MANIFEST_FILE_NAME


def load_ingest_manifest() -> Dict[str, Any]:
    """Load the resume manifest; rebuild the index when the backend or embedding configuration changed."""
    path = manifest_path()
    empty_manifest = {
        "version": 2,
        "storage_signature": STORAGE_SIGNATURE,
        "files": {},
    }
    if not path.exists():
        return empty_manifest
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise ValueError("Resume manifest has an invalid format")
        if manifest.get("storage_signature") != STORAGE_SIGNATURE:
            print("ℹ️ Storage backend or model configuration changed; the index will be rebuilt")
            return empty_manifest
        manifest["version"] = 2
        return manifest
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"⚠️ Cannot read the resume manifest, recreating it: {exc}")
        return empty_manifest


def save_ingest_manifest(manifest: Dict[str, Any]) -> None:
    """Replace the resume manifest atomically, retrying and falling back when Windows file locks collide."""
    manifest["version"] = 2
    manifest["storage_signature"] = STORAGE_SIGNATURE
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # Extra retries and a fallback write path for Windows file-lock contention
    last_exc = None
    for attempt in range(5):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.2 * (attempt + 1))
            
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    except Exception:
        if last_exc:
            raise last_exc


def get_file_signature(json_path: Path) -> Dict[str, Any]:
    """Compute the per-file resume signature; the content hash prevents false skips caused by relying on size or mtime alone."""
    digest = hashlib.sha256()
    with open(json_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    stat = json_path.stat()
    return {
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_completed_entry(entry: Any, signature: Dict[str, Any]) -> bool:
    return isinstance(entry, dict) and entry.get("status") == "completed" and entry.get("signature") == signature


def create_rag() -> LightRAG:
    configure_storage_environment()
    return LightRAG(
        working_dir=WORKING_DIR,
        kv_storage=KV_STORAGE_BACKEND,
        vector_storage=VECTOR_STORAGE_BACKEND,
        graph_storage=GRAPH_STORAGE_BACKEND,
        doc_status_storage=DOC_STATUS_STORAGE_BACKEND,
        workspace=STORAGE_WORKSPACE,
        embedding_func=embedding_component,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        rerank_model_func=rerank_func,
        # Keep the existing concurrency setting unchanged.
        llm_model_max_async=6,
        entity_extract_max_gleaning=0,  # disable the second-round gleaning pass
        # Give the Embedding worker enough time for the configured API timeout.
        embedding_batch_num=64,
        embedding_func_max_async=3, 
        default_embedding_timeout=360,
        top_k=40,
        chunk_top_k=80,
        max_total_tokens=1000000,
        entity_extraction_use_json=True,
        max_source_ids_per_entity=50000,
        max_source_ids_per_relation=50000,
        source_ids_limit_method="KEEP",
    )


async def check_doc_status_in_storage(
    rag: LightRAG,
    doc_id: str | None,
    file_path: str,
) -> str | None:
    """Check the document's real processing state in the database.
    Returns 'processed', 'failed', 'processing', 'pending' or None (not found).
    """
    try:
        # 1. Query by doc_id when it is known
        if doc_id:
            record = await rag.doc_status.get_by_id(doc_id)
            if record and isinstance(record, dict):
                return record.get("status")

        # 2. Otherwise look the document up by file_path in the doc_status table
        if hasattr(rag.doc_status, "db") and rag.doc_status.db is not None:
            sql = 'SELECT status FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND (file_path=$2 OR file_path LIKE $3) ORDER BY updated_at DESC LIMIT 1'
            base_name = Path(file_path).name
            res = await rag.doc_status.db.query(sql, [rag.doc_status.workspace, file_path, f"%{base_name}%"], True)
            if res and len(res) > 0:
                return res[0].get("status")
    except Exception as e:
        print(f"⚠️ Failed to query the document status: {e}")
    return None


async def preprocess_documents() -> None:
    """Index one JSON file at a time; skip only when the database confirms 'processed', and retry previously failed files."""
    rag = create_rag()
    await rag.initialize_storages()
    print("✅ Preprocessing storage initialised")

    manifest = load_ingest_manifest()
    files_state = manifest.setdefault("files", {})
    json_paths = get_input_json_paths()
    completed = failed = skipped = empty = 0

    for json_path in json_paths:
        source_key = str(json_path.resolve())
        signature = get_file_signature(json_path)
        doc_meta: Dict[str, Any] | None = None
        previous_entry = files_state.get(source_key)

        # Read doc_uuid from the JSON first so the database state can be checked precisely
        doc_uuid = None
        if previous_entry and isinstance(previous_entry.get("document_metadata"), dict):
            doc_uuid = previous_entry["document_metadata"].get("doc_uuid")

        # Only a 'processed' database state counts as genuinely finished
        db_status = await check_doc_status_in_storage(rag, doc_uuid, str(json_path))

        # Skip only when the manifest says completed, the file signature is unchanged and the database confirms 'processed'
        if is_completed_entry(previous_entry, signature) and db_status == "processed":
            print(f"⏭️ Skipping completed file: {json_path.name}")
            skipped += 1
            continue

        if db_status == "failed":
            print(f"🔄 This file is marked 'failed' in the database; clearing the failed and duplicate-claim records so it can be retried: {json_path.name}")
            if hasattr(rag.doc_status, "db") and rag.doc_status.db is not None:
                base_name = json_path.name
                # Clear the failed record and the derived "dup-" duplicate-guard records so LightRAG's filename deduplication no longer blocks this file
                clear_sql = """
                    DELETE FROM LIGHTRAG_DOC_STATUS 
                    WHERE workspace = $1 
                      AND (file_path = $2 OR file_path LIKE $3)
                      AND (status = 'failed' OR id LIKE 'dup-%')
                """
                await rag.doc_status.db.query(clear_sql, [rag.doc_status.workspace, str(json_path), f"%{base_name}%"], False)
        elif db_status in ("processing", "pending"):
            print(f"🔄 File is in an interrupted state ({db_status}); re-running preprocessing: {json_path.name}")

        try:
            full_text, chunks, all_field_names, doc_meta = parse_json_to_chunks(str(json_path))
            if not chunks:
                files_state[source_key] = {
                    "status": "completed",
                    "signature": signature,
                    "document_metadata": doc_meta,
                    "chunk_count": 0,
                    "inserted": False,
                    "updated_at": utc_now(),
                }
                save_ingest_manifest(manifest)
                print(f"ℹ️ Empty document: metadata recorded, embedding and knowledge extraction skipped: {json_path.name}")
                empty += 1
                continue

            def add_chunk_token_counts() -> None:
                for chunk in chunks:
                    if not isinstance(chunk.get("tokens"), int):
                        chunk["tokens"] = len(rag.tokenizer.encode(chunk["content"]))

            # PGKVStorage persists tokens as a required fixed column. Compute it
            # for the caller-supplied chunks before LightRAG builds its storage rows.
            await asyncio.to_thread(add_chunk_token_counts)
            all_field_names.add("tokens")
            all_field_names.add("content")
            rag.chunks_vdb.meta_fields.update(all_field_names)

            def custom_chunking_func(content, *args, current_chunks=chunks, **kwargs):
                return current_chunks

            rag.chunking_func = custom_chunking_func
            print(f"📥 Preprocessing: {json_path.name}, {len(chunks)} chunks")
            await rag.ainsert(
                full_text,
                ids=doc_meta.get("doc_uuid") or None,
                file_paths=str(json_path),
            )

            # Re-check the real database state once ainsert returns
            current_uuid = doc_meta.get("doc_uuid") if doc_meta else None
            final_status = await check_doc_status_in_storage(rag, current_uuid, str(json_path))
            if final_status == "failed":
                raise RuntimeError("LightRAG did not finish successfully (database state is 'failed')")

            files_state[source_key] = {
                "status": "completed",
                "signature": signature,
                "document_metadata": doc_meta,
                "chunk_count": len(chunks),
                "inserted": True,
                "updated_at": utc_now(),
            }
            save_ingest_manifest(manifest)
            completed += 1
            print(f"✅ Preprocessing finished: {json_path.name}")
        except Exception as exc:
            files_state[source_key] = {
                "status": "failed",
                "signature": signature,
                "document_metadata": doc_meta,
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": utc_now(),
            }
            save_ingest_manifest(manifest)
            failed += 1
            print(f"❌ Preprocessing failed; it will be retried on the next run: {json_path.name}\n   {exc}")

    print(
        f"📊 Preprocessing finished: {completed} succeeded, {skipped} skipped, "
        f"{empty} empty, {failed} failed"
    )
    await rag.finalize_storages()


async def run_query() -> None:
    """Load the existing index and run retrieval, reranking and answer generation only."""
    rag = create_rag()
    await rag.initialize_storages()
    print("✅ Query storage initialised")

    # Retrieval settings of the LightRAG-mix arm: internal reranking off, the candidate pool is
    # top_k = chunk_top_k = 50 (the external reranker then selects the evidence for the generator).
    param = QueryParam(
        mode="mix",
        enable_rerank=False,
        top_k=50,
        chunk_top_k=50,
        response_type="Multiple Paragraphs",
    )

    print(f"\n🔍 Query: {QUERY_TEXT}")
    result = await rag.aquery(QUERY_TEXT, param=param)
    print("\n🤖 Answer:")
    print(result)
    await rag.finalize_storages()


async def main():
    if RUN_MODE == "preprocess":
        await preprocess_documents()
    elif RUN_MODE == "query":
        await run_query()
    elif RUN_MODE == "full":
        await preprocess_documents()
        await run_query()
    else:
        raise ValueError("RUN_MODE must be 'preprocess', 'query' or 'full'")

if __name__ == "__main__":
    asyncio.run(main())