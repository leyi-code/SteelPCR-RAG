# Paper test harness: the four case-study runs

This directory holds the executable code of the four case-study runs (one per case). The retrieval and
generation framework they share lives in `../rag_core/`.

---

## 1. What maps to what

| Case | Row in the results table | Code in this directory | Configuration in one line |
|---|---|---|---|
| Case 1, biomass char blended into sintering | test A, single-round RAG | `case1.py` | ask the original question, one retrieval round, one generation |
| Case 2, biomass char injected into the blast furnace | test C, multi-round with term optimisation and context expansion | `case2.py` | ten term/mechanism retrieval intents, evidence pool merged round-robin |
| Case 3, hydrogen-based shaft furnace | test A, single-round RAG | `case3.py` | ask the original question, one retrieval round, one generation |
| Case 4, steel-slag-catalysed sludge pyrolysis | test C, multi-round with full evidence and the refined prompt | `case4.py` | model-driven decomposition, all evidence kept unfiltered, quantitative-completeness prompt |

Cases 2 and 4 are both classified as multi-round in that table: they share one architecture
(decompose -> retrieve per path -> merge evidence -> integrated generation) and differ in the
evidence-pool strategy and in the generation prompt. Case 2 is covered by a round-robin merged pool,
whereas case 4 hands every piece of evidence to the model unchanged and adds the
quantitative-completeness requirement.

---

## 2. Directory layout

```
probe_case/
├── README.md                          this file
├── cases.py                           case definitions, condition combinations, scoring anchors
├── rerun/                             output directory for a run (created by the first run)
├── case1.py
├── case2.py
├── case3.py
└── case4.py
```

The framework lives in `../rag_core/`:

| Module | Content |
|---|---|
| `rag_core/pipeline.py` | FAISS-HNSW + Qwen3-Embedding-8B + Qwen3-Reranker-8B (API) + `enhance_context` + `build_rag_context` + `SYSTEM_PROMPT` |
| `rag_core/multirag.py` | task decomposition + the multi-round architecture + `DECOMPOSE_SYSTEM` / `DECOMPOSE_V2` / `SUBANSWER_SYSTEM` / `INTEGRATE_SYSTEM` |
| `rag_core/helpers.py` | record trimming / deduplication / sub-question parsing / logging |

`multirag.py` is the orchestration layer that every case script calls:

| Function | Role |
|---|---|
| `decompose()` | split the compound question into single-intent retrieval paths (`DECOMPOSE_SYSTEM` or `DECOMPOSE_V2`) |
| `retrieve_paths()` | run one full pipeline round per path |
| `answer_subpoints()` | optional short answer per path, from that path's own evidence |
| `merge_round_robin()` / `merge_all()` | the two evidence-pool strategies |
| `build_prompt_with_table()` / `build_prompt_with_list()` | the two integration prompt layouts |
| `run_single()` / `run_multi()` | the two arms the case scripts invoke |

The retrieval and context-organisation logic is the paper's. `cases.OUT_ROOT` points at `rerun/`.

---

## 3. Full parameters of the four configurations

All four share the same retrieval chain and differ only in the rows below.

| Item | Case 1 / case 3 (single round) | Case 2 (multi-round, term optimisation and context expansion) | Case 4 (multi-round, full evidence and refined prompt) |
|---|---|---|---|
| Dense initial search | FAISS-HNSW top 50, efSearch = 128 | same, once per path | same, once per path |
| Query expansion | bilingual terminology alignment, `expand_query` | same | same |
| Reranking | Qwen3-Reranker-8B over HTTP, keeps **7** chunks | **5** chunks per path | **5** chunks per path |
| Context enhancement | `expand_window=1` plus paragraph merging and table fencing | same | same |
| Retrieval intents | none (the single original question) | **10** (hand-written property and mechanism terms) | **7-12**, decomposed automatically by the model with `DECOMPOSE_V2` |
| Intermediate step | none | one short answer per path (2500/8000 chars, at most 120 chars each) | none (no per-chunk extraction) |
| Evidence pool | the reranked top 7 go straight into the context | chunks taken **round-robin** across paths, pool capped at **8** | every chunk from every path, **deduplicated and kept as-is**, unsorted, untruncated, unfiltered |
| Context budget | 3000 per chunk / 12000 in total | 8000 per chunk / 30000 in total | **no cap** |
| Evidence order | U-shaped reordering | same | same |
| Generation prompt | `SYSTEM_PROMPT` (graded-evidence version) | `INTEGRATE_SYSTEM` (graded evidence, the intermediate draft may be overturned) | `SYSTEM_PROMPT` plus `NUM_RULE` and `COND_RULE` |
| Generation model | deepSeek-flash, temperature = 0.0, max_tokens = 32000 | same | same |

Notes:

- **`SYSTEM_PROMPT` (graded-evidence version)** lets the model decide whether each piece of evidence is
  on topic, trustworthy and applicable under which conditions, and lets it reason like an engineer.
  Wherever the evidence already covers the point it must stay with the evidence (no rewriting values,
  units, ratios or temperatures, and no carrying a value across operating conditions); anything the
  evidence does not cover must be marked as extrapolation; conflicts must be exposed and gaps stated as
  "not reported in the material". The answer follows a fixed five-section structure: core conclusion,
  point-by-point evidence anchoring, divergence and trade-offs, gaps and extrapolation, and a reference
  source list that consolidates the cited evidence by source document.
- **`NUM_RULE` (no quantitative conclusion may be omitted)** requires every quantitative conclusion in
  the material to appear in the answer with its source, and forbids dropping one because it "looks only
  loosely related to the question".
- **`COND_RULE` (every value must state its condition)** requires each value to be accompanied by the
  conditions it applies to (feedstock, ratio, temperature, atmosphere, measurement method); when the
  source gives no condition, the answer must say so.
- **`DECOMPOSE_V2`** changes exactly one rule of `DECOMPOSE_SYSTEM`: one dedicated sub-question must ask
  about the paper's own experimental conditions and apparatus (asking only how the conditions were set,
  never presupposing a value), and the sub-question count is relaxed from 6-10 to 7-12. The retrieval
  terms are still generated by the model and contain **no exact values and no copied sentences**.

---

## 4. How to run

```bat
rem the interpreter needs faiss / numpy / requests / openai
set PY=python

%PY% "probe_case\case1.py"
%PY% "probe_case\case2.py"
%PY% "probe_case\case3.py"
%PY% "probe_case\case4.py"
```

Artefacts are written to `probe_case/rerun/` (a JSON file plus a log of the same name). Before running,
copy `.env.example` from the repository root to `.env` and fill in `STEELPCR_EMBED_API_KEY` (embedding
and reranking) and `STEELPCR_GEN_STRONG_API_KEY` (generation and query expansion).

---

## 5. Dependencies and external resources

| Resource | Path / value |
|---|---|
| Dense index | `$STEELPCR_DENSE_INDEX_DIR/faiss_hnsw.index` (mmap read-only; defaults to `rag_core/cache/dense`) |
| Record store | `records.jsonl` in the same directory (read lazily by byte offset, never loaded whole) |
| Knowledge-base size | 1,954 documents / 207,866 text chunks |
| Embedding | `Qwen3-Embedding-8B` (4096-d) at `$STEELPCR_EMBED_API_BASE` |
| Reranker | `Qwen/Qwen3-Reranker-8B` at `$STEELPCR_QWEN3_RERANK_URL` |
| Generation | `deepSeek-flash` at `$STEELPCR_GEN_STRONG_API_BASE` |

---

## 6. Reproduction notes

1. Each condition was run once (temperature = 0.0) with no repeat-and-average. LLM sampling and the
   online reranking API can be non-deterministic under high concurrency or server-side batching, so a
   rerun is not guaranteed to be word-for-word identical, although the configuration and the chain are.
2. Every case in `cases.py` carries two fields, `kb_notes` and `rubric`. The criteria were predefined by
   domain experts along the four dimensions Anchor / Condition / Gap / Forbid, and are scored item by item
   against the generated answer by the independent LLM judge described in SI Text S16. The four case
   scripts only generate answers and never pass either field to a model: searching every `chat(` call site
   confirms that neither field is referenced.
3. `rag_core/multirag.py` asserts the `DECOMPOSE_SYSTEM` replacement anchor when it derives
   `DECOMPOSE_V2`, so if that prompt text is edited the import fails at start-up instead of silently
   falling back to the original decomposition rule.
