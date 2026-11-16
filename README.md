# SteelPCR-RAG

SteelPCR-RAG is a domain retrieval-augmented generation framework for decision support in
steel-industry pollution-and-carbon-reduction (carbon–pollutant synergy) problems. It answers
engineering questions about sintering, ironmaking, steelmaking, waste-gas and solid-waste treatment
with evidence that can be traced back to the literature: each conclusion is bound to the passage it
comes from, to the operating conditions under which it holds, and to the system boundary it applies to.

The framework is built around the way process-industry knowledge is actually written. Domain questions
are bilingual, so terminology is aligned before retrieval. Evidence is written as paragraphs that
carry their own conditions, tables and equations, so chunks are paragraph-level and tables and
formulas remain retrievable units. Answers must survive an engineering review, so generation grades
the evidence instead of trusting it, and reports what the material does not support.

---

## Framework at a glance

| Stage | Role | Component |
|---|---|---|
| Bilingual terminology expansion | Align a Chinese technical question with its international terminology before retrieval | LLM rewriter, `EXPAND_SYSTEM` |
| Dense retrieval | Recall candidates over the whole knowledge base | Qwen3-Embedding-8B (4,096-d) + FAISS-HNSW, top 50 |
| Evidence reranking | Rank the candidate pool by relevance to the question | Qwen3-Reranker-8B over HTTP, top 7 |
| Context organization and metadata annotation | Turn ranked chunks into an evidence unit that keeps argument chains and source attributes | adjacent-paragraph expansion, same-section merging, table fencing, equation pull-in, U-shaped ordering |
| Answer generation | Produce a quantitative, source-bound answer | DeepSeek-V4-Flash, deterministic decoding, evidence-graded prompt |
| Retrieval expansion (compound questions) | Cover questions whose evidence is dispersed across many constraints | task decomposition → per-path retrieval → evidence-pool merge → integrated generation |

The last stage is used by the case-validation experiment; the five stages above it are the single-chain
system evaluated in the main experiments.

---

## Knowledge base

The knowledge base is built from two source families.

| Family | Content | Parsing |
|---|---|---|
| Scientific literature | 1,822 papers screened from Elsevier, Springer and Nature (2000–2026, JCR Q1/Q2), filtered by title and abstract and then confirmed by domain experts | Grobid, on English PDFs with an extractable text layer |
| Industry documents | 145 reports, policy standards and professional books | PaddleOCR-VL 1.6 for image-based PDFs |

All documents are converted to text-only JSON with their metadata retained. Documents are segmented
into paragraph-level units, tables are converted to text with the original HTML kept as metadata, and
equations are converted to LaTeX and stored with context, so that a table or an equation can itself be
retrieved as evidence. The resulting knowledge base holds **1,954 documents and 207,866 paragraph
chunks**, with every chunk identified by a unique key that carries its source document, section and
block position.

| Artefact | Path | Built by |
|---|---|---|
| FAISS-HNSW index | `cache/dense/faiss_hnsw.index` | `rag_core/build_index.py` |
| Chunk store | `cache/dense/records.jsonl` | `rag_core/build_index.py` |
| BM25 index | `cache/sparse/bm25.pkl` | `rag_core/build_index.py` |
| Graph–vector stores (LightRAG-mix) | `cache/lightRAG/` | `rag_core/graphRAG.py` |

---

## Bilingual terminology expansion

A question is first rewritten into a bilingual retrieval string: the original question is kept and the
corresponding international academic terms, abbreviations, chemical formulae and process codes are
appended, for example `FOBF` for full-oxygen blast furnace, `COG` for coke oven gas and `DRM` for dry
reforming of methane. The model generates the expansion and it is cached, so a question is expanded
once and the same retrieval string is reused on every rerun. The expansion never answers the question:
it only produces the string that is embedded.

## Dense retrieval

Expanded queries are encoded with Qwen3-Embedding-8B (4,096 dimensions) and searched with a
FAISS-HNSW index (`M = 16`, `efConstruction = 200`, `efSearch = 128`), returning the top 50
candidates in a single pass over the index. No prompt-side filtering happens before reranking, so the
candidate pool stays auditable.

## Evidence reranking

The 50 candidates are reranked by Qwen3-Reranker-8B over HTTP and the top 7 are kept as the evidence
set (`context K = 7`). Experiments 1 and 2 keep five chunks (`context K = 5`), the setting reported
for those two experiments; the rerankers compared there are BGE-reranker-v2-m3, Cohere-Rank-4-Pro and
Qwen3-Reranker-8B.

## Context organization and metadata annotation

Ranked chunks are not pasted into the prompt as they are. Each chunk is expanded with its adjacent
paragraph, chunks from the same section are merged in their original order, internal gaps between
retrieved chunks are filled, and the table and equation blocks that belong to a retrieved passage are
pulled in. Every unit is annotated with its source document, section and block type
(`[paragraph]`, `[table]`, `[formula]`), which is what later allows an answer to cite
`[Evidence N]` and be traced back to a document. The assembled evidence is ordered head-to-tail so
that the most relevant material sits at the beginning and at the end of the prompt, and its size is
capped at 3,000 characters per chunk and 12,000 characters in total.

## Answer generation

The evidence units and the user question go to DeepSeek-V4-Flash with deterministic decoding and a
32,000-token ceiling. The prompt makes the model grade the evidence before it writes:

- where the material covers a point, the answer follows it exactly — values, units, ratios,
  temperatures and directions of a trend are restated as written and tied to the conditions under
  which they hold, never transferred to another operating condition;
- where the material does not cover a point, reasoning is allowed but must be labelled as
  extrapolation, and no value may be invented to make the answer look complete;
- directional conflicts and trade-offs between evidence items must be stated explicitly, and an
  evidence gap must be reported as "not provided in the material";
- each cited conclusion carries an `[Evidence N]` tag, and the answer closes with a reference-source
  list that consolidates the evidence actually used, one document per line.

## Retrieval expansion for compound questions

A single retrieval round assumes that the evidence for one question sits in one place. In compound
decision problems the evidence is dispersed across many constraints and operating conditions, so the
framework adds a multi-round layer:

1. **Decompose.** The model splits the compound question into single-intent retrieval paths, one
   variable dimension each, written as factual retrieval queries rather than as requests for synthesis;
   for the sludge-pyrolysis case the decomposition is also required to ask how the source paper set up
   its own experiments, so that the assumed conditions can be checked against the literature;
2. **Retrieve per path.** Every path runs the full chain above and, optionally, produces a short
   answer from its own evidence only;
3. **Merge the evidence pool.** Either one chunk per path is taken round-robin under a fixed pool
   size, or every chunk from every path is deduplicated and kept as-is;
4. **Integrate.** One generation call produces the final answer from the question, the path list and
   the merged pool. The per-path answers are intermediate material only: the evidence pool governs
   the final answer, and a per-path answer is overturned wherever it conflicts with the evidence.

Each of the four steps is a separate function, so a question that needs a different decomposition, pool
rule or prompt layout can be assembled by hand:

```python
from rag_core.pipeline import SteelPCRPipeline
from rag_core import multirag as M

pipe = SteelPCRPipeline(enable_expansion=True)
question = "your question"

subs, raw = M.decompose(question)          # retrieval intents; the raw model responses are kept
steps = M.retrieve_paths(pipe, subs)       # one full retrieval round per intent
M.answer_subpoints(steps)                  # optional: one short answer per path
merged, duplicates = M.merge_all(steps)    # or: merged = M.merge_round_robin(steps)
user = M.build_prompt_with_list(question, steps, M.build_rag_context(merged))
answer = M.integrate(user, M.INTEGRATE_SYSTEM)
```

---

## Experimental design

Four controlled experiments and one case-validation study accompany the framework. The RAG conditions
are generated by DeepSeek-V4-Flash, and Experiment 3 adds two closed-book native conditions
(DeepSeek-V4-Flash and the stronger DeepSeek-V4.1-Flash). Every answer is scored by GLM-5.3-Flash
acting as an independent cross-family judge.

| Experiment | Question | Design | Scale |
|---|---|---|---|
| 1 | Which retrieval paradigm? | Dense (FAISS-HNSW), Sparse (BM25), Hybrid (RRF, `k = 60`) and graph-based (LightRAG-mix, PostgreSQL + Qdrant) all reranked by one shared cross-encoder | 355 questions |
| 2 | Which reranker? | BGE-reranker-v2-m3, Cohere-Rank-4-Pro and Qwen3-Reranker-8B on a fixed top-50 pool, `context K = 5` | 355 questions |
| 3 | Does retrieval beat native models? | SteelPCR-RAG against two closed-book native conditions of the same model family | 355 questions |
| 4 | What does each component contribute? | Four single-component ablations of the full system (reranking, bilingual expansion, context expansion and merging, metadata injection), paired against the full system | 4 arms x 355 questions |
| Case validation | Can it support real decisions? | Four engineering scenarios (biomass-char sintering blend, blast-furnace biochar injection, hydrogen shaft furnace, steel-slag-catalysed sludge pyrolysis) scored against 27 expert-defined items in four dimensions: Anchor, Condition, Gap and Forbid | 27 items |

**Metrics.** Answer quality is measured with Answer Correctness, Fully Correct Rate
(`AC >= 0.75`) and graded correctness; evidential support with Faithfulness, Context Precision and
Context Recall; retrieval with `Recall@K`, `MRR@K` and `nDCG@K`; cost with P50 and P95 latency.
Answer-quality metrics come from RAGAS with an independent judge model, and component effects are
tested with the Wilcoxon signed-rank test, the exact McNemar test, Cliff's delta and Holm correction.

---

## Repository layout

```
SteelPCR-RAG/
├── rag_core/                 retrieval, reranking, context organization, generation
│   ├── general_RAG.py        single-chain RAG: retrieval, reranking, build_rag_context, generation
│   ├── graphRAG.py           LightRAG-mix stores (PostgreSQL + Qdrant) and its ingestion entry point
│   ├── pipeline.py           SteelPCR-RAG chain used by the case scripts
│   ├── multirag.py           task decomposition and the multi-round architecture
│   ├── helpers.py            record trimming, deduplication, sub-question parsing
│   └── build_index.py        FAISS-HNSW and BM25 index construction
├── evaluation/
│   ├── eval1.py              retrieval-paradigm comparison
│   ├── eval2.py              reranker comparison
│   ├── eval3.py              RAG versus native models
│   └── eval4.py              component ablation and paired statistics
├── probe_case/               case-validation scripts, case definitions and their README
├── QA dataset/               355 benchmark items: questions and reference answers, by task type and topic
├── lightrag/                 LightRAG engine used for the graph–vector paradigm
├── cache/                    indexes and stores (not part of the repository)
├── kb_metadata.jsonl         bibliographic record of every indexed document (no source text)
├── steelpcr_config.py        configuration layer, reads `.env` or plain environment variables
├── requirements.txt          environment used for the reported results
├── .env.example              every recognised variable, with safe placeholders
├── .gitignore                the artefacts that stay outside the repository
├── LICENSE                   MIT, covering the code and the benchmark metadata
└── README.md                 this file
```

---

## Quick start

Python 3.11 on Windows x64 was the environment the reported results were produced in;
`requirements.txt` pins it, including the CUDA 13.0 builds of `torch`, `torchaudio` and
`torchvision`.

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in the credentials
```

`.env` is git-ignored. Every value can equally be supplied as a plain environment variable, and a real
environment variable takes precedence over the file. Endpoints are configured per service: the
embedding service, the reranker, the generation model, the judge model and the evaluation embeddings.
Point `STEELPCR_DENSE_INDEX_DIR` and `STEELPCR_SPARSE_INDEX_DIR` at a built knowledge base.

The variables below are the ones needed to run the framework and the experiments; `.env.example` lists
every recognised name.

| Variable | Purpose | Default |
|---|---|---|
| `STEELPCR_EMBED_API_BASE` | embedding endpoint (retrieval and evaluation) | placeholder |
| `STEELPCR_EMBED_MODEL` | embedding model | `Qwen3-Embedding-8B` |
| `STEELPCR_QWEN3_RERANK_URL` | reranker endpoint | placeholder |
| `STEELPCR_GEN_MODEL` | generation model | `deepSeek-flash` |
| `STEELPCR_EVAL_MODEL` | judge model for the answer-quality metrics | `glm-5.3-flash` |
| `STEELPCR_DENSE_INDEX_DIR` | FAISS-HNSW index and chunk store | `rag_core/cache/dense` |
| `STEELPCR_TOP_K_RERANK` | evidence chunks passed to the generator (`context K`) | `7` |

### Ask one question

```python
from rag_core.pipeline import SteelPCRPipeline

pipe = SteelPCRPipeline(enable_expansion=True)
print(pipe.answer("your question")["answer"])
```

`answer()` returns the generated answer together with its retrieval trace: the expanded query, the
ranked evidence, the assembled prompt context and the per-stage latencies. `pipe.retrieve(question)`
stops before generation, and `python rag_core/pipeline.py` runs this example as a script.

### Ask a compound question

```python
from rag_core.pipeline import SteelPCRPipeline
from rag_core import multirag as M

pipe = SteelPCRPipeline(enable_expansion=True)
question = "your question"

res = M.run_multi(1, pipe, question, subpoint=True, merge_rule="round_robin")
print(res["answer"])
```

`run_multi()` also returns the decomposed intents, the per-path evidence, the merged pool and the exact
prompt context. Model decomposition is accepted when it returns 6 to 10 intents; pass `subs=[...]` to
supply your own instead.

---

## Reproducing the experiments

The evaluation and case scripts drive the framework through the same interfaces and add scoring,
checkpointing and reporting on top of them.

```bash
python evaluation/eval1.py                 # retrieval paradigms
python evaluation/eval2.py                 # rerankers
python evaluation/eval3.py                 # RAG versus native models
python evaluation/eval4.py                 # component ablation and paired statistics
python probe_case/case1.py                 # case studies
```

Every evaluation script is resumable: it keeps a checkpoint, and an interrupted run is diagnosed and
continued from the earliest stage that failed rather than from the beginning. `--help` lists the flags
of each script.

---

## Outputs

| Path | Content |
|---|---|
| `evaluation/cache/cache1 … cache4/` | per-experiment checkpoints and intermediate slices |
| `eval_results/results1 … results4/` | per-item metric tables and the aggregated report tables |
| `probe_case/rerun/` | case answers with their evidence, prompts and run metadata |

Intermediate slices are written independently of generation: the candidate pool, the reranked
top-k evidence and the exact prompt context are each stored, so a downstream failure never forces the
retrieval side to be recomputed and any reported number can be re-derived from the stored slice.

---

## Data availability

The source code of SteelPCR-RAG is available in this repository.

The benchmark dataset (questions and reference answers) and the bibliographic record of every indexed
document (`kb_metadata.jsonl`: title, authors, DOI, journal, year) are included in this repository.

Due to copyright restrictions imposed by the scientific publishers, the full text corpus
(`cache/dense/records.jsonl`, the indexed paragraphs), the retrieval index built from it
(FAISS-HNSW vectors and BM25 statistics) and the generated LightRAG graph-vector snapshots (the
PostgreSQL and Qdrant dumps, about 20 GB) are not publicly redistributed.

Researchers who wish to reproduce the full evaluation, including the retrieval metrics, or to verify the
chunk-level evidence, can request these files from the corresponding author through the GitHub profile
linked to this repository or by opening an issue, or rebuild the knowledge base from their own licensed
PDF collections with the released `rag_core/build_index.py` and `rag_core/graphRAG.py` scripts.
`kb_metadata.jsonl` identifies every document by DOI so that it can be located in the original
literature.

---

## Citation

If you use this framework, please cite the accompanying article.

---

## License

MIT; see `LICENSE`.
