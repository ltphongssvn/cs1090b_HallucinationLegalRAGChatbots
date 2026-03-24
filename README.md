# Reducing Hallucination in Legal RAG Chatbots
### A Comparative Study of Retrieval Architectures: From Non-Neural Baselines to Transformer-Based Deep Learning

[![CI](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml/badge.svg)](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml)

- **Author:** Thanh Phong Le — phl690@g.harvard.edu
- **Course:** COMPSCI 1090B: Data Science 2: Advanced Topics in Data Science — Harvard University
- **Cluster node:** 4× NVIDIA L4/A10G (23,034 MiB each) | SLURM job allocation: 1× NVIDIA L4/A10G visible to PyTorch via `CUDA_VISIBLE_DEVICES` | PyTorch build: torch 2.0.1+cu117 (node driver: CUDA 12.8) | Python 3.11.9

> - Although compute nodes physically contain 4× NVIDIA L4/A10G GPUs, student jobs are allocated a
> single GPU by the SLURM scheduler. `CUDA_VISIBLE_DEVICES` is set automatically by SLURM, and
> PyTorch correctly reports `torch.cuda.device_count() == 1`.
>
> - All code uses `.to("cuda")` or
> `.to("cuda:0")` — never a hardcoded physical ordinal — since SLURM remaps the allocated GPU
> to index 0 regardless of physical slot. All experiments are designed and validated under this
> single-GPU constraint.
>
> - KV cache + activations during Mistral generation on retrieved legal contexts consume
> several additional GB beyond model weights. Sequential model loading is the mitigation strategy.

---

## Certified Baseline Stack

| Component                | Certified version                                                                                                                                     |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python                   | 3.11.9                                                                                                                                                |
| PyTorch                  | 2.0.1+cu117                                                                                                                                           |
| transformers             | 4.39.3                                                                                                                                                |
| sentence-transformers    | 3.1.1                                                                                                                                                 |
| Dense retrieval baseline | BAAI/bge-m3 (single-vector dense, CLS pooling — confirmed in repo smoke tests and BAAI's published 1_Pooling/config.json)                             |
| Reranker                 | BAAI/bge-reranker-v2-m3 (sentence-transformers CrossEncoder path smoke-tested in this repo; GPU default, CPU fallback; max_length=1024, batch_size=4) |
| Sparse retrieval         | bm25s                                                                                                                                                 |
| Vector search            | faiss-cpu 1.13.2                                                                                                                                      |
| LLM generator            | mistralai/Mistral-7B-Instruct-v0.2                                                                                                                    |
| NLI classifier           | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli                                                                                              |

* This is the certified baseline stack for the current repository; newer upstream stacks are intentionally deferred for now, they will be adopted only after full re-certification in this repo.
* **`BAAI/bge-m3`** is used **only in single-vector dense retrieval mode**.
* It uses **CLS pooling**, not mean pooling.
* This is defined in BAAI’s published `1_Pooling/config.json`, where:
  * `pooling_mode_cls_token=true`
  * `pooling_mode_mean_tokens=false`
* The same pooling setup has also been **confirmed by smoke tests in this repository**.
* A **runtime assertion** in `src/model_loader.py` prevents accidental pooling overrides.
* The selected **pooling flags** are logged to **W&B once per run**.
* Sparse and multi-vector capabilities are out of scope.

---

## Revised Feasibility Statement

| Component             | Cap                      | Notes                                                                                 |
|-----------------------|--------------------------|---------------------------------------------------------------------------------------|
| Training pairs        | 500K–1M                  | Not 3.2M full LePaRD set                                                              |
| Retrieval evaluation  | 10K–50K queries          | Not 400K full test set                                                                |
| Generation evaluation | 1,000 stratified queries | Fixed budget, stratified by circuit                                                   |
| Iteration corpus      | ~150K opinions (10%)     | Loaded via DVC `data/raw/cl_federal_appellate_bulk` subset; full 1.46M for final runs |
| Architectures         | 3 core + 1 optional      | BM25, BGE-M3, Hybrid + Legal-BERT reference                                           |
| Datasets              | 2 core                   | CourtListener federal appellate subset + LePaRD                                       |

---

## Project Feasibility Statement

**Infrastructure is complete. Corpus preprocessing is underway. The core experiment is the remaining work.**

- ✅ Environment bootstrapped, verified, reproducible (`setup.sh`, tests passing, coverage verified)
- ✅ 1,465,484 federal appellate opinions downloaded, filtered, sharded (7.6GB)
- ✅ DVC + S3 artifact versioning operational
- ✅ All `src/` modules implemented and tested
- 🔄 CourtListener RAG-readiness refinement in progress (Cell 2)
- ⏳ LePaRD acquisition, model training, evaluation remaining

**Prioritization:** LePaRD first → 10–20% subset fast iteration → BM25+BGE-M3+Tier A → scale + Tier B/C

---

## VRAM Budget and Sequential Loading Strategy

**Single GPU (23.7GB L4/A10G, SLURM-allocated).**
* Each phase loads **only the model and data it needs**.
* After a phase finishes, its resources are **unloaded before the next phase begins**.
* The primary GPU dtype is **bfloat16**.
* At startup, `src/environment.py` enforces these checks:
  * `transformers.__version__ == "4.39.3"`
  * `torch.cuda.get_device_capability()[0] >= 8`
  * `torch.cuda.is_bf16_supported()`
* These checks ensure the environment matches the repo’s certified GPU and library assumptions before execution starts.
* `TARGET_GPU_COUNT=1` is set in `.env` to match the **single-GPU allocation** provided by SLURM.
* During preflight, `setup.sh` validates that:
  * `torch.cuda.device_count() == 1`
* This ensures the runtime environment matches the expected **single-GPU execution setup**.
* Per-model fallback to fp16/fp32 remains available if a path fails smoke tests.
* For the NLI phase, `torch.backends.cuda.matmul.allow_tf32 = True` is set as a repo-level performance optimization on L4/A10G; this can trade some FP32 numerical precision for speed and is treated as an opt-in inference optimization, not a semantic guarantee.
* The `allow_tf32` setting is logged in **W&B** for each phase. This provides **transparency** about whether TF32 acceleration was enabled during that phase.
* As a **repo-level cleanup safeguard**, both the **DataLoader** and its **iterator** are explicitly deleted between phases.
* This deletion happens **before** calling:
  * `gc.collect()`
  * `torch.cuda.empty_cache()`
* The goal is to reduce the chance of **stale memory references** and improve **GPU memory cleanup** between pipeline phases.
* `torch.cuda.memory_stats()` is logged at each **phase boundary**.
* This records GPU memory diagnostics **before and after major pipeline stages**.
* It helps track **allocation behavior**, **memory pressure**, and possible **VRAM leaks** across phases.
* The pipeline logs **CUDA stream synchronization time** for each phase.
* **Per phase**, W&B logs:
  * **peak allocated CUDA memory**
  * **peak reserved CUDA memory**

| Phase      | Model loaded                                | Est. VRAM                                                                                                    | Strategy                                                                                                                                                                                                   |
|------------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Retrieval  | BGE-M3                                      | ~2.27GB                                                                                                      | Load (bfloat16) → encode corpus → log embedding norm distribution → save index → unload → empty_cache                                                                                                      |
| Reranking  | bge-reranker-v2-m3                          | ~2GB (GPU default; CPU fallback)                                                                             | Load (bfloat16) → rerank top-50 (max_length=1024, batch_size=4) → log score distribution (min/mean/max/entropy) → serialize scores + ranks → return top-10 → unload → empty_cache                          |
| Generation | Mistral-7B-Instruct-v0.2                    | ~14–15GB + KV cache                                                                                          | Load (bfloat16) → apply chat template → assert max(prompt_tokens) < 32768 → generate (do_sample=False) → log prompt token count (Mistral tokenizer) + completion token count → save → unload → empty_cache |
| NLI eval   | DeBERTa-v3-large-mnli-fever-anli-ling-wanli | ~3GB + activations (overflow sliding windows; DataCollatorWithPadding pad_to_multiple_of=8; pin_memory=True) | Load (bfloat16) → classify per atomic claim → del dataloader → unload → empty_cache                                                                                                                        |
| Citation   | SQLite                                      | 0GB                                                                                                          | CPU only (read-only; check_same_thread=False)                                                                                                                                                              |

Projected peak per phase is expected to remain within the 23.7GB budget; actual peaks logged in W&B.

---

## Methodology Strengths

**1 — Automated Hallucination Measurement (No Human Annotation Bottleneck)**

- **Tier A:**
  - LePaRD 4M+ expert-annotated citation pairs — gold-standard retrieval ground truth
- **Tier B:**
  - `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` classifies each atomic claim independently against individual retrieved chunks. DeBERTa-v3-large has a strict 512-token position limit (`max_position_embeddings=512`; `tokenizer.model_max_length=512` enforced).
  - The NLI tokenizer uses `use_fast=False` as the repo-certified path based on local smoke tests over legal citation text. In this pinned repo environment, overflow-window generation (`return_overflowing_tokens=True, max_length=512, stride=64`) has been regression-tested for this exact model/version combination; public tokenizer docs describe overflow helpers primarily on fast tokenizers, so this behavior is treated as repo-certified rather than generally assumed.
  - Window count per chunk distribution logged to diagnose pathological long passages.
  - Window-level logits are aggregated per chunk (not per window) to preserve retrieval-level semantics and avoid double-counting.
  - The window index triggering each entailment/contradiction label is logged for post-hoc diagnosis.
  - Text is tokenized on the fly in `__getitem__` to yield CPU tensors suitable for `pin_memory=True`; `DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)` pads variable-length tensors for Tensor Core alignment on the L4/A10G.
  - `torch.backends.cuda.matmul.allow_tf32 = True` is set for the NLI phase as a repo-level performance optimization — trades some FP32 numerical precision for speed, treated as opt-in inference optimization not a semantic guarantee; state logged per phase.
  - `num_workers` is configurable (repo default 2; 4 for dedicated runs; 0 as fallback under SLURM cgroup restrictions).
  - As a repo-level cleanup safeguard, the DataLoader and iterator are explicitly deleted before `gc.collect()` + `torch.cuda.empty_cache()`.
  - Aggregation rule: if any window Entails → Entailment; else if any contradicts → Contradiction; else → Neutral.
  - Zero-claim responses logged, excluded from normalization, and reported as a separate metric.
  - Contradiction rate normalized by per-query claim count and as claims per 1K tokens.
  - NLI confidence scores are diagnostic indicators, not calibrated probabilities.
  - Neutral = evidence-gap.
  - Fully local, no API calls.
- **Tier C:**
  - SQLite connection uses `check_same_thread=False`; the evaluation path is read-only so no concurrent writes occur.
  - A unique citation hash (`opinion_id + anchor span`) is logged per Tier C lookup for deduplication analysis.
  - If NULL → Hard Citation Hallucination logged, NLI skipped.
  - If found + no local NLI support → CitationFound_NoLocalSupport logged.
  - When sliding-window fallback is used, the token offset of the citation anchor within the selected window is logged to diagnose citation drift. Windowing is citation-anchor-first:
    - (1) Hybrid uses highest-scoring reranker passage;
    - (2) keyword/regex on citation anchor;
    - (3) sliding-window fallback only if anchor extraction fails.
  - **Tier C verifies citation existence and local evidence support, not full legal reasoning correctness.**

**2 — Clean Experimental Design**

- `mistralai/Mistral-7B-Instruct-v0.2` held **constant** across all architectures with greedy decoding (`do_sample=False`; `temperature` omitted to suppress warnings in `transformers 4.39.3`).
- All prompts formatted with `tokenizer.apply_chat_template(...)` before tokenization.
- A runtime assertion `assert max(prompt_tokens) < 32768` fires loudly before generation if context exceeds Mistral's hard limit.
- Prompt token count (Mistral tokenizer) and completion token count logged per query. Observed differences are attributable to the retrieval setup.

**3 — Grounded in a Real Failure Case**

- Targets *Mata v. Avianca Airlines* (2023). Narrow, testable, motivated by documented consequence.

**4 — Production-Grade Reproducibility Already Operational**

- `src/repro.configure()` + `uv.lock` + DVC + manifest checksums + tests passing.
- `src/environment.py` asserts exact library versions and GPU configuration at startup.
- `HF_TOKEN` is required by this repo on the shared cluster for authenticated Hub access and to reduce resolver/rate-limit failures; loaded via `dotenv` in `src/environment.py`.

---

## Addressing TF Reviewer Comments and Instructor Notes

### Comment 1 — Feasibility of Hallucination Measurement
The evaluation is organized into **three tiers** to keep hallucination measurement feasible, automated, and scalable.

#### Tier A
Retrieval Grounding
* Uses a **capped LePaRD subset**.
* Reports standard retrieval metrics:
  * **Recall@k**
  * **MRR**
  * **NDCG@10**

#### Tier B
* Use a **1,000-query stratified sample** for evaluation.
* Run NLI locally with:
  * `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
* This evaluation path uses **no external API calls**.
* The **NLI tokenizer** is configured with:
  * `use_fast=False` *(repo-certified setting)*
  * `tokenizer.model_max_length = 512` *(strictly enforced)*
* **Overflow window generation** has been **repo-certified** for this **exact model and version combination**. This means the behavior has been **verified within this repository setup**, rather than assumed to work generically across all environments.
* The system logs the **distribution of window counts per chunk**.
* **Window-level logits** are aggregated at the **chunk level** before final scoring.
* Text is tokenized inside `__getitem__`.
* Batch padding is handled by:
  * `DataCollatorWithPadding(pad_to_multiple_of=8)`
* Padding to a multiple of 8 helps keep tensor shapes hardware-friendly during batching. Per-claim aggregation:

| Category          | Definition                             | Aggregation                                 | Reported As                                     |
|-------------------|----------------------------------------|---------------------------------------------|-------------------------------------------------|
| **Entailment**    | Any window supports the claim          | any(entail) → Entailment                    | Faithfulness metric + confidence score          |
| **Contradiction** | No entailment, any window contradicts  | no entail + any(contradict) → Contradiction | Contradiction rate (normalized + per 1K tokens) |
| **Neutral**       | No entailment or contradiction         | else → Neutral                              | Evidence-gap — **not hallucination by default** |

* The **window index** that triggered each label is logged.
* **NLI confidence scores** are recorded for diagnostics only.
* These confidence scores are **not treated as calibrated probabilities**.
* **Zero-claim responses** are excluded from the main hallucination-rate calculation.
* The **zero-claim rate** is reported separately as its own metric.


**Tier C:**
* SQLite is configured with:
  * `check_same_thread=False`
  * **read-only access**
* A unique **citation hash** is logged for each lookup.
* If the citation resolves to **`NULL`**, it is labeled **Hard Citation Hallucination**.
* If the citation is found but has **no supporting evidence**, it is labeled **CitationFound_NoLocalSupport**.
* If fallback passage localization is required, the system logs the **anchor offset** for debugging.
* Passage selection follows this order:
  * **(1) Hybrid reranker**
  * **(2) Keyword / regex matching**
  * **(3) Sliding-window fallback**
- **Tier C verifies citation existence and local evidence support, not full legal reasoning correctness.**

### Comment 2 — Embedding Model Training

| Architecture                     | Base Model                                | Training                     | Key Hyperparameters                                                                              |
|----------------------------------|-------------------------------------------|------------------------------|--------------------------------------------------------------------------------------------------|
| BM25                             | None                                      | None                         | k1=1.5, b=0.75                                                                                   |
| BGE-M3                           | BAAI/bge-m3 (CLS pooling per BAAI config) | MultipleNegativesRankingLoss | lr=1e-5, warmup=10%, batch=32, epochs=3                                                          |
| Hybrid: BM25+BGE-M3+CrossEncoder | BGE-M3 + BAAI/bge-reranker-v2-m3          | RRF top-50 → rerank → top-10 | sentence-transformers CrossEncoder path smoke-tested in this repo; max_length=1024, batch_size=4 |
| Legal-BERT (optional)            | Legal-BERT (12GB legal text)              | MultipleNegativesRankingLoss | lr=2e-5, warmup=10%, batch=32, epochs=3                                                          |

---

## Self-Audit Findings and Corrections

### Revision 1 — Domain Gap
- Claims scoped to federal appellate opinions.

### Revision 2 — Hallucination Metric Definition

* **Contradiction rate** is reported in two normalized forms:
  * by **claim count**
  * per **1,000 tokens**
* **Zero-claim responses** are tracked and reported as a **separate metric**.
* **CitationFound_NoLocalSupport** is logged separately from other citation outcomes.
* The system logs the **distribution of window counts per chunk**.
* **Window-level logits** are aggregated at the **chunk level**.
* **NLI scores** are treated as **diagnostic indicators only**, not calibrated probabilities.
* **Neutral** is **not automatically counted as hallucination**.

### Revision 3 — Citation Existence vs. Citation Correctness

* **Tier C** assigns and logs a unique citation hash for every citation lookup.
* If the citation resolves to `NULL`, it is labeled **Hard Citation Hallucination**.
* If the citation is found but the cited passage does **not** provide local NLI support for the claimed proposition, it is labeled **CitationFound_NoLocalSupport**.
* If fallback passage localization is needed, the system logs the **citation anchor offset** for debugging.
* Local evidence is selected in this order:
  * **(1) Hybrid reranker**
  * **(2) Keyword / regex matching**
  * **(3) Sliding-window fallback**

### Revision 4 — Architecture Alignment

* The retrieval systems are organized along a spectrum:
  * **BM25** → lexical baseline
  * **BGE-M3** → dense retriever
  * **Hybrid** → combined retrieval pipeline
* **BM25** is indexed over the **same pre-chunked payloads** used by **BGE-M3**.
* This keeps the comparison fair by ensuring both methods operate on the same chunked inputs.
* The **1024-subword chunk budget** is a **controlled experimental design choice**.
* It is **not** a hard limitation of **BGE-M3** itself.

### Revision 5 — Scientific Claim Precision
Scoped to federal appellate opinions.

### Revision 6 — Outdated Architectures
CNN/BiLSTM replaced — see Architecture Classification.

### Revision 7 — Chunking Accuracy

* **spaCy tokens are not the same as BPE subword tokens.**
* `nlp.max_length` is set high enough to process full federal appellate opinions safely.
* Chunks are built using:
  * `AutoTokenizer.from_pretrained(encoder_model)`
  * inside the chunking loop
* The chunking policy is standardized as:
  * **1024 subwords per chunk**
  * **128-subword overlap**
  * **512-subword chunks for Legal-BERT**
* This chunk budget is a **controlled experimental design choice** for retriever–reranker consistency.
* It is **not** a hard limit of **BGE-M3** itself.
* The **same pre-chunked payloads** are used for both:
  * `bm25s`
  * `BGE-M3`
* Effective prompt token count is logged per query using the **Mistral tokenizer**.
* A runtime guard enforces context length:
  * `assert max(prompt_tokens) < 32768`
* If that assertion fails, the pipeline stops loudly rather than silently overflowing context.
* Optional ablation:
  * test **64-subword overlap**
  * on a **10% subset**

### Revision 8 — LLM Generator

* The generator model is **`mistralai/Mistral-7B-Instruct-v0.2`**.
* It has been **smoke-tested in this repository** under **`transformers 4.39.3`**.
* All prompts are formatted using:
  * `tokenizer.apply_chat_template(...)`
* Generation uses **greedy decoding** with:
  * `do_sample=False`
* `temperature` is intentionally **omitted**.
* A **runtime assertion** checks prompt length before generation starts.
* This ensures the pipeline fails loudly if the prompt exceeds the allowed context size.
* The model has **no built-in moderation layer**.

### Revision 9 — VRAM / KV Cache Risk

* All experiments run on a **single SLURM-allocated GPU** with **23.7GB VRAM**.
* All models use **bfloat16** as the primary dtype.
* The environment asserts the following before execution:
  * `torch.cuda.is_bf16_supported()`
  * GPU capability **≥ 8.0**
  * pinned `transformers` version
* To stay within VRAM limits, the pipeline uses **sequential model loading**.
* Memory cleanup is enforced between phases using:
  * explicit **DataLoader deletion**
  * `torch.cuda.empty_cache()`
  * `gc.collect()`
* The pipeline logs memory and execution diagnostics at each phase, including:
  * `torch.cuda.memory_stats()`
  * CUDA stream synchronization time
  * `allow_tf32` state
* The **CrossEncoder reranker** is constrained to:
  * `max_length=1024`
  * `batch_size=4`
* The **NLI phase** uses:
  * repo-certified overflow windowing
  * `DataCollatorWithPadding(pad_to_multiple_of=8)`
  * `allow_tf32=True` *(opt-in)*
  * `pin_memory=True`
  * configurable `num_workers`
* For SLURM-restricted environments, `num_workers=0` is available as a fallback.
* The **projected peak memory usage** is expected to remain within the **23.7GB VRAM budget**.


### Revision 10 — Tier C API Rate Limits

* Tier C does **not** rely on live API calls at evaluation time.
* API rate-limit risk is avoided by using a **local SQLite index** instead.
* The SQLite connection is configured with:
  * `check_same_thread=False`
  * **read-only access**
* This allows citation checks to run **locally at scale** without external API throttling.
* Result:
  * no live API dependency
  * no rate-limit bottleneck during large evaluation runs

### Revision 11 — Training Cost vs Timeline
Explicit compute caps set up front — see Revised Feasibility Statement.

---

## Research Question
* The central research question is:
  * **Which retrieval setup most improves evidence grounding and reduces contradiction and neutral-evidence failures in a legal RAG system built over U.S. federal appellate opinions?**
* The study compares **three core retrieval systems**:
  * **BM25**
  * **BGE-M3**
  * **Hybrid BM25 + BGE-M3 + CrossEncoder**
* **Legal-BERT** is included as an **optional domain-reference model**.
* The generator model is held constant as:
  * **`mistralai/Mistral-7B-Instruct-v0.2`**
* The generator is used under the following fixed conditions:
  * **frozen weights**
  * **greedy decoding**
  * **chat template applied**
  * **sequential loading**

---

## Scientific Methods

### Exact Hypotheses

* **H1:** The **Hybrid** system (**BM25 + BGE-M3 + CrossEncoder**) is expected to achieve **significantly higher Recall@10** than:
    * **BM25 alone**
    * **BGE-M3 alone**
  * Statistical significance will be tested using:
    * **paired bootstrap**
    * **p < 0.05**

* **H2:** Retrieval architectures with higher **Recall@10** are expected to produce significantly lower **contradiction rates** in downstream generation.

  * Contradiction rate is reported in two normalized forms:
    * by **claim count**
    * per **1,000 tokens**
  * **Zero-claim responses are excluded** from this calculation.
  * The purpose of this hypothesis is to test whether better retrieval reduces **active hallucination**.
  * This also helps separate true contradiction errors from simple **retrieval-coverage gaps**.

* **H3:** The **Hybrid** retrieval system is expected to achieve **higher Recall@10** than either of the single-method baselines used alone:
    * **BGE-M3**
    * **BM25**

### Task Definition

* **Retrieval:** Minimize rank(p* | q, C) over CourtListener federal appellate corpus C. The system receives a legal question **`q`**, searches the corpus **`C`** of federal appellate opinions, and tries to place the **correct supporting passage `p*`** as high as possible in the results list. `rank(p* | q, C)` is the retrieval position of the gold legal passage `p*` for query `q` within corpus `C`, and minimizing it means pushing the correct evidence as close to the top of the retrieved results as possible.

* **Generation input** consists of:
  * the query `q`
  * the **top-10 retrieved chunks**
  * each chunk capped at **1024 subwords**
“q + top-10 chunks” means the user’s query combined with the 10 highest-ranked retrieved evidence chunks, which together form the grounded input sent to the generator model.
  * The full prompt is formatted using:
    * `tokenizer.apply_chat_template(...)`
  * The generator model is:
    * `mistralai/Mistral-7B-Instruct-v0.2`
  * The generator is kept **frozen** during evaluation.
  * Decoding uses:
    * `do_sample=False`
  * The model is loaded using a **sequential loading strategy**.
  * Output response is:
    * `R`
  * Before generation starts, a **runtime assertion** checks prompt length.
  * The **effective prompt token count** is logged for each query using the **Mistral tokenizer**.

  * Begin with **Tier A** on a **10–20% subset** of the data.
  * Use this smaller run to **validate the pipeline and metrics first**.
  * Add **Tier B** and **Tier C** only after Tier A has been successfully validated.

### Data Split Strategy (Capped)

| Split | Capped size | Use |
|-------|-------------|-----|
| Train | **500K–1M pairs** | Fine-tuning BGE-M3, reranker |
| Validation | **50K pairs** | Hyperparameter tuning, RRF grid search |
| Test (retrieval) | **10K–50K pairs** | Tier A evaluation |
| Test (generation) | **1,000 queries** | Tier B/C evaluation |

### Sample Size

| Component | Capped size | Justification |
|-----------|-------------|---------------|
| Training pairs | 500K–1M | Contrastive convergence in hours on L4/A10G |
| Retrieval eval | 10K–50K | Sufficient precision; avoids multi-day NLI cost |
| Generation eval | 1,000 queries | ±2.5pp at 95% CI; stratified by circuit |

### Evaluation Protocol

1. Acquire LePaRD; cap training at 500K–1M pairs
2. Train/adapt all retrievers on capped train split; `bm25s` indexed over pre-chunked payloads
3. Validate on 10–20% CourtListener subset; scale if results hold; log recall@k vs nprobe on validation set to justify IVF parameters
4. Sequential evaluation per 1,000 queries (allocator diagnostics + CUDA stream sync + allow_tf32 state logged + explicit DataLoader deletion + `empty_cache` + `gc.collect` between phases):
   - Load BGE-M3 (bfloat16) → log pooling flags to W&B → retrieve top-50 (1024-subword chunks) → log embedding norm distribution → save → unload
   - Load bge-reranker-v2-m3 (bfloat16; CrossEncoder; GPU default; max_length=1024, batch_size=4) → rerank top-50 → log score distribution (min/mean/max/entropy) → serialize scores + ranks → return top-10 → unload
   - Load Mistral-7B-Instruct-v0.2 (bfloat16) → apply chat template → assert max(prompt_tokens) < 32768 → generate (do_sample=False) → log prompt token count (Mistral tokenizer) + completion token count → save → unload
   - Load DeBERTa-v3 NLI (bfloat16; use_fast=False; model_max_length=512; repo-certified overflow windowing max_length=512, stride=64; log window count per chunk distribution; tokenized in __getitem__; DataCollatorWithPadding pad_to_multiple_of=8; allow_tf32=True opt-in; pin_memory=True; num_workers configurable, 0 as SLURM fallback) → classify each atomic claim; window-level logits aggregated per chunk → log window index per label → log labels + confidence scores + claim counts → del dataloader → unload
   - SQLite (check_same_thread=False; read-only) → log citation hash (opinion_id + anchor span) → NULL: Hard Citation Hallucination; found + no support: CitationFound_NoLocalSupport; log anchor offset on fallback; found + support: (1) reranker scores, (2) keyword/regex, (3) sliding-window → NLI 512-token passage (CPU)
5. Aggregate; report contradiction rate (normalized + per 1K tokens), zero-claim rate, Hard Citation Hallucination rate, CitationFound_NoLocalSupport rate; significance tests; W&B

### Significance Testing

Paired bootstrap (B=10,000). P-values + 95% CIs. Cohen's d. Benjamini-Hochberg FDR.

### Ablation Plan

| Ablation | Isolates |
|----------|---------|
| BGE-M3 vs Hybrid | BM25+RRF+reranker contribution |
| Hybrid w/o reranker vs full | Cross-encoder reranker contribution |
| Legal-BERT vs BGE-M3 | Domain adaptation vs scale |
| k ∈ {1, 5, 10, 20} | Retrieval depth sensitivity |
| Training size: 100K vs 500K vs 1M | Data scaling effect |
| With vs without Stage 3 normalization | Preprocessing contribution |
| Contradiction vs Neutral vs combined | Metric sensitivity |
| 128 vs 64 subword overlap | Chunk overlap sensitivity |

---

## Architecture Classification

| ID | Architecture | Type | Role |
|----|-------------|------|------|
| (a) | BM25 | Non-neural baseline | Reference floor |
| (b) | BGE-M3 (BAAI/bge-m3, CLS pooling per BAAI config) | Modern dense retriever — CLS pooling per published BAAI config | Primary dense baseline — 1024-subword chunks (design choice) |
| (c) | Hybrid: BM25+BGE-M3+bge-reranker-v2-m3 | Transformer + lexical + CrossEncoder reranker | **Expected strongest** |
| (d) | Legal-BERT Bi-Encoder | Domain-specific Transformer | Optional domain-reference model |

* **Sparse retrieval** uses **`bm25s`**, `bm25s` is indexed over the **same pre-chunked payloads** embedded by **BGE-M3**. “pre-chunked payloads” means the retrieval-ready text chunks created ahead of time from legal opinions and reused identically by BM25 and BGE-M3 so the architecture comparison stays fair and reproducible.**

* **Dense retrieval** uses the repo-certified stack:
  * `sentence-transformers 3.1.1`
  * `transformers 4.39.3`
* It uses **CLS pooling**, following BAAI’s published:
  * `1_Pooling/config.json`
* A **runtime assertion** in `model_loader.py` prevents accidental pooling override.
* The active **pooling flags** are logged to **Weights & Biases (W&B)** once per run.

* The chunking policy is standardized as follows:
  * **1024 subwords per chunk**
  * **128-subword overlap**
  * **512 subwords per chunk for Legal-BERT**
* This is a **controlled experimental design choice**.
* It is intended to keep chunking **consistent across the pipeline**.

Reranker (`BAAI/bge-reranker-v2-m3`) via `sentence-transformers CrossEncoder` path smoke-tested in this repo; bfloat16; GPU default with CPU fallback; max_length=1024 + batch_size=4; score distributions (min/mean/max/entropy) logged; ranks serialized; top-50 → top-10.
FAISS Flat for capped/validation; IVF for full-corpus final runs — `index.train()` on ~100K-vector subset; `assert index.is_trained`; recall@k vs nprobe logged on validation set to justify IVF parameters; nprobe/nlist logged per run in W&B; (`OMP_NUM_THREADS` + `MKL_NUM_THREADS` in `setup.sh`).
CPU FAISS (`faiss-cpu 1.13.2`).

---

## Current Pipeline Status

| Stage | Status | What exists | What remains |
|-------|--------|-------------|--------------|
| Environment bootstrap | ✅ Complete | Tests passing, coverage verified, manifest generated | — |
| CourtListener download | ✅ Complete | 1,465,484 opinions · 159 shards · 7.6GB | — |
| DVC + S3 | ✅ Complete | Remote configured | `dvc push` data shards |
| CourtListener RAG prep | 🔄 In progress | JSONL with 23-field schema | Tokenizer-aware chunking (1024 subwords) + SQLite citation index |
| LePaRD acquisition | ⏳ **Priority 1** | `src/dataset_loader.py` ready | Download + DVC (cap at 500K–1M) |
| Feature/index generation | ⏳ Not started | `src/lightning_datamodule.py`, `src/split.py` | BM25 (pre-chunked payloads) + FAISS Flat (eval) / IVF (train+add, final) |
| Model training | ⏳ Not started | Architectures + compute caps specified | Training runs |
| Evaluation (Tiers A/B/C) | ⏳ Not started | Sequential loading + repo-certified overflow windowing documented | Capped eval runs |
| Experiment tracking | ⏳ Not started | `src/wandb_logger.py` implemented | W&B setup and per-phase VRAM logging integration pending |

---

## DL System Stack

### Stage 1 — Raw Data Acquisition

**CourtListener federal appellate subset (🔄):** 1,465,484 opinions, 23-field JSONL.
SQLite citation index for Tier C (`check_same_thread=False`; read-only evaluation path;
citation hash logged per lookup). Fast iteration: ~150K (10% subset); full 1.46M for final runs.

**LePaRD (⏳ Priority 1):** ~4M pairs; training capped at 500K–1M. HuggingFace + GitHub.
`src/dataset_loader.py` ready. `HF_TOKEN` is required by this repo on the shared cluster for
authenticated Hub access and to reduce resolver/rate-limit failures; loaded via `dotenv` in
`src/environment.py` and passed to every `from_pretrained()` call.

All tracked by DVC → S3 `cs1090b-hallucinationlegalragchatbots` (us-east-2).

### Stage 2 — Raw Artifact Registration
Immutable manifest: 159 checksums, 1,465,484 rows, filter chain, `uv.lock` SHA256.
Contract tests validate manifest on every pipeline run. SBOM (357 components).

### Stage 3 — Preprocessing + Tokenizer-Aware Chunking
Streaming CSV. HTML strip, encoding cleanup, Westlaw boilerplate removal.
**spaCy stripped-down pipeline** (`exclude=["ner", "parser", "lemmatizer"]`) with `nlp.max_length`
set high enough for full appellate opinions. Chunks built using
`AutoTokenizer.from_pretrained(encoder_model)` inside the chunking loop. **Chunk budget
standardized to 1024 subwords with 128-subword overlap (512 for Legal-BERT) as a controlled
design choice for retriever–reranker consistency, not a limit of BGE-M3 itself.** Pre-chunked
string payloads saved and reused by both BGE-M3 and `bm25s`. Effective prompt token count
(Mistral tokenizer) logged per query; `assert max(prompt_tokens) < 32768` fires loudly if
violated.
Citation-aware splits, metadata per chunk (court_id, year, is_precedential, opinion_id, chunk_index).
SQLite citation index built from `data/raw/cl_federal_appellate_bulk` via `src/extract.py`.

### Stage 4 — Index Generation *(not started)*
BM25 (`bm25s`) indexed over pre-chunked payloads from Stage 3 (not raw text). FAISS dense index:
**Flat for capped/validation** (no training required); **IVF for full-corpus final runs** —
`index.train()` on ~100K-vector random subset; `assert index.is_trained` guard before search;
recall@k vs nprobe logged on validation set to justify IVF parameters; nprobe/nlist logged per
run in W&B (`OMP_NUM_THREADS` + `MKL_NUM_THREADS` in `setup.sh`). `BAAI/bge-reranker-v2-m3`
via `sentence-transformers CrossEncoder` path smoke-tested in this repo; bfloat16; GPU default,
CPU fallback; max_length=1024, batch_size=4; score distributions (min/mean/max/entropy) logged;
scores and ranks serialized; returns top-10. SQLite citation index.

### Stage 5 — Model Training *(not started)*
Capped at 500K–1M pairs. Early stopping + gradient accumulation. GPU hours → W&B.

### Stage 6 — Evaluation *(not started)*
All models primary bfloat16; `transformers.__version__ == "4.39.3"`, `get_device_capability()[0] >= 8`,
and `is_bf16_supported()` asserted in `environment.py`; `TARGET_GPU_COUNT=1` validated against
`torch.cuda.device_count()`. Sequential loading per VRAM Budget table. `torch.cuda.memory_stats()`
+ CUDA stream sync time + `allow_tf32` state logged per phase + explicit DataLoader deletion
(repo-level cleanup safeguard) + `torch.cuda.empty_cache()` + `gc.collect()` at every phase
boundary. BGE-M3: pooling flags logged to W&B + embedding norm distribution logged. Reranker
score distributions (min/mean/max/entropy) logged. Mistral: `tokenizer.apply_chat_template(...)`
enforced; `assert max(prompt_tokens) < 32768`; `do_sample=False`; prompt token count (Mistral
tokenizer) and completion token count logged per query.
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`: `use_fast=False` (repo-certified);
`tokenizer.model_max_length=512`; overflow windowing repo-certified for this exact model/version;
window count per chunk distribution logged; window-level logits aggregated per chunk; window index
per label logged; `DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)`;
`allow_tf32=True` (opt-in; state logged); `pin_memory=True`; `num_workers` configurable.
SQLite: citation hash logged per lookup; NULL → Hard Citation Hallucination;
found + no support → CitationFound_NoLocalSupport; anchor offset logged on fallback.
Reported metrics: contradiction rate (normalized + per 1K tokens), zero-claim rate, Hard Citation
Hallucination rate, CitationFound_NoLocalSupport rate. Projected peak VRAM within 23.7GB.

### Stage 7 — Experiment Tracking *(not started)*
`src/wandb_logger.py` ready. Per-phase GPU hours, peak CUDA memory, allocator diagnostics, CUDA
stream sync times, `allow_tf32` state per phase, BGE-M3 pooling flags, embedding norm
distributions, reranker score distributions (min/mean/max/entropy), FAISS recall@k vs nprobe +
nprobe/nlist, NLI window count distributions, NLI confidence distributions, window indices per
label, per-query claim counts, claims per 1K tokens, zero-claim rate, Hard Citation Hallucination
counts, CitationFound_NoLocalSupport counts, citation hashes, citation anchor token offsets,
prompt token counts (Mistral tokenizer) + completion token counts, gradient norms — integration pending.

---

## Datasets

| Dataset | Size | License | Role | Status |
|---------|------|---------|------|--------|
| CourtListener federal appellate subset | 1,465,484 opinions | CC BY-ND 4.0 | Retrieval corpus + SQLite citation index | 🔄 In progress |
| LePaRD (ACL 2024) | ~4M pairs | Open research | Training (500K–1M cap) + evaluation | ⏳ **Priority 1** |

---

## Reproducibility
```bash
bash setup.sh          # bootstrap environment, manifests, and verification tests
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"  # verify torch + CUDA + single GPU
uv run dvc pull        # pull DVC artifacts from S3
uv run pytest --cov=src --cov-report=term-missing  # run full test suite with coverage
jupyter lab notebooks/cs1090b_HallucinationLegalRAGChatbots.ipynb
```

`src/repro.configure()` — first statement in every notebook cell and CLI script.
`HF_TOKEN` is required by this repo on the shared cluster; add to `.env` before running.
`TARGET_GPU_COUNT=1` must be set in `.env` to match SLURM single-GPU allocation.
All device references use `.to("cuda")` or `.to("cuda:0")` — never a hardcoded physical ordinal.
`uv.lock` + DVC → configuration-reproducible and determinism-controlled experiments.

---

## Project Structure
```
cs1090b_HallucinationLegalRAGChatbots/
├── src/
│   ├── repro.py                 # GITIGNORED — generated by setup.sh
│   ├── environment.py           # Runtime assertions: transformers version, bf16 support, capability ≥ 8.0, TARGET_GPU_COUNT=1; loads HF_TOKEN
│   ├── config.py                # PipelineConfig
│   ├── pipeline.py              # Stage orchestration
│   ├── bulk_download.py         # CourtListener S3 download
│   ├── s3_discovery.py          # S3 discovery + pinned snapshots
│   ├── filter_chain.py          # Courts → dockets → clusters
│   ├── extract.py               # Streaming CSV → JSONL shards + SQLite citation index
│   ├── manifest.py              # Artifact manifest
│   ├── manifest_collector.py    # Environment manifest + SBOM
│   ├── schemas.py               # FilterResult, OpinionRecord
│   ├── row_normalizer.py        # Text normalization
│   ├── row_validator.py         # Schema validation
│   ├── validation.py            # Pipeline contract validation
│   ├── split.py                 # Train/val/test split
│   ├── dataset_config.py        # Hydra DatasetConfig; num_workers configurable (default 2)
│   ├── dataset_loader.py        # HuggingFace / artifact loader — ready for LePaRD
│   ├── dataset_probe.py         # Quality signal analysis
│   ├── lightning_datamodule.py  # PyTorch DataModule; repo-certified overflow windowing; tokenized in __getitem__; DataCollatorWithPadding
│   ├── model_loader.py          # Safetensors model loader; CLS pooling assertion + W&B logging for BGE-M3
│   ├── hf_export.py             # HuggingFace Hub export
│   ├── drift_check.py           # Manifest drift detection
│   ├── wandb_logger.py          # W&B tracking + per-phase GPU hours + VRAM
│   ├── exceptions.py            # PipelineError
│   └── timer.py                 # cell_timer
├── notebooks/cs1090b_HallucinationLegalRAGChatbots.ipynb
├── tests/                       # test suite with coverage
├── configs/                     # Hydra YAML
├── scripts/                     # setup.sh helpers
├── data/                        # GITIGNORED — DVC → S3 us-east-2
│   └── raw/
│       ├── cl_bulk/             # ~57GB raw CSVs
│       └── cl_federal_appellate_bulk/  # 159 shards ~7.6GB
├── logs/                        # GITIGNORED — runtime artifacts
├── .dvc/                        # S3: cs1090b-hallucinationlegalragchatbots
├── setup.sh
├── pyproject.toml               # requires-python = ">=3.11,<3.12"
└── uv.lock                      # pinned dependency lockfile
```

---

## Certified Baseline Tech Stack

| Layer | Tool | Certified version |
|-------|------|------------------|
| Language | Python | 3.11.9 |
| Package manager | uv | 0.10.2 (repo-pinned) |
| Deep learning | PyTorch | 2.0.1+cu117 (node driver: CUDA 12.8) |
| Transformers | HuggingFace transformers | 4.39.3 (version-asserted at startup) |
| Sentence embeddings | sentence-transformers | 3.1.1 |
| Dense retrieval | BAAI/bge-m3 (CLS pooling per BAAI published config; runtime assertion + pooling flags logged to W&B) | HuggingFace — ~2.27GB (bfloat16) |
| CrossEncoder reranker | BAAI/bge-reranker-v2-m3 | sentence-transformers CrossEncoder path smoke-tested in this repo — bfloat16; GPU ~2GB, CPU fallback; max_length=1024, batch_size=4; score distributions (min/mean/max/entropy) logged; scores serialized; top-50→top-10 |
| BM25 retrieval | bm25s | 0.3.2.post1 — indexed over pre-chunked payloads (not raw text) |
| Vector search | faiss-cpu | 1.13.2 — Flat for eval; IVF (index.train() + assert index.is_trained; recall@k vs nprobe logged; nprobe/nlist logged) for final corpus |
| LLM generator | mistralai/Mistral-7B-Instruct-v0.2 | smoke-tested in repo; bfloat16; chat template applied; do_sample=False; prompt length assertion; prompt/completion token counts logged |
| Tokenizer dependency | sentencepiece | 0.2.1 |
| NLI classifier | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | smoke-tested in repo; bfloat16; use_fast=False (repo-certified); model_max_length=512; overflow windowing repo-certified; window count distribution logged; DataCollatorWithPadding pad_to_multiple_of=8; allow_tf32=True (opt-in; state logged); pin_memory=True; window-level logits aggregated per chunk; citation hash logged |
| NLP sentence boundaries | spaCy + en_core_web_sm | 3.8.11 / 3.8.0 (stripped, nlp.max_length set for long opinions) |
| Chunking tokenizer | AutoTokenizer (transformers) | 1024-subword chunks, 128 overlap — design choice (512 for Legal-BERT) |
| Citation index | SQLite (stdlib) | check_same_thread=False; read-only; citation hash logged; built via src/extract.py |
| Experiment tracking | W&B | 0.25.1 |
| Data versioning | DVC 3.67.0 + dvc-s3 3.3.0 | S3 remote: cs1090b-hallucinationlegalragchatbots (us-east-2) |
| Test framework | pytest + hypothesis | lockfile-pinned |
| Linting | ruff | lockfile-pinned |
| Type checking | mypy | lockfile-pinned |

---

## Ethical Considerations

All datasets publicly available. CourtListener CC BY-ND 4.0.
`mistralai/Mistral-7B-Instruct-v0.2` has no built-in moderation mechanisms per its model card;
outputs are used strictly for retrieval research under academic supervision.
PII follows provider redaction practices. No human annotation for hallucination measurement.

---

GPU pipeline comparing retrieval architectures (TF-IDF, CNN, LSTM, BERT bi-encoder, KG-augmented) to reduce hallucination in legal RAG chatbots.

**Hardware:** 4x NVIDIA L4/A10G GPUs | Python 3.11.9 | torch 2.0.1+cu117 | CUDA 11.7 (driver 12.8)

## Quick Start
```bash
# Clone and install hooks (required once)
git clone https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots.git
cd cs1090b_HallucinationLegalRAGChatbots
uv run pre-commit install && uv run pre-commit install --hook-type pre-push

# Full GPU setup
bash setup.sh

# CPU-only (no GPU allocation needed)
SKIP_GPU=1 bash setup.sh

# Preview all side effects without executing
DRY_RUN=1 bash setup.sh
```

## Setup Modes

| Mode | Command |
|---|---|
| Full GPU setup | `bash setup.sh` |
| CPU-only | `SKIP_GPU=1 bash setup.sh` |
| Dry run | `DRY_RUN=1 bash setup.sh` |
| Quiet (CI) | `LOG_LEVEL=0 bash setup.sh` |
| Verbose | `LOG_LEVEL=2 bash setup.sh` |
| No download | `NO_DOWNLOAD=1 bash setup.sh` |
| No Jupyter | `NO_JUPYTER=1 bash setup.sh` |
| Single step | `STEP=<fn_name> bash setup.sh` |

## Notebook Cell 1 (required first line)
```python
from src.repro import configure
repro_cfg = configure()
```

## Testing
```bash
# Unit tests (fast, no GPU)
uv run pytest tests/ -m unit -v

# Contract tests
uv run pytest tests/ -m contract -v

# GPU tests (requires allocation)
uv run pytest tests/ -m gpu -v

# All with coverage
uv run pytest tests/ -m "unit or contract" --cov=src --cov-report=term-missing

# Shell script tests (requires bats-core)
bats tests/shell/

# All pre-commit hooks
uv run pre-commit run --all-files
```

## Coverage Enforcement

80% per-file minimum enforced at three levels:
- **pre-push hook** — blocks push if below threshold
- **CI pipeline** — `unit-tests` job with coverage report
- **pyproject.toml** — `fail_under = 80`

## CI/CD Pipeline

| Job | Trigger | Purpose |
|---|---|---|
| lint | push/PR | ruff + mypy |
| shell-tests | push/PR | bats-core hook tests |
| unit-tests | push/PR | pytest unit + coverage |
| cpu-smoke | push/PR | CPU forward pass + FAISS IVF |
| security | push/PR | pip-audit + CycloneDX SBOM |

## Module Map

| File | Responsibility |
|---|---|
| `scripts/lib.sh` | Constants, colors, step framework, messaging, guards |
| `scripts/bootstrap_env.sh` | uv, lockfile, venv, deps, drift |
| `scripts/validate_gpu.sh` | GPU detection, hardware policy, smoke tests |
| `scripts/setup_nlp.sh` | spaCy model download and verification |
| `scripts/setup_notebook.sh` | Repro env, repro module, stability, kernel |
| `scripts/validate_tests.sh` | Tiered test execution (5 tiers) |
| `scripts/manifest.sh` | Environment manifest + SBOM |
| `scripts/download_datasets.sh` | Legal HF corpus download |
| `src/repro.py` | Reproducibility config (notebook/CLI parity) |
| `src/environment.py` | Preflight checks and environment verification |
| `src/drift_check.py` | 5-tier dependency drift detection |
| `src/manifest_collector.py` | Environment provenance collection |

## Security

See [SECURITY.md](SECURITY.md) for full security practices.

Key protections:
- **detect-secrets** baseline scan on every commit
- **pre-push hooks** block .env, SSH keys, model binaries
- **pip-audit** CVE scan in CI and pre-push
- **CycloneDX SBOM** generated each CI run → `logs/sbom.json`

## Reproducibility

All experiments are reproducible via:
- `uv.lock` — pinned dependency snapshot
- `src/repro.py` — seeds + deterministic flags
- `logs/environment_manifest.json` — full provenance (git SHA, hardware, freeze snapshot, SLURM job)

## Git Workflow (GitFlow)
```
main (production) ← develop (integration) ← feature/* (work)
```

After cloning: `uv run pre-commit install && uv run pre-commit install --hook-type pre-push`

---
