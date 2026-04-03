# Reducing Hallucination in Legal RAG Chatbots
### A Comparative Study of Retrieval Architectures: From Non-Neural Baselines to Transformer-Based Deep Learning
[![CI](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml/badge.svg)](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml)
- **Author:** Alex Oort Alonso, Allan Korir, and PHONG LE
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
| DataFrame / corpus scan  | polars 1.39.3 (CPU-only; **mandatory hard dependency**; imported at module top level in `src/dataset_probe.py`; always used for exact full-corpus scan via `scan_ndjson`)   |
* This is the certified baseline stack for the current repository; newer upstream stacks are intentionally deferred for now, they will be adopted only after full re-certification in this repo.
* **`BAAI/bge-m3`** is used **only in single-vector dense retrieval mode**.
* It uses **CLS pooling**, not mean pooling.
* This is defined in BAAI's published `1_Pooling/config.json`, where:
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
* The default GPU execution dtype is **bfloat16**.
* At startup, `src/environment.py` enforces these checks:
  * `transformers.__version__ == "4.39.3"`
  * `torch.cuda.get_device_capability()[0] >= 8`
  * `torch.cuda.is_bf16_supported()`
* These checks ensure the environment matches the repo's certified GPU and library assumptions before execution starts.
* `TARGET_GPU_COUNT=1` is set in `.env` to match the **single-GPU allocation** provided by SLURM.
* During preflight, `setup.sh` validates that:
  * `torch.cuda.device_count() == 1`
* This ensures the runtime environment matches the expected **single-GPU execution setup**.
* Per-model fallback to fp16/fp32 remains available if a path fails smoke tests.
* For the NLI phase, `torch.backends.cuda.matmul.allow_tf32 = True` is set as a repo-level performance optimization targeting remaining float32 matmul/convolution paths on L4; in a bfloat16-heavy pipeline its impact is limited but it is retained as an opt-in inference optimization — this can trade some FP32 numerical precision for speed and is not a semantic guarantee.
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
| Corpus scan | Polars scan_ndjson                         | 0GB GPU (CPU-only)                                                                                           | **Mandatory** exact full-corpus scan — always uses `_full_scan_with_polars()` for all 1.46M opinions; no GPU memory contention                                                                             |
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
  - `torch.backends.cuda.matmul.allow_tf32 = True` is set for the NLI phase as a repo-level performance optimization targeting remaining float32 matmul/convolution paths; in a bfloat16-heavy pipeline its impact is limited but it is retained as an opt-in inference optimization, not a semantic guarantee; state logged per phase.
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
* If fallback passage localization is required, the system logs the **citation anchor offset** for debugging.
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
  * `allow_tf32=True` (opt-in; targets remaining float32 paths; state logged)
  * `pin_memory=True`
  * configurable `num_workers`
* For SLURM-restricted environments, `num_workers=0` is available as a fallback.
* The **projected peak memory usage** is expected to remain within the **23.7GB VRAM budget**.
* **Polars** (`polars 1.39.3`) is a **mandatory hard dependency** in `src/dataset_probe.py`.
  * It is **CPU-only** and causes **zero GPU memory contention**.
  * It is imported at **module top level** — not lazily, not optionally.
  * It always performs the exact full-corpus scan via `pl.scan_ndjson` per shard.
  * It is not loaded during training, retrieval, generation, or NLI phases.
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
"q + top-10 chunks" means the user's query combined with the 10 highest-ranked retrieved evidence chunks, which together form the grounded input sent to the generator model.
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
| Split             | Capped size       | Use                                    |
|-------------------|-------------------|----------------------------------------|
| Train             | **500K–1M pairs** | Fine-tuning BGE-M3, reranker           |
| Validation        | **50K pairs**     | Hyperparameter tuning, RRF grid search |
| Test (retrieval)  | **10K–50K pairs** | Tier A evaluation                      |
| Test (generation) | **1,000 queries** | Tier B/C evaluation                    |
### Sample Size
| Component       | Capped size   | Justification                                   |
|-----------------|---------------|-------------------------------------------------|
| Training pairs  | 500K–1M       | Contrastive convergence in hours on L4/A10G     |
| Retrieval eval  | 10K–50K       | Sufficient precision; avoids multi-day NLI cost |
| Generation eval | 1,000 queries | ±2.5pp at 95% CI; stratified by circuit         |
### Evaluation Protocol
1. **Acquire LePaRD** as the core retrieval dataset.**Cap the training set** to **500K–1M pairs** for feasibility and controlled compute.
2. Train or adapt **all retrievers** using the **capped training split**, index **`bm25s`** over the **same pre-chunked payloads** used in the experiment.
3. Validate first on a **10–20% subset** of the CourtListener corpus, scale up to the full corpus **only if the subset results remain stable and promising**. On the validation set, log **Recall@k versus `nprobe`**, use those validation results to **justify the chosen IVF search parameters**.
4. Evaluation runs **sequentially over each batch of 1,000 queries**.
  * Between phases, the pipeline logs and enforces:
    * allocator diagnostics
    * CUDA stream synchronization time
    * `allow_tf32` state
    * explicit **DataLoader deletion**
    * `torch.cuda.empty_cache()`
    * `gc.collect()`
  * **Phase 1 — Dense retrieval with BGE-M3**
    * Load **BGE-M3** in **bfloat16**
    * Log **pooling flags** to W&B
    * Retrieve **top-50** passages using **1024-subword chunks**
    * Log **embedding norm distribution**
    * Save results
    * Unload model
  * **Phase 2 — Reranking with bge-reranker-v2-m3**
    * Load **`bge-reranker-v2-m3`** in **bfloat16**
    * Use:
      * **CrossEncoder**
      * **GPU by default**
      * `max_length=1024`
      * `batch_size=4`
    * Rerank the **top-50** candidates
    * Log score distribution:
      * minimum
      * mean
      * maximum
      * entropy
    * Serialize scores and ranks
    * Return the **top-10** results
    * Unload model
  * **Phase 3 — Generation with Mistral-7B-Instruct-v0.2**
    * Load **`Mistral-7B-Instruct-v0.2`** in **bfloat16**
    * Apply the **chat template**
    * Assert:
      * `max(prompt_tokens) < 32768`
    * Generate with:
      * `do_sample=False`
    * Log:
      * prompt token count using the **Mistral tokenizer**
      * completion token count
    * Save outputs
    * Unload model
  * **Phase 4 — NLI evaluation with DeBERTa-v3**
    * Load **DeBERTa-v3 NLI** in **bfloat16**
    * Use:
      * `use_fast=False`
      * `model_max_length=512`
      * repo-certified overflow windowing with:
        * `max_length=512`
        * `stride=64`
    * Log **window count per chunk distribution**
    * Tokenize in `__getitem__`
    * Use:
      * `DataCollatorWithPadding(pad_to_multiple_of=8)`
      * `allow_tf32=True` opt-in targets remaining float32 paths
      * `pin_memory=True`
      * configurable `num_workers`
      * `num_workers=0` as SLURM fallback
    * Classify each **atomic claim**
    * Aggregate **window-level logits** at the **chunk level**
    * Log:
      * window index per label
      * labels
      * confidence scores
      * claim counts
    * Delete DataLoader
    * Unload model
  * **Phase 5 — Citation checking with SQLite**
    * Use **SQLite** with:
      * `check_same_thread=False`
      * read-only mode
    * Log **citation hash**:
      * `opinion_id + anchor span`
    * If lookup returns `NULL`:
      * label as **Hard Citation Hallucination**
    * If citation is found but has **no support**:
      * label as **CitationFound_NoLocalSupport**
5. Aggregate all evaluation outputs across queries and phases.
  * Report **contradiction rate** in two forms:
    * normalized by **claim count**
    * per **1,000 tokens**
  * Report the **zero-claim rate** separately.
  * Report the **Hard Citation Hallucination rate** separately.
  * Report the **CitationFound_NoLocalSupport rate** separately.
  * Run **significance tests** on the reported metrics.
  * Log all results and diagnostics to **Weights & Biases (W&B)**.
### Significance Testing
* **Paired bootstrap** with **B = 10,000** resamples
* Report **p-values**
* Report **95% confidence intervals (CIs)**
* Report **effect size** using **Cohen's d**
* Apply **Benjamini–Hochberg FDR correction** for multiple comparisons
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
* **Sparse retrieval** uses **`bm25s`**, `bm25s` is indexed over the **same pre-chunked payloads** embedded by **BGE-M3**. "pre-chunked payloads" means the retrieval-ready text chunks created ahead of time from legal opinions and reused identically by BM25 and BGE-M3 so the architecture comparison stays fair and reproducible.**
* **Dense retrieval** uses the repo-certified stack:
  * `sentence-transformers 3.1.1`
  * `transformers 4.39.3`
* It uses **CLS pooling**, following BAAI's published:
  * `1_Pooling/config.json`
* A **runtime assertion** in `model_loader.py` prevents accidental pooling override.
* The active **pooling flags** are logged to **Weights & Biases (W&B)** once per run.
* The chunking policy is standardized as follows:
  * **1024 subwords per chunk**
  * **128-subword overlap**
  * **512 subwords per chunk for Legal-BERT**
* This is a **controlled experimental design choice**.
* It is intended to keep chunking **consistent across the pipeline**.
* **Reranker model:** `BAAI/bge-reranker-v2-m3`
  * is being loaded and run using the **CrossEncoder class provided by the `sentence-transformers` library**.
  * is smoke-tested in this repository
  * uses **bfloat16** as the primary dtype
  * runs on **GPU by default**
  * supports **CPU fallback** if GPU execution is unavailable
  * is configured with:
    * `max_length=1024`
    * `batch_size=4`
  * logs reranker **score distributions**, including:
    * minimum
    * mean
    * maximum
    * entropy
  * **Ranks are serialized** for later analysis
  * Reranking flow **top-50 → top-10**: the system first retrieves 50 likely relevant legal chunks, then reranks those candidates and keeps the 10 strongest chunks as final evidence for downstream answer generation.
* **FAISS Flat** is used for:
  * capped experiments
  * validation runs
* **FAISS IVF** is reserved for:
  * full-corpus final runs
* Before IVF search is used:
  * `index.train()` is run on an approximately **100K-vector subset**
  * `assert index.is_trained` verifies that training completed successfully
* To justify the chosen IVF settings:
  * **recall@k vs. nprobe** is logged on the validation set
* The following FAISS search parameters are logged to **W&B** for each run:
  * `nprobe`
  * `nlist`
* CPU thread settings are controlled in `setup.sh` through:
  * `OMP_NUM_THREADS`
  * `MKL_NUM_THREADS`
* **FAISS runs on CPU** in this setup.
* The pinned package version is **`faiss-cpu 1.13.2`**.
---
## Current Pipeline Status
| Stage | Status | What exists | What remains |
|-------|--------|-------------|--------------|
| Environment bootstrap | ✅ Complete | Tests passing, coverage verified, manifest generated | — |
| CourtListener download | ✅ Complete | 1,465,484 opinions · 159 shards · 7.6GB | — |
| DVC + S3 | ✅ Complete | Remote configured | `dvc push` data shards |
| CourtListener RAG prep | 🔄 In progress | JSONL with 23-field schema; dataset_probe.py v2.5.11 with mandatory Polars full-corpus scan, ProbeConfig (defines corpus evaluation for downstream legal RAG), module-level constants, contract test suite (303 tests), GateResult+ProbeReport typed contracts, STAGE3_REQUIRED_FIELDS (17 fields), GATE_REGISTRY, stratify_by minority-group-preserving stratified subsampling | Tokenizer-aware chunking (1024 subwords) + SQLite citation index |
| LePaRD acquisition | ⏳ **Priority 1** | `src/dataset_loader.py` ready | Download + DVC (cap at 500K–1M) |
| Feature/index generation | ⏳ Not started | `src/lightning_datamodule.py`, `src/split.py` | BM25 (pre-chunked payloads) + FAISS Flat (eval) / IVF (train+add, final) |
| Model training | ⏳ Not started | Architectures + compute caps specified | Training runs |
| Evaluation (Tiers A/B/C) | ⏳ Not started | Sequential loading + repo-certified overflow windowing documented | Capped eval runs |
| Experiment tracking | ⏳ Not started | `src/wandb_logger.py` implemented; W&B lazily imported; `wandb>=0.16` pinned in uv.lock | W&B run init + per-phase VRAM logging integration pending |
---
## DL System Stack
### Stage 1 — Raw Data Acquisition
* **CourtListener federal appellate subset (🔄)**
  * Contains **1,465,484 opinions**
  * Stored as **23-field JSONL** records
* **Tier C citation support**
  * Uses a local **SQLite citation index**
  * Configured with `check_same_thread=False`
  * Evaluation path is **read-only**
  * A **citation hash** is logged for every lookup
* **Iteration strategy**
  * **Fast iteration:** ~150K opinions (**10% subset**)
  * **Final runs:** full **1.46M-opinion** corpus
* **LePaRD** is **Priority 1**.
* The full dataset contains **~4 million pairs**.
* Training will be **capped at 500K–1M pairs** for feasibility.
* The dataset source is:
  * **Hugging Face**
  * **GitHub**
* `src/dataset_loader.py` is already **ready** for LePaRD ingestion.
* This repo requires **`HF_TOKEN`** on the shared cluster.
* `HF_TOKEN` is used for:
  * authenticated Hugging Face access
  * reducing resolver failures
  * reducing rate-limit failures
* The token is loaded via **`dotenv`** in:
  * `src/environment.py`
* It is then passed to every:
  * `from_pretrained()` call
* All datasets and artifacts are tracked with **DVC**.
* The DVC remote storage is **S3**.
* The configured S3 bucket is:
  * `cs1090b-hallucinationlegalragchatbots`
* The bucket region is:
  * `us-east-2`
### Stage 2 — Raw Artifact Registration
* An **immutable manifest** is generated for the dataset and pipeline state.
* The manifest records:
  * **159 shard checksums**
  * **1,465,484 total rows**
  * the full **filter-chain provenance**
  * the **`uv.lock` SHA256** value
* **Contract tests** validate the manifest on **every pipeline run**.
* An **SBOM** is also generated, covering **357 software components**.
### Stage 3 — Preprocessing + Tokenizer-Aware Chunking
* Data is processed through a **streaming CSV pipeline**.
* Preprocessing includes:
  * **HTML stripping**
  * **encoding cleanup**
  * **Westlaw boilerplate removal**
* A **stripped-down spaCy pipeline** is used with:
  * `exclude=["ner", "parser", "lemmatizer"]`
* `nlp.max_length` is set high enough to safely handle **full federal appellate opinions**.
* Chunks are built using:
  * `AutoTokenizer.from_pretrained(encoder_model)`
  * inside the **chunking loop**
* The chunking policy is standardized as:
  * **1024 subwords per chunk**
  * **128-subword overlap**
  * **512-subword chunks for Legal-BERT**
* This chunk budget is a **controlled experimental design choice** for **retriever–reranker consistency**.
* It is **not** a hard limitation of **BGE-M3**.
* Pre-chunked string payloads are:
  * **saved once**
  * **reused by both BGE-M3 and `bm25s`**
* Effective prompt token count is logged per query using the **Mistral tokenizer**.
* A runtime guard enforces prompt length:
  * `assert max(prompt_tokens) < 32768`
* If that assertion fails, the pipeline stops loudly instead of silently exceeding context limits.
* The pipeline creates **citation-aware data splits**.
* Metadata stored for each chunk includes:
  * `court_id`
  * `year`
  * `is_precedential`
  * `opinion_id`
  * `chunk_index`
* A **SQLite citation index** is built from:
  * `data/raw/cl_federal_appellate_bulk`
  * via `src/extract.py`
* **Dataset readiness probing** (`src/dataset_probe.py` v2.5.11) runs before chunking:
  * **Polars is a mandatory hard dependency** — `import polars as pl` at module top level.
  * `run_probe()` **always** calls `_full_scan_with_polars()`.
  * All probe behaviour is governed by **`ProbeConfig`** — the frozen dataclass that defines how the probe evaluates the CourtListener corpus for downstream legal RAG research.
  * **Contract tests** in `tests/test_dataset_probe.py` ensure the probe's formal structure and behavior remain intact — see Contract Test Suite below.
  * `run_probe()` returns a **typed `ProbeReport(BaseModel)`**.

#### Contract Test Suite — `tests/test_dataset_probe.py`

##### Purpose
* `tests/test_dataset_probe.py` is the **single authoritative contract test file** for `src/dataset_probe.py`.
* Its role is to ensure that the dataset probe, its constants, its `ProbeConfig`, its gates, and its typed outputs continue to satisfy the **formal structure and behavior that downstream legal RAG research infrastructure relies on**.
* The test file is organized into **over 60 named test classes** covering distinct contractual obligations. All tests run under the `pytest` + `hypothesis` framework and are enforced on every push via CI.
* **303 tests pass** as of the current version (confirmed by `uv run pytest tests/test_dataset_probe.py -m "unit or contract" -q`).

##### Contract test categories and what they protect

**Constants contracts (`TestModuleLevelConstants`, `TestProbeVersion`)**
* Assert that `PROVISIONAL_MIN_TEXT_LENGTH == 1500`, `CHUNK_SIZE_SUBWORDS == 1024`, `CHUNK_OVERLAP_SUBWORDS == 128`, `ENCODER_MODEL == "BAAI/bge-m3"`, `SPACY_MODEL == "en_core_web_sm"`, `MIN_SENTENCE_COUNT == 20` — all as integer or string literals with no type annotations.
* Assert that the source file contains these literal assignments (e.g., `"CHUNK_SIZE_SUBWORDS = 1024" in source`) — preventing silent drift where a constant changes value without the test catching it.
* Assert `PROBE_VERSION` is a string of length ≥ 3 — the instrument version recorded in every `ProbeReport`.

**`ProbeConfig` contracts (`TestProbeConfig`, `TestA11ChunkCountFormula`)**
* Assert `ProbeConfig()` is **frozen** — mutating a field raises `AttributeError` or `TypeError`.
* Assert `ProbeConfig().min_text_length == PROVISIONAL_MIN_TEXT_LENGTH` — the critical invariant that `ProbeConfig` defaults must equal module constants.
* Assert `_probe_config_to_dict(ProbeConfig())` is JSON-serializable — guarantees the config snapshot in `provenance["probe_config"]` can always be written to disk and W&B.
* Assert `ProbeConfig()` has `a11_generative_model` field — the secondary Mistral tokenizer check.
* Assert all new `ProbeConfig` fields appear in `report["provenance"]["probe_config"]` — confirms the config snapshot is complete in every `ProbeReport`.
* Assert the A11 chunk count formula (`stride = chunk_size - overlap; n_chunks = max(1, ceil((total - overlap) / stride))`) matches `CHUNK_SIZE_SUBWORDS=1024` and `CHUNK_OVERLAP_SUBWORDS=128` — regression test against chunking formula drift.

**`GateResult` contracts (`TestGateResultModel`)**
* Assert `GateResult` is importable and is a Pydantic `BaseModel`.
* Assert `GateResult` has required fields `gate` and `severity`.
* Assert `GateResult.model_dump()` is JSON-serializable.
* Assert `GateResult` is **frozen** — mutation raises `FrozenInstanceError`.

**`ProbeReport` contracts (`TestProbeReportModel`)**
* Assert `ProbeReport` is importable and is a Pydantic `BaseModel`.
* Assert `run_probe()` returns a `ProbeReport` instance — not a raw dict.
* Assert `report["gates"]` works via `__getitem__` (backward-compatible dict-style access).
* Assert `"gates" in report` works via `__contains__`.
* Assert `report.gates` attribute access works natively.
* Assert `report.summary` has `all_passed` key.
* Assert `json.dumps(report.model_dump())` succeeds — full JSON serializability.

**`GATE_REGISTRY` contracts (`TestGateRegistry`)**
* Assert registry is non-empty and iterable.
* Assert all 7 core gates are present: A7, A8, A9, A11, A12, A13, B6.
* Assert every entry has `"name"`, `"fn"` (callable), and `"severity"` keys.
* Assert A7, A8, A11, A12, A13 are `"blocking"`; A9 and B6 are `"advisory"`.
* Assert every non-tokenizer, non-spaCy gate function is callable with `(records, cfg)` and returns a dict with a `"gate"` key — live execution contract.

**`STAGE3_REQUIRED_FIELDS` contracts (`TestStage3RequiredFields`)**
* Assert `STAGE3_REQUIRED_FIELDS` is a `frozenset` and is exported.
* Assert it is a subset of `DOCUMENTED_FIELDS`.
* Assert it contains key Stage 3 fields: `id`, `court_id`, `text`, `text_length`, `text_source`, `date_filed`, `text_hash`.
* Assert `validate_schema()` reports `stage3_pass` and `stage3_missing_counts` for records missing Stage 3 fields.
* Assert `stage3_pass=True` appears in `report["provenance"]` when all 17 fields are present.

**Stratified sampling contracts (`TestStratifiedSampling`)**
* Assert `run_probe()` with `ProbeConfig(stratify_by="court_id")` does not crash.
* Assert `report["subset_n"] <= subset` after stratified subsampling.

**W&B isolation contracts (`TestLogReportToWandbIsolation`, `TestRunProbeNoLogToWandbParam`, `TestRunProbeNoInlineWandb`, `TestLogReportToWandbSingleCall`)**
* Assert `run_probe()` signature does not include a `log_to_wandb` parameter — telemetry must stay in `main()`.
* Assert `_log_report_to_wandb` is not called anywhere inside `run_probe()` — verified by inspecting the source AST.
* Assert `wandb.log` is called **exactly once** per `_log_report_to_wandb` invocation — the single consolidated log call contract.

**Lazy import contracts (`TestLazyImportBehavior`, `TestImportStyle`)**
* Assert `spacy` is NOT imported at module top level (lazy import pattern).
* Assert `transformers.AutoTokenizer` is NOT imported at module top level.
* Assert the source contains `import polars as pl` at module level (mandatory hard import).

**Full-scan and shard audit contracts (`TestFullScanCLI`, `TestShardAuditTotalRecordsDecoded`, `TestShardAuditOnePassStreaming`)**
* Assert `--full-scan` CLI flag triggers Polars path and reports `full_scan=True` in provenance.
* Assert `shard_audit["total_records_decoded"]` reflects the actual number of records loaded.
* Assert the one-pass streaming audit correctly counts parse errors and blank lines.

**Schema helper contracts (`TestValidateSchemaHelpers`, `TestValidateSchemaDocumentedFields`, `TestValidateSchemaTextLengthConsistency`)**
* Assert each composable helper (`_check_presence`, `_check_types_and_ranges`, `_check_vocabulary`, `_check_consistency`, `_check_documented_coverage`) returns dicts with expected keys.
* Assert `validate_schema()` `pass=True` when all required fields present and valid.
* Assert consistency check uses OR logic — absolute tolerance OR relative tolerance.

**Property-based tests (`TestPercentileProperty`, `TestGateA8Property`)**
* Use `hypothesis` to assert `_percentile()` satisfies monotonicity and boundary conditions over arbitrary sorted lists.
* Use `hypothesis` to assert A8 gate `pass` field is always a boolean regardless of input distribution.

##### Contract test dependency provenance
| Item | pyproject.toml | uv.lock pinned |
|------|---------------|----------------|
| `pytest>=9.0.2` | ✅ line 43 | ✅ lockfile-pinned |
| `pytest-cov>=7.0.0` | ✅ line 44 | ✅ lockfile-pinned |
| `hypothesis>=6.151.9` | ✅ line 41 | ✅ lockfile-pinned |
| `pydantic>=2.12.5` — `GateResult` + `ProbeReport` contract assertions | ✅ line 37 | ✅ 49 locked entries |
| `polars>=1.39.3` — full-scan contract assertions | ✅ line 36 | ✅ 16 locked entries |
| Python stdlib `dataclasses` — `ProbeConfig` frozen assertion | N/A | ✅ Python 3.11.9 |

#### `ProbeConfig` — Defines How the Probe Evaluates the Corpus for Downstream Legal RAG

##### What `ProbeConfig` is
* `ProbeConfig` is a **frozen dataclass** (`@dataclasses.dataclass(frozen=True)`) — the single source of truth for how `src/dataset_probe.py` evaluates the CourtListener corpus.
* Every gate threshold, subsample size, model name, quality signal pattern, and stratification strategy are controlled by `ProbeConfig` fields — no gate uses hardcoded values.
* **Frozen** means: once constructed, values cannot change. Every probe run is a reproducible measurement.
* `_probe_config_to_dict(cfg)` serializes to a JSON-safe dict (frozensets → sorted lists, tuples → lists, None preserved) — recorded in `provenance["probe_config"]` and passed to `wandb.init(config=...)`.

##### The six evaluation dimensions `ProbeConfig` controls

| Dimension | Key fields | What it evaluates |
|-----------|-----------|-------------------|
| Text viability | `min_text_length=1500`, `chunk_size_subwords=1024`, `chunk_overlap_subwords=128` | Minimum chars for meaningful chunking; chunk window for A11 |
| Source format | `a7_known_formats_pass_pct=80.0`, `a7_known_formats={"plain_text","html_with_citations"}` | % of records from formats confirmed to survive normalization |
| Citation anchor | `a12_min_pct_with_anchor=60.0`, `a9_zero_citation_pass_pct=20.0` | SQLite Tier C viability |
| Chunk count | `a11_min_median_chunks=2.0`, `encoder_model="BAAI/bge-m3"`, `a11_generative_model` | Multi-chunk splitting + Mistral context limit advisory |
| Sentence density | `min_sentence_count=20`, `a13_max_below_threshold_pct=15.0`, `spacy_model` | NLI window coverage floor |
| Entropy / quality | `b6_entropy_spot_check_tolerance=1.0`, `quality_signals_html_pattern`, `quality_signals_boilerplate_phrases` | Tokenization drift + boilerplate survival |

Plus: schema consistency (`text_length_consistency_tolerance=200`, `text_length_relative_tolerance=0.05`) and stratified sampling (`stratify_by: str | None = None`).

##### `ProbeConfig` contracts enforced by test suite
* `ProbeConfig()` is frozen — mutation raises exception (`TestProbeConfig.test_is_frozen`)
* `ProbeConfig().min_text_length == PROVISIONAL_MIN_TEXT_LENGTH` — defaults equal module constants (`TestProbeConfig.test_default_min_text_length`)
* `_probe_config_to_dict(ProbeConfig())` is JSON-serializable (`TestProbeConfig.test_is_json_serializable`)
* All new fields appear in `report["provenance"]["probe_config"]` (`TestProbeConfig.test_all_new_fields_in_provenance`)

#### Module-Level Constants — Controlled Measurement Infrastructure
| Constant | Value | Module constant → `ProbeConfig` field invariant |
|----------|-------|------------------------------------------------|
| `PROBE_VERSION` | `"2.5.11"` | N/A — instrument version |
| `PROVISIONAL_MIN_TEXT_LENGTH` | `1500` | = `ProbeConfig().min_text_length` |
| `CHUNK_SIZE_SUBWORDS` | `1024` | = `ProbeConfig().chunk_size_subwords` |
| `CHUNK_OVERLAP_SUBWORDS` | `128` | = `ProbeConfig().chunk_overlap_subwords` |
| `ENCODER_MODEL` | `"BAAI/bge-m3"` | = `ProbeConfig().encoder_model` |
| `SPACY_MODEL` | `"en_core_web_sm"` | = `ProbeConfig().spacy_model` |
| `MIN_SENTENCE_COUNT` | `20` | = `ProbeConfig().min_sentence_count` |
| `DOCUMENTED_FIELDS` | 23-field frozenset | Full schema advisory coverage |
| `KNOWN_TEXT_SOURCES` | 8-value frozenset | A7 vocabulary check |
| `MIN_REQUIRED_FIELDS` | 11-field frozenset | Blocking CI gate minimum |
| `STAGE3_REQUIRED_FIELDS` | 17-field frozenset | Stage 3 readiness gate |
| `GATE_REGISTRY` | 7-entry ordered list | Self-describing gate manifest |

#### `stratify_by` — Minority-Group-Preserving Stratified Subsampling
* `ProbeConfig.stratify_by: str | None = None` — tells the probe which record attribute to use for proportional post-scan stratified subsampling.
* Called **after** `_full_scan_with_polars()` — guarantees every stratum gets **at least 1 record**.
* `shard_audit["stratified_by"]` records provenance. Uses Python stdlib `random.Random`.

#### Mandatory Exact Full-Corpus Scan
* `import polars as pl` at module top level — no fallback.
* `run_probe()` defaults `full_scan=True` — always scans all 1,465,484 opinions via `pl.scan_ndjson()` per shard.
* `provenance["full_scan"]=True` and `provenance["polars_version"]` always recorded.

#### Stage 3 Readiness Gate — `STAGE3_REQUIRED_FIELDS` (17 fields)
| Field | Stage 3 Role |
|-------|-------------|
| `id` | Primary key for chunk metadata and SQLite citation index |
| `court_id` | Circuit-stratified evaluation; chunk metadata |
| `court_name` | Human-readable circuit label for W&B logging |
| `text` | Source text for tokenizer-aware chunking |
| `text_length` | Pre-filter: `< 1500` chars excluded before chunking |
| `text_source` | Determines preprocessing path in `row_normalizer.py` |
| `citation_count` | Fast-iteration corpus filter |
| `citation_density` | Quality signal; logged in W&B |
| `is_precedential` | Chunk metadata; weights retrieval results |
| `text_entropy` | B6 gate input; p10 used as low-entropy filter |
| `token_count` | A11 gate cross-validation input |
| `paragraph_count` | A13 sentence density proxy |
| `date_filed` | Chunk metadata; temporal filtering |
| `opinion_type` | Chunk metadata; majority/concurrence/dissent |
| `precedential_status` | Vocabulary-checked; gates chunking of non-binding opinions |
| `text_hash` | Deduplication key for FAISS and BM25 index |
| `source` | Provenance field for DVC artifact lineage |

#### Typed Contracts
* `GateResult(BaseModel)`: `extra=allow, frozen=True` — immutable; fields `gate: str`, `severity: str`.
* `ProbeReport(BaseModel)`: `extra=allow, frozen=False` — `__getitem__`/`__contains__` backward compat; `model_dump()` carries full `ProbeConfig` snapshot.
* **pydantic>=2.12.5** — pyproject.toml line 37; 49 locked entries in uv.lock.

#### Lazy Loading
| Dependency | Import strategy | Triggered by |
|------------|----------------|-------------|
| `polars` | **Hard import at module top** | Always |
| `spacy` | Lazy — `_load_spacy_nlp()` | Gate A13 only |
| `transformers.AutoTokenizer` | Lazy | Gate A11 only |
| `wandb` | Optional `try/except ImportError` | `--log-to-wandb` in `main()` only |

#### W&B Telemetry Contract
* `_log_report_to_wandb()` is **exclusively a `main()` concern** — enforced by `TestLogReportToWandbIsolation`.
* `wandb.log` called **exactly once** per invocation — enforced by `TestLogReportToWandbSingleCall`.
* `ProbeConfig` snapshot passed to `wandb.init(config=...)`.

#### W&B Metrics Logged Per Phase
| Namespace | Metric | Source |
|-----------|--------|--------|
| `probe/` | `all_passed`, `passed_count`, `failed_blocking_count`, `subset_n`, `parse_errors` | `_log_report_to_wandb` |
| `gate/A8/` | `p5`, `p10`, `p25`, `p75`, `p90`, `p95`, `mean`, `median`, `below_provisional_pct` | A8 gate result |
| `gate/A11/` | `median_chunks_per_doc`, `mean_chunks_per_doc`, `multi_chunk_pct`, `mean_token_length` | A11 gate result |
| `gate/A12/` | `pct_with_citation_anchor`, `mean_anchors_per_doc`, `field_nonzero_regex_zero_pct` | A12 gate result |
| `gate/A13/` | `median_sentences`, `below_threshold_pct`, `records_after_a8_filter` | A13 gate result |
| `gate/B6/` | `p5`, `p10`, `p25`, `p75`, `mean`, `median`, `zero_entropy_count` | B6 gate result |
| `data/` | `n_valid_samples`, `avg_token_length`, `token_length_distribution` (bar chart) | `log_dataset_stats` |
| `data/quality/` | per-signal counts (`html_remnants`, `boilerplate`, `no_citations`, …) | `log_quality_signals` |

#### W&B CLI Usage
```bash
uv run python -m src.dataset_probe \
    --data-dir data/raw/cl_federal_appellate_bulk \
    --output logs/dataset_probe_report.json \
    --log-to-wandb \
    --wandb-entity phl690-harvard-extension-schol \
    --wandb-project cs1090b \
    --wandb-name dataset_probe_v2.5.11_full
```

When integrated, the full logger tracks:
* **Per-phase GPU hours**
* **Peak CUDA memory**
* **Allocator diagnostics**
* **CUDA stream synchronization times**
* **`allow_tf32` state** for each phase
* **BGE-M3 pooling flags**
* **Embedding norm distributions**
* **Reranker score distributions**, including:
  * minimum
  * mean
  * maximum
  * entropy
* **FAISS retrieval diagnostics**, including:
  * Recall@k vs `nprobe`
  * `nprobe`
  * `nlist`
* **NLI window count distributions**
* **NLI confidence distributions**
* **Window indices per label**
* **Per-query claim counts**
* **Claims per 1,000 tokens**
* **Zero-claim rate**
* **Hard Citation Hallucination counts**
* **CitationFound_NoLocalSupport counts**
* **Citation hashes**
* **Citation anchor token offsets**
* **Prompt token counts** using the Mistral tokenizer
* **Completion token counts**
* **Gradient norms**
* **Dataset probe gate results** (A7–A13, B6) including parse error counts and full_scan provenance
### Stage 4 — Index Generation *(not started)*
* **BM25 (`bm25s`)** is indexed over the **pre-chunked payloads from Stage 3**, not over raw text.
* The **FAISS dense index** uses two modes:
  * **Flat index** for capped / validation runs
    * no index training required
  * **IVF index** for full-corpus final runs
    * `index.train()` runs on a random subset of about **100K vectors**
    * `assert index.is_trained` is checked before search
* To justify IVF settings, the pipeline logs:
  * **recall@k vs. nprobe** on the validation set
  * **nprobe** and **nlist** per run in **W&B**
* Threading controls are set in `setup.sh` with:
  * `OMP_NUM_THREADS`
  * `MKL_NUM_THREADS`
* **`BAAI/bge-reranker-v2-m3`** is used through the **`sentence-transformers` CrossEncoder** path.
* This reranker path has been **smoke-tested in the repository**.
* Reranker execution settings:
  * primary dtype: **bfloat16**
  * **GPU by default**
  * **CPU fallback available**
  * `max_length=1024`
  * `batch_size=4`
* The reranker logs score distribution statistics, including:
  * **minimum**
  * **mean**
  * **maximum**
  * **entropy**
* Reranker outputs are persisted by:
  * serializing **scores**
  * serializing **ranks**
* The reranker returns the **top 10** results after reranking.
* A **SQLite citation index** is also maintained for citation-related lookup tasks.
### Stage 5 — Model Training *(not started)*
* Training is **capped at 500K–1M pairs**.
* **Early stopping** is used to avoid unnecessary training once performance stops improving.
* **Gradient accumulation** is used to support effective larger-batch training under GPU memory constraints.
* **GPU hours** are logged to **Weights & Biases (W&B)**.
### Stage 6 — Evaluation *(not started)*
* All models use **bfloat16** as the primary dtype.
* `environment.py` asserts the following at startup:
  * `transformers.__version__ == "4.39.3"`
  * `torch.cuda.get_device_capability()[0] >= 8`
  * `torch.cuda.is_bf16_supported()`
* `TARGET_GPU_COUNT=1` is validated against:
  * `torch.cuda.device_count()`
* Models are loaded **sequentially** according to the VRAM budget plan.
* At every phase boundary, the pipeline performs cleanup and logging:
  * `torch.cuda.memory_stats()`
  * CUDA stream synchronization time
  * `allow_tf32` state
  * explicit DataLoader deletion
  * `torch.cuda.empty_cache()`
  * `gc.collect()`
* **BGE-M3**
  * pooling flags are logged to **W&B**
  * embedding norm distribution is logged
* **Reranker**
  * score distributions are logged, including:
    * minimum
    * mean
    * maximum
    * entropy
* **Mistral**
  * prompt formatting is enforced with:
    * `tokenizer.apply_chat_template(...)`
  * prompt length is guarded by:
    * `assert max(prompt_tokens) < 32768`
  * generation uses:
    * `do_sample=False`
  * per-query logging includes:
    * prompt token count
    * completion token count
* **`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`**
  * uses `use_fast=False` as the repo-certified path
  * enforces:
    * `tokenizer.model_max_length=512`
  * overflow windowing is repo-certified for this exact model/version
  * logs:
    * window count per chunk distribution
    * window index for each assigned label
  * aggregates:
    * window-level logits at the **chunk level**
  * uses:
    * `DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)`
    * `allow_tf32=True` (opt-in; targets remaining float32 matmul/convolution paths; limited impact in bfloat16-heavy pipeline; state logged)
    * `pin_memory=True`
    * configurable `num_workers`
* **SQLite citation layer**
  * logs a citation hash for every lookup
  * `NULL` is labeled:
    * **Hard Citation Hallucination**
  * found citation but no support is labeled:
    * **CitationFound_NoLocalSupport**
  * citation anchor offset is logged when fallback logic is used
* Reported metrics include:
  * contradiction rate
    * normalized by claim count
    * reported per 1K tokens
  * zero-claim rate
  * Hard Citation Hallucination rate
  * CitationFound_NoLocalSupport rate
* The projected peak VRAM usage remains within the **23.7GB** budget.
### Stage 7 — Experiment Tracking *(not started)*
* `src/wandb_logger.py` is **implemented and ready**.
* **W&B run initialization is still pending** — integration fires once LePaRD training begins.
* `wandb` is declared as **`wandb>=0.16`** in `pyproject.toml` and fully pinned in `uv.lock` (13 locked entries).
* In `src/dataset_probe.py`, `wandb` is imported inside a **`try/except ImportError`** block — degrades gracefully with `wandb = None` if unavailable.
* In `src/wandb_logger.py`, all W&B calls use **lazy imports inside each function**.
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
* `src/repro.configure()` must be the **first statement** in every notebook cell and CLI script.
* `HF_TOKEN` is required by this repository on the shared cluster.
* Add `HF_TOKEN` to `.env` **before running** the project.
* `TARGET_GPU_COUNT=1` must also be set in `.env`.
* This value must match the **SLURM single-GPU allocation**.
* All device placement must use:
  * `.to("cuda")`
  * or `.to("cuda:0")`
* Never use a **hardcoded physical GPU ordinal**.
* `uv.lock` and DVC together provide:
  * **configuration reproducibility**
  * **determinism-controlled experiments**
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
│   ├── dataset_probe.py         # Dataset readiness probe v2.5.11: ProbeConfig frozen dataclass (6 evaluation dimensions, 27 fields, injectable, JSON-serializable); module constants as controlled measurement infrastructure; mandatory Polars hard import; _full_scan_with_polars() always called; GateResult(BaseModel) frozen=True + ProbeReport(BaseModel) frozen=False (pydantic>=2.12.5); STAGE3_REQUIRED_FIELDS (17 fields); GATE_REGISTRY 7-entry manifest; stratify_by minority-group-preserving post-scan subsampling; lazy-loaded spaCy + AutoTokenizer; 5 composable schema sub-helpers; wandb optional try/except; --log-to-wandb exclusively in main()
│   ├── lightning_datamodule.py  # PyTorch DataModule; repo-certified overflow windowing; tokenized in __getitem__; DataCollatorWithPadding
│   ├── model_loader.py          # Safetensors model loader; CLS pooling assertion + W&B logging for BGE-M3
│   ├── hf_export.py             # HuggingFace Hub export
│   ├── drift_check.py           # Manifest drift detection
│   ├── wandb_logger.py          # W&B tracking: setup_wandb_auth, log_run_start, log_dataset_stats, log_quality_signals; all wandb imports lazy; wandb>=0.16 pinned in uv.lock
│   ├── exceptions.py            # PipelineError
│   └── timer.py                 # cell_timer
├── notebooks/cs1090b_HallucinationLegalRAGChatbots.ipynb
├── tests/
│   └── test_dataset_probe.py    # Single authoritative contract test file — 60+ named test classes, 303 tests; contracts: module constants, ProbeConfig frozen+serializable, GateResult frozen, ProbeReport typed+backward-compat, GATE_REGISTRY callable, STAGE3_REQUIRED_FIELDS subset+readiness, stratified sampling, W&B isolation (no log_to_wandb in run_probe), single wandb.log call, lazy imports, full-scan provenance, schema helpers, property-based (hypothesis)
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
| Layer                   | Tool                                                                                                 | Certified version |
|-------------------------|------------------------------------------------------------------------------------------------------|------------------|
| Language                | Python                                                                                               | 3.11.9 |
| Package manager         | uv                                                                                                   | 0.10.2 (repo-pinned) |
| Deep learning           | PyTorch                                                                                              | 2.0.1+cu117 (node driver: CUDA 12.8) |
| Transformers            | HuggingFace transformers                                                                             | 4.39.3 (version-asserted at startup) |
| Sentence embeddings     | sentence-transformers                                                                                | 3.1.1 |
| Dense retrieval         | BAAI/bge-m3 (CLS pooling per BAAI published config; runtime assertion + pooling flags logged to W&B) | HuggingFace — ~2.27GB (bfloat16) |
| CrossEncoder reranker   | BAAI/bge-reranker-v2-m3                                                                              | sentence-transformers CrossEncoder path smoke-tested in this repo — bfloat16; GPU ~2GB, CPU fallback; max_length=1024, batch_size=4; score distributions (min/mean/max/entropy) logged; scores serialized; top-50→top-10 |
| BM25 retrieval          | bm25s                                                                                                | 0.3.2.post1 — indexed over pre-chunked payloads (not raw text) |
| Vector search           | faiss-cpu                                                                                            | 1.13.2 — Flat for eval; IVF (index.train() + assert index.is_trained; recall@k vs nprobe logged; nprobe/nlist logged) for final corpus |
| LLM generator           | mistralai/Mistral-7B-Instruct-v0.2                                                                   | smoke-tested in repo; bfloat16; chat template applied; do_sample=False; prompt length assertion; prompt/completion token counts logged |
| Tokenizer dependency    | sentencepiece                                                                                        | 0.2.1 |
| NLI classifier          | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli                                             | smoke-tested in repo; bfloat16; use_fast=False (repo-certified); model_max_length=512; overflow windowing repo-certified; window count distribution logged; DataCollatorWithPadding pad_to_multiple_of=8; `allow_tf32=True` (opt-in; targets remaining float32 paths; state logged); pin_memory=True; window-level logits aggregated per chunk; citation hash logged |
| NLP sentence boundaries | spaCy + en_core_web_sm                                                                               | 3.8.11 / 3.8.0 (stripped, nlp.max_length set for long opinions; lazily imported — only loaded when gate A13 runs) |
| Chunking tokenizer      | AutoTokenizer (transformers)                                                                         | 1024-subword chunks, 128 overlap — design choice (512 for Legal-BERT); lazily imported in gate_a11_tokenizer_chunk_count |
| Contract test suite     | pytest + hypothesis                                                                                  | pytest>=9.0.2 (pyproject.toml line 43); hypothesis>=6.151.9 (line 41); pytest-cov>=7.0.0 (line 44); lockfile-pinned; 303 tests in tests/test_dataset_probe.py across 60+ named test classes; covers: module constants literals, ProbeConfig frozen+serializable+defaults-equal-constants, GateResult frozen Pydantic, ProbeReport typed+backward-compat, GATE_REGISTRY callable+severity, STAGE3_REQUIRED_FIELDS subset+readiness, stratified sampling, W&B isolation (no log_to_wandb in run_probe signature), single wandb.log call, lazy imports (spaCy/AutoTokenizer not at top level; polars IS at top level), full-scan provenance, schema helpers, property-based percentile+A8 monotonicity |
| Corpus evaluation config | Python stdlib dataclasses                                                                           | ProbeConfig @dataclasses.dataclass(frozen=True); 27 fields across 6 evaluation dimensions; defaults must equal module constants (enforced by TestProbeConfig + TestModuleLevelConstants); injectable; _probe_config_to_dict JSON-serializable; provenance["probe_config"] + wandb.init(config=...) |
| Measurement instrument constants | Python stdlib (module-level literals + frozensets)                                        | PROBE_VERSION="2.5.11"; PROVISIONAL_MIN_TEXT_LENGTH=1500; CHUNK_SIZE_SUBWORDS=1024; CHUNK_OVERLAP_SUBWORDS=128; ENCODER_MODEL; SPACY_MODEL; MIN_SENTENCE_COUNT=20; frozensets; GATE_REGISTRY; no type annotations |
| Stratified sampling     | Python stdlib `random`                                                                               | _stratified_reservoir_sample post-Polars-scan; min 1 per stratum; ProbeConfig.stratify_by; no pyproject.toml entry needed |
| Stage 3 readiness gate  | Python stdlib (frozenset)                                                                            | STAGE3_REQUIRED_FIELDS: 17 fields; stage3_pass + stage3_missing_counts in validate_schema() |
| Typed contracts         | pydantic                                                                                             | pydantic>=2.12.5 (pyproject.toml line 37); 49 locked entries in uv.lock; GateResult(BaseModel) frozen=True; ProbeReport(BaseModel) frozen=False; model_dump() carries ProbeConfig snapshot |
| Citation index          | SQLite (stdlib)                                                                                      | check_same_thread=False; read-only; citation hash logged; built via src/extract.py |
| DataFrame / corpus scan | polars                                                                                               | 1.39.3 — mandatory hard dependency; import at module top level; _full_scan_with_polars() always called; pl.scan_ndjson per shard; CPU-only; 16 locked entries in uv.lock; 303 tests pass |
| Experiment tracking     | W&B                                                                                                  | 0.25.1 (wandb>=0.16 in pyproject.toml; 13 locked entries in uv.lock; ProbeConfig in wandb.init; PROBE_VERSION in run name; lazy imports; optional try/except; --log-to-wandb exclusively in main()) |
| Data versioning         | DVC 3.67.0 + dvc-s3 3.3.0                                                                            | S3 remote: cs1090b-hallucinationlegalragchatbots (us-east-2) |
| Linting                 | ruff                                                                                                 | lockfile-pinned |
| Type checking           | mypy                                                                                                 | lockfile-pinned |
---
## Ethical Considerations
* All datasets used in the project are **publicly available**.
* **CourtListener** data is used under **CC BY-ND 4.0**.
* **`mistralai/Mistral-7B-Instruct-v0.2`** does **not** include built-in moderation mechanisms, according to its model card.
* Model outputs are used **strictly for retrieval research** under **academic supervision**.
* Any **PII handling** follows the **redaction practices of the original data provider**.
* The project uses **no human annotation** for hallucination measurement.
---
GPU pipeline comparing retrieval architectures (TF-IDF, CNN, LSTM, BERT bi-encoder, KG-augmented) to reduce hallucination in legal RAG chatbots.
**Hardware:** 4x NVIDIA L4/A10G GPUs | Python 3.11.9 | torch 2.0.1+cu117 | CUDA 11.7 (driver 12.8)
---
## Research-Pipeline Infrastructure

| Component | Module | Purpose |
|-----------|--------|---------|
| Reproducibility | `src/repro.py` | Seeds, deterministic flags, env loading via `.env` |
| W&B Telemetry | `src/wandb_logger.py` | `log_run_start`, `log_dataset_stats`, `log_quality_signals` |
| Manifest & Provenance | `src/manifest.py` | Shard checksums (SHA256), git SHA, `write_manifest` / `read_manifest` |
| Pipeline Orchestration | `src/pipeline.py` | `run_pipeline`, `validate_pipeline` |
| Data Audit | `scripts/audit_jsonl_nan.py` | NaN/Infinity detection, semantic repair, Polars validation, W&B provenance logging (`git_sha`, `python_version`, `polars_version`) |

### Verified functional (2026-04)
```python
from src.repro import configure
from src.wandb_logger import log_run_start, log_dataset_stats
from src.manifest import write_manifest, read_manifest
from src.pipeline import run_pipeline, validate_pipeline
```

### Data audit CLI
```bash
# Audit with schema-driven gating (2026 policy)
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk --schema-advisory --json

# Repair in parallel with Polars post-validation
uv run python scripts/audit_jsonl_nan.py --fix --parallel-repair --validate --workers 8

# Strict encoding audit (surfaces corrupt bytes)
uv run python scripts/audit_jsonl_nan.py --strict-encoding --emit-shard-ids
```

### Config precedence

`YAML (--config)` > `env vars (AUDIT_*)` > `defaults`
---

## Data Audit — `scripts/audit_jsonl_nan.py`

### Verified CLI results on real data (1,465,484 lines, 159 shards, 2026-04)

#### Default text output
```bash
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk 2>/dev/null
```
```
clean pct:            100.0000%
nan fields:           {}
verdict:              CLEAN
```

#### JSON output
```bash
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk --json 2>/dev/null
```
```json
{
  "total_lines": 1465484,
  "nan_lines": 0,
  "nonfinite_lines": 0,
  "string_sentinel_lines": 0,
  "decode_error_lines": 0,
  "nan_shards": 0,
  "total_shards": 159,
  "clean_pct": 100.0,
  "nan_fields": {},
  "gate_verdict": "CLEAN",
  "contaminated_shards": []
}
```

#### CSV report
```bash
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk --csv /tmp/audit_out.csv 2>/dev/null
```
```
field,nan_count
(empty — dataset clean)
```

#### Parallel workers
```bash
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk --workers 4 2>/dev/null
```
```
nan fields:           {}
verdict:              CLEAN
```

#### Strict encoding audit
```bash
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk --strict-encoding 2>/dev/null
```
```
nan fields:           {}
verdict:              CLEAN
```

#### Schema-driven 2026 gating
```bash
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk --schema-advisory 2>/dev/null
```
```
nan fields:           {}
verdict:              CLEAN
```

#### Dry-run repair
```bash
uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk --fix --dry-run 2>&1 | tail -2
```
```
repairing: 100%|████████████████████| 159/159 [03:51<00:00,  1.45s/shard]
[INFO] Would repair 0 lines.
```

### Gate verdicts

| Verdict | Meaning |
|---------|---------|
| `CLEAN` | No contamination — pipeline unblocked |
| `REPAIRABLE` | NaN only in advisory fields (`case_name`, `raw_text`, `cleaning_flags`) — use `--fix` |
| `HARD_FAILURE` | NaN in required schema fields — blocks Stage 3 |
| `PARSE_FAILURE` | Malformed JSON with no field mapping — manual inspection required |

### Config precedence

`YAML (--config)` > `env vars (AUDIT_*)` > `defaults`

---
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
