# LePaRD Dataset Summary

**Dataset**: [rmahari/LePaRD](https://huggingface.co/datasets/rmahari/LePaRD) on HuggingFace  
**Paper**: [LePaRD: A Large-Scale Dataset of Judicial Citations to Precedent (ACL 2024)](https://aclanthology.org/2024.acl-long.532/)  
**Source code**: [github.com/rmahari/LePaRD](https://github.com/rmahari/LePaRD)  
**Size**: ~22.7 million rows (citation pairs)

---

## What Is LePaRD?

LePaRD (Legal Passage Retrieval Dataset) is a dataset of **22.7 million passage-level citation pairs** extracted from U.S. federal court opinions. It was built from the **Case Law Access Project (CAP)** — Harvard's digitized archive of ~1.7 million federal opinions.

Each row represents a single instance of a judge **quoting** a passage from a prior case. It is *not* document-level — if one opinion quotes three different passages from another case, that produces three separate rows.

## Schema

| Column | Type | Description | Example |
|---|---|---|---|
| `dest_id` | int | CAP case ID of the **citing** opinion | `8969838` |
| `dest_name` | string | Case name of the citing opinion | `Singh v. George Washington University` |
| `dest_cite` | string | Full citation string of the citing opinion | `Singh v. George Washington University, 368 F. Supp. 2d 58 (2005)` |
| `dest_court` | string | Court of the citing opinion | `United States District Court for the District of Columbia` |
| `dest_date` | string | Date of the citing opinion (YYYY-MM-DD) | `2005-03-22` |
| `source_id` | int | CAP case ID of the **cited** opinion | `11800677` |
| `source_name` | string | Case name of the cited opinion | `Kaltenberger v. Ohio College of Podiatric Medicine` |
| `source_cite` | string | Full citation string of the cited opinion | `Kaltenberger v. Ohio College of Podiatric Medicine, 162 F.3d 432 (1998)` |
| `source_court` | string | Court of the cited opinion | `United States Court of Appeals for the Sixth Circuit` |
| `source_date` | string | Date of the cited opinion (YYYY-MM-DD) | `1998-12-09` |
| `passage_id` | string | Unique ID for the quoted passage: `<source_id>_<n>` | `11800677_13` |
| `quote` | string | The **exact quoted text** from the cited opinion | `on the basis of a disability.` |
| `destination_context` | string | The **paragraph** in the citing opinion surrounding the quotation | (avg ~724 chars) |

### Key relationships

- **`passage_id`** always follows the pattern `<source_id>_<sequence_number>`. The prefix is the `source_id` of the cited case; the suffix is the nth passage quoted from that case. Verified 100% match in a 10K sample.
- **`dest_cite`** and **`source_cite`** contain standard legal citation strings with embedded volume/reporter/page (e.g., `368 F. Supp. 2d 58`). These are the key to merging with CourtListener.

## Key Statistics (from 10K sample of first 50K rows)

| Metric | Value |
|---|---|
| Total rows | 22,744,882 |
| Unique citing cases (dest) | ~7,460 per 10K sample |
| Unique cited cases (source) | ~839 per 10K sample |
| Quote length | Mean 136 chars, median 111, range 21-4,283 |
| Context length | Mean 724 chars, median 458, range 5-4,563 |
| Date range (citing) | 1893-2018 |
| Date range (cited) | 1828-2010 |
| Federal appellate (citing side) | ~46% |
| Federal appellate (cited side) | ~60% |
| Supreme Court (cited side) | ~33% |
| Unique courts | 222 total |

## How to Load It

```python
from datasets import load_dataset

# Streaming (low memory, good for sampling)
ds = load_dataset("rmahari/LePaRD", split="train", streaming=True)
for row in ds:
    print(row["dest_cite"], "→", row["quote"][:80])
    break

# Full download (requires ~30GB RAM + disk; takes ~10 min)
ds = load_dataset("rmahari/LePaRD", split="train")
ds.save_to_disk("lepard_data/")  # saves as Arrow files for fast reload

# Reload from disk
from datasets import load_from_disk
ds = load_from_disk("lepard_data/")
```

The full dataset is saved locally at `lepard_eda/lepard_data/` (59 Arrow shards).

---

## How LePaRD Relates to CourtListener

### The problem

LePaRD was built from **CAP (Case Law Access Project)** data. The `dest_id` and `source_id` are **CAP case IDs**, not CourtListener opinion IDs. CourtListener has its own separate ID space. There is no shared ID and no published crosswalk.

However, in March 2024, Free Law Project merged **all ~9 million CAP cases into CourtListener**. So the underlying case data is almost certainly present in both — the challenge is linking them.

### The merge strategy: citation string matching

Both datasets share **standard legal citation strings** — this is the bridge.

**LePaRD side**: `source_cite` contains strings like `"Kaltenberger v. Ohio College of Podiatric Medicine, 162 F.3d 432 (1998)"`, which embed a `(volume, reporter, page)` tuple: `(162, "F.3d", 432)`.

**CourtListener side**: The bulk `citations.csv` file maps `(volume, reporter, page)` to a `cluster_id`. A cluster groups all opinions for a case (majority, dissent, concurrence).

```
LePaRD source_cite
    → parse with eyecite → (volume, reporter, page)
    → JOIN CourtListener citations.csv ON (volume, reporter, page)
    → get cluster_id
    → JOIN opinion-clusters.csv ON cluster_id → case metadata
    → JOIN opinions on cluster_id → full opinion text
```

### What you need to download

From CourtListener's quarterly bulk data dump ([S3 bucket](https://www.courtlistener.com/help/api/bulk-data/)):

1. **`citations-<date>.csv.bz2`** — the lookup table with columns:
   ```
   id, volume, reporter, page, type, cluster_id, date_created, date_modified
   ```

2. **`opinion-clusters-<date>.csv.bz2`** — case-level metadata with columns:
   ```
   id (= cluster_id), case_name, date_filed, docket_id,
   precedential_status, filepath_json_harvard, ...
   ```

These are much smaller than the full opinion text corpus and are all you need for the join.

### Tools

- **[`eyecite`](https://github.com/freelawproject/eyecite)** (Free Law Project) — parses legal citation strings into structured `(volume, reporter, page)` tuples.
- **[`reporters-db`](https://github.com/freelawproject/reporters-db)** (Free Law Project) — normalizes reporter name variants (e.g., `"F. Supp. 2d"` vs `"F.Supp.2d"`) to canonical forms. Important for making the join work reliably.

### Example

```python
from eyecite import get_citations

text = "162 F.3d 432"
cites = get_citations(text)
# → FullCaseCitation: volume=162, reporter="F.3d", page=432

# Then look up (162, "F.3d", 432) in CourtListener's citations.csv
# → cluster_id → full opinion text
```

### Pitfalls

1. **Reporter normalization** is the hardest part. LePaRD and CourtListener may use different string variants for the same reporter. Use `reporters-db` to canonicalize both sides before joining.
2. **Court coverage mismatch**: LePaRD covers all federal courts (Supreme Court, appellate, district). Our project only uses federal appellate. ~33% of LePaRD's cited cases are Supreme Court and won't appear in our filtered corpus.
3. **One cluster, multiple opinions**: A CourtListener `cluster_id` may group majority + dissent + concurrence opinions. You may need to pick the right one (usually the majority).
4. **Older citations** (1800s) may use archaic reporter names that don't parse cleanly.
5. **No one has published this crosswalk before** — we would be the first to do this specific LePaRD-to-CourtListener join.

---

## How We Plan to Use It

1. **Tier A retrieval evaluation**: Given `destination_context` as a query, can our retriever (BM25 / BGE-M3 / Hybrid) find the correct passage from the source opinion in the CourtListener corpus? Scored by recall@k, MRR, NDCG@10.

2. **Training data**: Fine-tune the dense retriever so it learns that `destination_context` → `quote` is a relevant match.

The merge is necessary because our retrieval corpus is indexed by CourtListener IDs, but LePaRD's "answer key" uses CAP IDs. Without the merge, we can't score whether the retriever found the right document.
