import gc
import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

corpus = [
    json.loads(line)["text"]
    for line in Path("data/processed/baseline/corpus_chunks.jsonl")
    .read_text()
    .splitlines()[:3000]
]

m = SentenceTransformer("BAAI/bge-m3", device="cuda")
m.max_seq_length = 512  # conservative; chunks have median ~300 tokens

for bs in [16, 32, 64, 128]:
    gc.collect()
    torch.cuda.empty_cache()
    try:
        t0 = time.perf_counter()
        m.encode(
            corpus[:2000],
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dt = time.perf_counter() - t0
        print(f"batch={bs} max_seq=512: {2000 / dt:.1f} chunks/sec  ({dt:.2f}s)")
    except torch.cuda.OutOfMemoryError:
        print(f"batch={bs} max_seq=512: OOM")
        gc.collect()
        torch.cuda.empty_cache()
