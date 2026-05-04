# scripts/parade_aggregator.py
"""PARADE-Transformer passage representation aggregator (Li et al. SIGIR 2020).

Architecture
------------
1. Frozen reranker encoder produces [CLS] embedding per (query, passage) pair
2. PARADE aggregator: small Transformer over passage [CLS] vectors + scoring head
3. Cluster relevance = aggregator output

Mathematically:
    h_i = Encoder([CLS] + query + [SEP] + passage_i + [SEP])  # frozen reranker
    H = stack([h_1, ..., h_N])  # (batch, N_passages, hidden)
    z = TransformerAggregator(H)  # (batch, N_passages, hidden)
    score = Linear(z[:, 0])  # take [CLS] output, project to scalar

Why PARADE
----------
- Outperforms BERT-MaxP by 9% nDCG on TREC Robust04 (legal-document-like)
- Beats Longformer 1.5x faster (Li et al. TOIS 2023, ACL 2024)
- Industry standard for long legal docs (used in COLIEE, MORES+)

Stack
-----
Stays on certified transformers==4.41.2. Aggregator is pure PyTorch nn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "1.0.0"
DEFAULT_AGG_HIDDEN_SIZE = 1024  # XLM-RoBERTa-large hidden = 1024
DEFAULT_N_AGG_LAYERS = 2
DEFAULT_N_HEADS = 8
DEFAULT_MAX_PASSAGES = 8
DEFAULT_LR = 5e-5
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 8
DEFAULT_MAX_LENGTH = 512
DEFAULT_SEED = 0

# Reranker base used to extract per-passage [CLS] vectors
ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("parade_aggregator")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[parade_aggregator] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def build_aggregator(
    *,
    hidden_size: int = DEFAULT_AGG_HIDDEN_SIZE,
    n_layers: int = DEFAULT_N_AGG_LAYERS,
    n_heads: int = DEFAULT_N_HEADS,
    max_passages: int = DEFAULT_MAX_PASSAGES,
    dropout: float = 0.1,
):
    """Build the PARADE-Transformer aggregator module.

    Input:  (batch, n_passages, hidden_size) — passage [CLS] vectors
    Output: (batch,) — relevance score per (query, document) example

    The aggregator prepends a learnable [AGG] token (mirrors BERT [CLS]
    pattern), runs N Transformer encoder layers, then projects the [AGG]
    output to a scalar.
    """
    import torch
    import torch.nn as nn

    class PARADETransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden_size
            # Learnable positional embeddings + [AGG] token
            self.agg_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.agg_token, std=0.02)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_passages + 1, hidden_size))
            nn.init.normal_(self.pos_emb, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.score_head = nn.Linear(hidden_size, 1)

        def forward(self, passage_cls: "torch.Tensor") -> "torch.Tensor":
            # passage_cls: (batch, n_passages, hidden)
            batch, n_passages, _ = passage_cls.shape
            agg = self.agg_token.expand(batch, -1, -1)  # (batch, 1, hidden)
            x = torch.cat([agg, passage_cls], dim=1)    # (batch, 1+n, hidden)
            x = x + self.pos_emb[:, : x.shape[1], :]
            x = self.transformer(x)
            agg_out = x[:, 0, :]                        # (batch, hidden)
            score = self.score_head(agg_out).squeeze(-1)  # (batch,)
            return score

    return PARADETransformer()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _seed_all(seed: int) -> None:
    import random as _r
    _r.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main(
    *,
    train_path: Path,
    val_path: Path | None,
    encoder_dir: Path,
    output_dir: Path,
    max_passages: int = DEFAULT_MAX_PASSAGES,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Train PARADE aggregator on top of (frozen) fine-tuned reranker encoder."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    if world_size > 1:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)

    logger.info("=" * 60)
    logger.info(f"Train PARADE aggregator over {encoder_dir}")
    logger.info(f"  rank={rank}/{world_size}  device={device}")
    logger.info(f"  max_passages={max_passages}  hidden_size=auto  n_layers={DEFAULT_N_AGG_LAYERS}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    encoder = AutoModel.from_pretrained(encoder_dir).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    hidden_size = encoder.config.hidden_size

    aggregator = build_aggregator(
        hidden_size=hidden_size,
        n_layers=DEFAULT_N_AGG_LAYERS,
        n_heads=DEFAULT_N_HEADS,
        max_passages=max_passages,
    ).to(device)
    if world_size > 1:
        aggregator = torch.nn.parallel.DistributedDataParallel(
            aggregator, device_ids=[local_rank], output_device=local_rank,
        )

    # Load training rows: query, pos passages list, neg passages list
    # We treat each (query, pos_passages) and (query, neg_passages_sample) as
    # a (label=1, label=0) example, where the aggregator scores the document
    # as a whole. Since hard_negatives.jsonl has per-cluster pos/neg, we use
    # them as the document — pos = document of pos_cluster, neg = document of
    # one neg_cluster. Each gets passage-level [CLS] vectors via encoder.
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        rows = []
        for r in _iter_jsonl(Path(path)):
            if not r.get("pos") or not r.get("neg"):
                continue
            rows.append(r)
        return rows

    train_rows = _load_rows(Path(train_path))
    val_rows = _load_rows(Path(val_path)) if val_path else []
    logger.info(f"  train rows: {len(train_rows):,}")
    logger.info(f"  val rows  : {len(val_rows):,}")

    def _passages_from_doc(doc_text: str, max_p: int = max_passages, p_len: int = max_length) -> list[str]:
        """Split a document into passages by character chunking. Replace with
        sentence-boundary chunking later if needed."""
        if not doc_text:
            return [""]
        words = doc_text.split()
        passage_word_len = max(1, len(words) // max_p)
        out = []
        for i in range(max_p):
            start = i * passage_word_len
            end = start + passage_word_len if i < max_p - 1 else len(words)
            chunk = " ".join(words[start:end])
            if chunk:
                out.append(chunk)
            if len(out) >= max_p:
                break
        return out[:max_p] if out else [""]

    def _encode_passages(query: str, passages: list[str]) -> "torch.Tensor":
        """Encode each (query, passage) pair → [CLS] vector. Pad to max_passages."""
        with torch.no_grad():
            pairs = [[query, p] for p in passages]
            enc = tokenizer(pairs, padding=True, truncation=True,
                            return_tensors="pt", max_length=max_length).to(device)
            out = encoder(**enc).last_hidden_state[:, 0, :]  # (n_pas, hidden)
            n = out.shape[0]
            if n < max_passages:
                pad = torch.zeros(max_passages - n, hidden_size, device=device)
                out = torch.cat([out, pad], dim=0)
            return out[:max_passages]

    class PARADEDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            # Flatten: each row → 1 pos example + 1 neg example
            self.examples: list[tuple[str, str, int]] = []
            for r in rows:
                q = r["query"]
                self.examples.append((q, r["pos"][0], 1))
                self.examples.append((q, r["neg"][0], 0))

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return self.examples[i]

    def collate(batch):
        qs = [b[0] for b in batch]
        docs = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        # Encode each (query, passages) → (batch, max_passages, hidden)
        embs = []
        for q, doc in zip(qs, docs):
            passages = _passages_from_doc(doc)
            embs.append(_encode_passages(q, passages))
        embs_tensor = torch.stack(embs, dim=0)
        return embs_tensor, labels

    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            PARADEDataset(train_rows), num_replicas=world_size,
            rank=rank, shuffle=True, seed=seed,
        )

    train_ds = PARADEDataset(train_rows)
    val_ds = PARADEDataset(val_rows) if val_rows else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   collate_fn=collate, num_workers=0)
        if val_ds is not None else None
    )

    optimizer = torch.optim.AdamW(aggregator.parameters(), lr=lr, weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()

    # --- Resume from latest epoch checkpoint if present ---
    start_epoch = 0
    ckpts = sorted(output_dir.glob("parade_aggregator.epoch*.pt"))
    if ckpts:
        latest = ckpts[-1]
        # Filename format: parade_aggregator.epoch{N}.pt → parse N
        try:
            done_epoch = int(latest.stem.split("epoch")[-1])
            state = torch.load(latest, map_location=device)
            inner = aggregator.module if world_size > 1 else aggregator
            inner.load_state_dict(state)
            start_epoch = done_epoch + 1
            if rank == 0:
                logger.info(f"  RESUMING from {latest.name} -> start_epoch={start_epoch}")
        except Exception as e:
            if rank == 0:
                logger.info(f"  could not resume from {latest}: {e}; starting fresh")
            start_epoch = 0

    t0 = time.perf_counter()
    global_step = 0
    log_interval = max(1, len(train_loader) // 50)

    for epoch in range(start_epoch, epochs):
        aggregator.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()
        for step, (embs, labels) in enumerate(train_loader):
            labels = labels.to(device)
            scores = aggregator(embs)
            loss = bce(scores, labels) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % log_interval == 0 and rank == 0:
                    logger.info(
                        f"    epoch {epoch+1} step {global_step}  "
                        f"loss={float(loss.item())*grad_accum:.4f}  "
                        f"elapsed={time.perf_counter()-t0:.0f}s"
                    )

        # Save per-epoch checkpoint (rank 0 only)
        if rank == 0:
            inner = aggregator.module if world_size > 1 else aggregator
            ckpt_path = output_dir / f"parade_aggregator.epoch{epoch}.pt"
            torch.save(inner.state_dict(), ckpt_path)
            logger.info(f"  saved checkpoint -> {ckpt_path}")

        if val_loader is not None and rank == 0:
            aggregator.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.inference_mode():
                for embs, labels in val_loader:
                    labels = labels.to(device)
                    scores = aggregator(embs)
                    v_loss += float(bce(scores, labels).item()) * len(labels)
                    v_correct += int(((scores > 0).float() == labels).sum().item())
                    v_total += len(labels)
            logger.info(
                f"  epoch {epoch+1} VAL loss={v_loss/max(v_total,1):.4f}  "
                f"acc={v_correct/max(v_total,1):.4f}  n={v_total}"
            )

    train_seconds = time.perf_counter() - t0

    if rank == 0:
        save_agg = aggregator.module if world_size > 1 else aggregator
        agg_path = output_dir / "parade_aggregator.pt"
        torch.save(save_agg.state_dict(), agg_path)

        summary = {
            "schema_version": SCHEMA_VERSION,
            "encoder_dir": str(encoder_dir),
            "hidden_size": hidden_size,
            "n_agg_layers": DEFAULT_N_AGG_LAYERS,
            "n_heads": DEFAULT_N_HEADS,
            "max_passages": max_passages,
            "max_length": max_length,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "n_train_rows": len(train_rows),
            "n_val_rows": len(val_rows),
            "train_seconds": round(train_seconds, 3),
            "seed": seed,
            "git_sha": _git_sha(),
        }
        (output_dir / "parade_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        logger.info(f"  saved -> {agg_path}")
        return summary
    return {}


def _build_arg_parser() -> argparse.ArgumentParser:
    import os
    ap = argparse.ArgumentParser(
        description="Train PARADE-Transformer aggregator on top of fine-tuned reranker.",
    )
    ap.add_argument("--train-path", type=Path, required=True)
    ap.add_argument("--val-path", type=Path, default=None)
    ap.add_argument("--encoder-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-passages", type=int, default=DEFAULT_MAX_PASSAGES)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    ap.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(
        train_path=args.train_path,
        val_path=args.val_path,
        encoder_dir=args.encoder_dir,
        output_dir=args.output_dir,
        max_passages=args.max_passages,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        seed=args.seed,
        rank=args.rank,
        world_size=args.world_size,
    )
