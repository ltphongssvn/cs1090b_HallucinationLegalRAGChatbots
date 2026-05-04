# scripts/finetune_reranker.py
"""Fine-tune BAAI/bge-reranker-v2-m3 on cleaned LePaRD hard negatives.

Industry-standard cross-encoder reranker fine-tuning per BAAI guidance + 4Huiter SoICT 2024 (Vietnamese legal, +23% MRR via semi-hard negatives) + LegalDuet.

Approach
--------
- Pointwise binary classification: (query, doc) -> {0, 1}
- Pos pair from gold; N neg pairs from RRF rank 2-100
- BCE loss on the reranker logit (single-class head from XLM-RoBERTa-large)
- AdamW, cosine LR schedule, fp16 mixed precision
- Saves model + tokenizer to --output-dir

Why pointwise BCE (not pairwise / listwise)
-------------------------------------------
bge-reranker-v2-m3 has a single-logit classification head. BAAI\'s own
fine-tuning examples (FlagEmbedding finetune/) use pointwise BCE with
hard negatives. Same approach used by 4Huiter Vietnamese-legal team.

Training data format (input)
----------------------------
JSONL produced by scripts/mine_hard_negatives.py:
    {query, pos: [pos_text], neg: [neg_text_1, neg_text_2, ...]}

Stack
-----
Stays on certified transformers==4.41.2. No FlagEmbedding install.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
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
BASE_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_LENGTH = 1024
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 0


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("finetune_reranker")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[finetune_reranker] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_training_pairs(path: Path) -> tuple[list[list[str]], list[int]]:
    """Expand training JSONL into flat (pair, label) lists.

    Each row {query, pos: [...], neg: [...]} -> 1 pos pair + len(neg) neg pairs.
    Pos pairs labeled 1, neg pairs labeled 0.
    """
    pairs: list[list[str]] = []
    labels: list[int] = []
    for r in _iter_jsonl(Path(path)):
        if not r.get("pos"):
            continue
        q = r["query"]
        for pt in r["pos"]:
            pairs.append([q, pt])
            labels.append(1)
        for nt in r.get("neg", []):
            pairs.append([q, nt])
            labels.append(0)
    return pairs, labels


def _git_sha() -> str:
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
    output_dir: Path,
    base_model: str = BASE_MODEL,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_length: int = DEFAULT_MAX_LENGTH,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Fine-tune bge-reranker-v2-m3 on hard-negative pairs (pointwise BCE)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    if world_size > 1:
        # torchrun sets LOCAL_RANK; bind each process to its own GPU
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

    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if device.startswith("cuda") else "cpu"
    )

    logger.info("=" * 60)
    logger.info(
        f"Fine-tune {base_model} (rank={rank}/{world_size}, device={device_name})"
    )
    logger.info("=" * 60)
    logger.info(f"  train_path  : {train_path}")
    logger.info(f"  val_path    : {val_path}")
    logger.info(f"  output_dir  : {output_dir}")
    logger.info(f"  lr={lr}  epochs={epochs}  batch={batch_size}  "
                f"grad_accum={grad_accum}  max_length={max_length}")

    # Load
    train_pairs, train_labels = _load_training_pairs(Path(train_path))
    logger.info(f"  train pairs : {len(train_pairs):,} ({sum(train_labels):,} pos)")

    val_pairs: list[list[str]] = []
    val_labels: list[int] = []
    if val_path is not None and Path(val_path).is_file():
        val_pairs, val_labels = _load_training_pairs(Path(val_path))
        logger.info(f"  val pairs   : {len(val_pairs):,} ({sum(val_labels):,} pos)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, torch_dtype=torch.float32,
    ).to(device)
    # Gradient checkpointing — recompute activations during backward pass to
    # fit XLM-RoBERTa-large (568M params) on L4 22GB at max_length=1024.
    # Required because OOM observed at batch_size=8 max_length=1024 without it.
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        # Disable use_cache when checkpointing (mutually exclusive)
        model.config.use_cache = False
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )

    class PairDataset(Dataset):
        def __init__(self, pairs: list[list[str]], labels: list[int]):
            self.pairs = pairs
            self.labels = labels

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> tuple[list[str], int]:
            return self.pairs[idx], self.labels[idx]

    def collate(batch: list[tuple[list[str], int]]):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        enc = tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=max_length,
        )
        return enc, labels

    train_ds = PairDataset(train_pairs, train_labels)
    val_ds = PairDataset(val_pairs, val_labels) if val_pairs else None

    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                     rank=rank, shuffle=True, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   collate_fn=collate, num_workers=2, pin_memory=True)
        if val_ds is not None else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    t0 = time.perf_counter()
    global_step = 0
    log_interval = max(1, total_steps // 50)
    train_loss_sum = 0.0
    train_loss_count = 0

    for epoch in range(epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()
        for step, (enc, labels) in enumerate(train_loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            if scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(**enc).logits.view(-1)
                    loss = bce(logits, labels) / grad_accum
                scaler.scale(loss).backward()
            else:
                logits = model(**enc).logits.view(-1)
                loss = bce(logits, labels) / grad_accum
                loss.backward()
            train_loss_sum += float(loss.item()) * grad_accum
            train_loss_count += 1
            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % log_interval == 0 and rank == 0:
                    avg = train_loss_sum / max(train_loss_count, 1)
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        f"    epoch {epoch+1} step {global_step}/{total_steps}  "
                        f"loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                        f"elapsed={elapsed:.0f}s"
                    )
                    train_loss_sum = 0.0
                    train_loss_count = 0

        # Validation
        if val_loader is not None and rank == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.inference_mode():
                for enc, labels in val_loader:
                    enc = {k: v.to(device) for k, v in enc.items()}
                    labels = labels.to(device)
                    logits = model(**enc).logits.view(-1).float()
                    val_loss += float(bce(logits, labels).item()) * len(labels)
                    val_correct += int(((logits > 0).float() == labels).sum().item())
                    val_total += len(labels)
            logger.info(
                f"  epoch {epoch+1} VAL  loss={val_loss/max(val_total,1):.4f}  "
                f"acc={val_correct/max(val_total,1):.4f}  n={val_total:,}"
            )

    train_seconds = time.perf_counter() - t0

    # Save model + tokenizer (only rank 0)
    if rank == 0:
        save_model = model.module if world_size > 1 else model
        save_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        summary = {
            "schema_version": SCHEMA_VERSION,
            "base_model": base_model,
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "max_length": max_length,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "seed": seed,
            "world_size": world_size,
            "device": device,
            "device_name": device_name,
            "train_seconds": round(train_seconds, 3),
            "git_sha": _git_sha(),
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        logger.info(f"  saved model -> {output_dir}")
        return summary
    return {}


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Fine-tune bge-reranker-v2-m3 on legal hard-negative pairs.",
    )
    ap.add_argument("--train-path", type=Path, required=True)
    ap.add_argument("--val-path", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=str, default=BASE_MODEL)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    ap.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[finetune_reranker] DRY RUN  base={args.base_model}  "
            f"git_sha={_git_sha()}  args={vars(args)}"
        )
        sys.exit(0)
    main(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        rank=args.rank,
        world_size=args.world_size,
    )
