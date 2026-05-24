#!/usr/bin/env python3
"""
Train RotatE on FB15k-237 with custom two-stage negative sampling.

Example:
    python code/train_rotate_custom.py --strategy random
    python code/train_rotate_custom.py --strategy hard --epochs 50 --patience 10
    python code/train_rotate_custom.py --strategy mixed --device cuda

Quick smoke run:
    python code/train_rotate_custom.py --strategy hard --epochs 1 --batch-size 64 \\
        --pool-size 16 --limit-batches 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from negative_sampling import build_sampling_context  # noqa: E402
from score_candidates import best_device  # noqa: E402
from select_candidates import SelectionStrategy, sample_training_negatives_batch  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filtered_mrr(
    model: torch.nn.Module,
    mapped_triples: torch.LongTensor,
    additional_filter_triples: list[torch.LongTensor],
    device: str,
    *,
    batch_size: int = 256,
    limit_triples: int | None = None,
) -> float:
    from pykeen.evaluation import RankBasedEvaluator

    model.eval()
    if limit_triples is not None:
        mapped_triples = mapped_triples[:limit_triples]
    evaluator = RankBasedEvaluator(filtered=True)
    result = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        additional_filter_triples=additional_filter_triples,
        device=device,
        batch_size=batch_size,
    )
    return float(result.get_metric("both.realistic.inverse_harmonic_mean_rank"))


def compute_loss(
    model: torch.nn.Module,
    positive_batch: torch.LongTensor,
    negative_batch: np.ndarray,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    pos_scores = model.score_hrt(positive_batch).view(-1)
    neg_tensor = torch.as_tensor(negative_batch, dtype=torch.long, device=positive_batch.device)

    if neg_tensor.ndim == 2:
        neg_scores = model.score_hrt(neg_tensor).view(-1)
        return loss_fn(pos_scores, neg_scores)

    if neg_tensor.ndim != 3:
        raise ValueError(f"Expected negative_batch with 2 or 3 dims, got {neg_tensor.ndim}.")

    batch_size, num_negs, _ = neg_tensor.shape
    flat_neg = neg_tensor.reshape(batch_size * num_negs, 3)
    neg_scores = model.score_hrt(flat_neg).view(batch_size, num_negs)
    return loss_fn.process_slcwa_scores(pos_scores, neg_scores)


def train_one_epoch(
    model: torch.nn.Module,
    train_triples: np.ndarray,
    *,
    strategy: SelectionStrategy,
    batch_size: int,
    pool_size: int,
    num_negs: int,
    num_entities: int,
    bernoulli,
    triple_index,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    rng: np.random.Generator,
    limit_batches: int | None = None,
    hard_fraction: float = 0.5,
) -> float:
    model.train()
    order = rng.permutation(len(train_triples))
    running_loss = 0.0
    num_batches = 0

    for start in range(0, len(order), batch_size):
        if limit_batches is not None and num_batches >= limit_batches:
            break

        batch_idx = order[start : start + batch_size]
        positive_np = train_triples[batch_idx]
        positive_batch = torch.as_tensor(positive_np, dtype=torch.long, device=device)

        negative_batch = sample_training_negatives_batch(
            positive_np,
            strategy,
            model,
            pool_size=pool_size,
            k=num_negs,
            num_entities=num_entities,
            bernoulli=bernoulli,
            triple_index=triple_index,
            rng=rng,
            device=device,
            hard_fraction=hard_fraction,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, positive_batch, negative_batch, loss_fn)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu())
        num_batches += 1

    return running_loss / max(num_batches, 1)


def save_curves(out_dir: Path, losses: list[float], val_mrrs: list[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping curves.png")
        return

    n_plots = int(bool(losses)) + int(bool(val_mrrs))
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if losses:
        axes[idx].plot(losses, linewidth=1.5)
        axes[idx].set_title("Training loss")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss")
        axes[idx].grid(True, linestyle="--", alpha=0.5)
        idx += 1

    if val_mrrs:
        axes[idx].plot(val_mrrs, linewidth=1.5, color="tab:orange")
        axes[idx].set_title("Validation MRR")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Filtered MRR")
        axes[idx].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    plt.close(fig)


def main() -> int:
    try:
        from pykeen.datasets import FB15k237
        from pykeen.losses import MarginRankingLoss
        from pykeen.models import RotatE
    except ImportError:
        print("Install PyKEEN: pip install -r requirements.txt", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Train RotatE with custom negative sampling.")
    p.add_argument("--strategy", choices=[s.value for s in SelectionStrategy], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--pool-size", type=int, default=32, help="Candidate pool size n.")
    p.add_argument("--num-negs", type=int, default=1, help="Selected negatives k per positive.")
    p.add_argument("--margin", type=float, default=9.0)
    p.add_argument("--hard-fraction", type=float, default=0.5,
                   help="Fraction of hard negatives in mixed strategy (default 0.5 = 50/50).")
    p.add_argument("--device", default=None)
    p.add_argument("--limit-batches", type=int, default=None, help="Debug: cap batches per epoch.")
    p.add_argument("--limit-val-eval", type=int, default=None, help="Debug: cap validation triples per epoch.")
    args = p.parse_args()

    args.device = args.device or best_device()
    if args.device == "mps":
        print("Note: RotatE requires complex ops — falling back to CPU.")
        args.device = "cpu"

    if args.num_negs > args.pool_size:
        print("--num-negs cannot exceed --pool-size", file=sys.stderr)
        return 1

    strategy = SelectionStrategy(args.strategy)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.chdir(REPO_ROOT)
    if strategy is SelectionStrategy.MIXED:
        hard_pct = round(args.hard_fraction * 100)
        run_label = f"RotatE_mixed_{hard_pct}_{100 - hard_pct}"
    else:
        run_label = f"RotatE_{strategy.value}"
    out_dir = REPO_ROOT / "artifacts" / "custom" / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FB15k-237 ...")
    dataset = FB15k237()
    triple_index, bernoulli, num_entities = build_sampling_context(dataset.training)
    train_triples = dataset.training.mapped_triples.cpu().numpy()

    print(
        f"Training RotatE | strategy={strategy.value} | d={args.dim} | bs={args.batch_size} | "
        f"pool={args.pool_size} | num_negs={args.num_negs} | lr={args.lr} | "
        f"max_epochs={args.epochs} | patience={args.patience} | device={args.device}"
        + (f" | hard_fraction={args.hard_fraction}" if strategy is SelectionStrategy.MIXED else "")
    )

    model = RotatE(triples_factory=dataset.training, embedding_dim=args.dim).to(args.device)
    loss_fn = MarginRankingLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    val_filter = [dataset.training.mapped_triples]
    test_filter = [dataset.training.mapped_triples, dataset.validation.mapped_triples]

    losses: list[float] = []
    val_mrrs: list[float] = []
    best_val_mrr = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(
            model,
            train_triples,
            strategy=strategy,
            batch_size=args.batch_size,
            pool_size=args.pool_size,
            num_negs=args.num_negs,
            num_entities=num_entities,
            bernoulli=bernoulli,
            triple_index=triple_index,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=args.device,
            rng=rng,
            limit_batches=args.limit_batches,
            hard_fraction=args.hard_fraction,
        )
        val_mrr = filtered_mrr(
            model,
            dataset.validation.mapped_triples,
            val_filter,
            args.device,
            limit_triples=args.limit_val_eval,
        )

        losses.append(epoch_loss)
        val_mrrs.append(val_mrr)
        print(f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | val MRR={val_mrr:.4f}")

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val MRR={best_val_mrr:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    from pykeen.evaluation import RankBasedEvaluator

    model.eval()
    evaluator = RankBasedEvaluator(filtered=True)
    test_result = evaluator.evaluate(
        model=model,
        mapped_triples=(
            dataset.testing.mapped_triples[: args.limit_val_eval]
            if args.limit_val_eval is not None
            else dataset.testing.mapped_triples
        ),
        additional_filter_triples=test_filter,
        device=args.device,
        batch_size=256,
    )
    test_mrr = float(test_result.get_metric("both.realistic.inverse_harmonic_mean_rank"))
    hits1 = float(test_result.get_metric("both.realistic.hits_at_1"))
    hits3 = float(test_result.get_metric("both.realistic.hits_at_3"))
    hits10 = float(test_result.get_metric("both.realistic.hits_at_10"))

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        header = (
            f"model=RotatE strategy={strategy.value} dim={args.dim} epochs={args.epochs} "
            f"bs={args.batch_size} pool={args.pool_size} num_negs={args.num_negs} "
            f"lr={args.lr} margin={args.margin} patience={args.patience} "
            + (f"hard_fraction={args.hard_fraction} " if strategy is SelectionStrategy.MIXED else "")
            + f"optimizer=adam device={args.device} seed={args.seed} best_epoch={best_epoch}\n\n"
        )
        f.write(header)
        lines = {
            "val_mrr_best": best_val_mrr,
            "test_mrr": test_mrr,
            "test_hits_at_1": hits1,
            "test_hits_at_3": hits3,
            "test_hits_at_10": hits10,
        }
        print("\n=== Test metrics ===")
        for key, value in lines.items():
            line = f"{key}: {value:.4f}"
            print(line)
            f.write(line + "\n")

    torch.save(model, out_dir / "trained_model.pkl")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"losses": losses, "val_mrrs": val_mrrs, "best_epoch": best_epoch}, f, indent=2)

    save_curves(out_dir, losses, val_mrrs)
    print(f"\nArtifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())