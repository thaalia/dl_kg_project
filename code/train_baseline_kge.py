#!/usr/bin/env python3
"""
FB15k-237 baseline: TransE / RotatE via PyKEEN.

Install:  pip install -r requirements.txt
Run from repo root:
    python code/train_baseline_kge.py --model TransE
    python code/train_baseline_kge.py --model RotatE

Defaults are tuned for a fast preliminary run (15 epochs, bs=1024, patience=3).
For full runs: add --epochs 50 --batch_size 512 --patience 10

Saves:
    artifacts/baseline/<Model>_summary.txt   -- filtered metrics (MRR, Hits@k)
    artifacts/baseline/<Model>_curves.png    -- training loss + validation MRR curves
    artifacts/baseline/pykeen_<Model>/       -- full PyKEEN result folder
    artifacts/baseline/dataset_stats.json    -- relation/degree statistics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent


def best_device() -> str:
    """Pick the fastest available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def dataset_stats(triples_factory) -> dict:
    """Relation-frequency and entity degree stats from a TriplesFactory."""
    triples = triples_factory.mapped_triples.numpy()  # (N, 3): h, r, t
    heads, rels, tails = triples[:, 0], triples[:, 1], triples[:, 2]

    rel_freq = Counter(rels.tolist())
    out_deg = Counter(heads.tolist())
    in_deg = Counter(tails.tolist())

    return {
        "num_triples": int(len(triples)),
        "num_entities": int(triples_factory.num_entities),
        "num_relations": int(triples_factory.num_relations),
        "top10_relations_by_freq": rel_freq.most_common(10),
        "mean_out_degree": round(sum(out_deg.values()) / max(len(out_deg), 1), 2),
        "mean_in_degree": round(sum(in_deg.values()) / max(len(in_deg), 1), 2),
        "max_out_degree": int(max(out_deg.values())) if out_deg else 0,
        "max_in_degree": int(max(in_deg.values())) if in_deg else 0,
    }


def main() -> int:
    try:
        from pykeen.pipeline import pipeline
        from pykeen.datasets import FB15k237
    except ImportError:
        print("Install PyKEEN:  pip install pykeen torch", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Train TransE or RotatE on FB15k-237.")
    p.add_argument("--model", choices=["TransE", "RotatE"], default="TransE")
    p.add_argument("--epochs", type=int, default=15,
                   help="Maximum training epochs (early stopping may cut short).")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=128, help="Embedding dimension d.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stopping patience: epochs without val MRR improvement.")
    p.add_argument("--device", default=None,
                   help="Device override: cpu | cuda | mps. Auto-detected if omitted.")
    args = p.parse_args()
    args.device = args.device or best_device()
    # RotatE uses complex-valued embeddings; MPS lacks complex norm support
    if args.model == "RotatE" and args.device == "mps":
        print("Note: MPS does not support complex ops required by RotatE — falling back to CPU.")
        args.device = "cpu"

    out_dir = REPO_ROOT / "artifacts" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset + statistics ---
    print("Loading FB15k-237 ...")
    try:
        dataset = FB15k237()
    except Exception as e:
        print(f"Could not load FB15k-237: {e}", file=sys.stderr)
        return 1

    stats = dataset_stats(dataset.training)
    print("\n=== Dataset statistics (training split) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    stats_path = out_dir / "dataset_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"  -> saved to {stats_path}\n")

    # --- Training ---
    print(
        f"Training {args.model} | d={args.dim} | bs={args.batch_size} | "
        f"lr={args.lr} | max_epochs={args.epochs} | patience={args.patience} | "
        f"optimizer=adam | device={args.device}"
    )

    result = pipeline(
        model=args.model,
        dataset=dataset,
        model_kwargs={"embedding_dim": args.dim},
        training_kwargs={"num_epochs": args.epochs, "batch_size": args.batch_size},
        optimizer=optim.Adam,
        optimizer_kwargs={"lr": args.lr},
        stopper="early",
        stopper_kwargs={
            "frequency": 1,
            "patience": args.patience,
            "metric": "mean_reciprocal_rank",
            "larger_is_better": True,
        },
        random_seed=args.seed,
        device=args.device,
    )

    # --- Save metrics ---
    out_file = out_dir / f"{args.model}_summary.txt"
    with out_file.open("w", encoding="utf-8") as f:
        header = (
            f"model={args.model} dim={args.dim} epochs={args.epochs} "
            f"bs={args.batch_size} lr={args.lr} patience={args.patience} "
            f"optimizer=adam device={args.device} seed={args.seed}\n\n"
        )
        f.write(header)
        print("\n=== Test metrics ===")
        try:
            metrics = result.metric_results.to_flat_dict()
            for metric, val in sorted(metrics.items()):
                line = f"{metric}: {val:.4f}" if isinstance(val, float) else f"{metric}: {val}"
                print(line)
                f.write(line + "\n")
        except Exception:
            f.write(repr(result))

    if hasattr(result, "save_to_directory"):
        result.save_to_directory(str(out_dir / f"pykeen_{args.model}"))

    # --- Plots ---
    _save_plots(result, args.model, out_dir)

    print(f"\nSummary written to {out_file}")
    return 0


def _save_plots(result, model_name: str, out_dir: Path) -> None:
    """Save training-loss and validation-MRR curves as a single PNG."""
    losses = _extract_losses(result)
    val_mrrs = _extract_val_mrr(result)

    if not losses and not val_mrrs:
        print("No loss/MRR history found — skipping plots.")
        return

    n_plots = int(bool(losses)) + int(bool(val_mrrs))
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    if losses:
        axes[ax_idx].plot(losses, linewidth=1.5)
        axes[ax_idx].set_title(f"{model_name} — Training loss")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].grid(True, linestyle="--", alpha=0.5)
        ax_idx += 1

    if val_mrrs:
        axes[ax_idx].plot(val_mrrs, linewidth=1.5, color="tab:orange")
        axes[ax_idx].set_title(f"{model_name} — Validation MRR")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Filtered MRR")
        axes[ax_idx].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_path = out_dir / f"{model_name}_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Curves saved to {plot_path}")


def _extract_losses(result) -> list[float]:
    """Pull per-epoch training losses out of a PyKEEN PipelineResult."""
    # PyKEEN >= 1.9 stores losses_per_epoch on the training loop
    try:
        return list(result.losses)
    except Exception:
        pass
    try:
        return list(result.training_loop.losses_per_epoch)
    except Exception:
        pass
    return []


def _extract_val_mrr(result) -> list[float]:
    """Pull per-epoch validation MRR out of a PyKEEN PipelineResult."""
    try:
        # stopper keeps a list of results; each has a metric value
        history = result.stopper.results
        return [r for r in history if r is not None]
    except Exception:
        pass
    try:
        # some versions expose metric history directly
        return list(result.metric_results_per_epoch)
    except Exception:
        pass
    return []


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    raise SystemExit(main())
