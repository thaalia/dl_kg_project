#!/usr/bin/env python3
"""
Fine-grained (sliced) evaluation of trained KGE checkpoints on FB15k-237.

For each model we compute filtered MRR / Hits@{1,3,10} on the test set:
  - globally
  - per relation-frequency bucket (rare / medium / frequent)
  - per head-entity out-degree bucket (low / mid / high)
  - per tail-entity in-degree bucket (low / mid / high)
  - per prediction side (head corruption vs tail corruption)

Buckets are computed once from the TRAINING split only, then frozen and reused
across all runs so the comparison is apples-to-apples.

Usage:
    # Evaluate a single checkpoint
    python code/evaluate_slices.py --checkpoint artifacts/custom/RotatE_random/trained_model.pkl

    # Evaluate all checkpoints found under artifacts/
    python code/evaluate_slices.py --all

    # Smoke test (cap test triples for a quick check)
    python code/evaluate_slices.py \
        --checkpoint artifacts/custom/RotatE_random/trained_model.pkl --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
BUCKETS_PATH = ARTIFACTS_DIR / "slice_buckets.json"


# ---------------------------------------------------------------------------
# Bucket definition (computed once on training split, then cached on disk)
# ---------------------------------------------------------------------------

def _tertile_thresholds(values: np.ndarray) -> tuple[float, float]:
    """Return (lo, hi) split points that put the values into 3 roughly equal buckets."""
    lo, hi = np.quantile(values, [1 / 3, 2 / 3])
    return float(lo), float(hi)


def _bucket_label(value: float, lo: float, hi: float) -> str:
    if value <= lo:
        return "low"
    if value <= hi:
        return "mid"
    return "high"


def compute_buckets(train_triples: np.ndarray) -> dict:
    """Compute relation / head-degree / tail-degree buckets from training only."""
    heads = train_triples[:, 0]
    rels = train_triples[:, 1]
    tails = train_triples[:, 2]

    rel_counts = Counter(rels.tolist())
    out_degree = Counter(heads.tolist())
    in_degree = Counter(tails.tolist())

    # Aggregate per *unique* relation/entity to define quantile thresholds.
    rel_freq_array = np.array(sorted(rel_counts.values()), dtype=np.int64)
    out_deg_array = np.array(sorted(out_degree.values()), dtype=np.int64)
    in_deg_array = np.array(sorted(in_degree.values()), dtype=np.int64)

    rel_lo, rel_hi = _tertile_thresholds(rel_freq_array)
    out_lo, out_hi = _tertile_thresholds(out_deg_array)
    in_lo, in_hi = _tertile_thresholds(in_deg_array)

    relation_bucket = {
        int(rel): _bucket_label(float(count), rel_lo, rel_hi)
        for rel, count in rel_counts.items()
    }
    out_degree_bucket = {
        int(ent): _bucket_label(float(count), out_lo, out_hi)
        for ent, count in out_degree.items()
    }
    in_degree_bucket = {
        int(ent): _bucket_label(float(count), in_lo, in_hi)
        for ent, count in in_degree.items()
    }

    return {
        "thresholds": {
            "relation_frequency": {"low_max": rel_lo, "mid_max": rel_hi},
            "head_out_degree": {"low_max": out_lo, "mid_max": out_hi},
            "tail_in_degree": {"low_max": in_lo, "mid_max": in_hi},
        },
        "relation_bucket": relation_bucket,
        "head_out_degree_bucket": out_degree_bucket,
        "tail_in_degree_bucket": in_degree_bucket,
    }


def load_or_compute_buckets(train_triples: np.ndarray) -> dict:
    if BUCKETS_PATH.exists():
        with BUCKETS_PATH.open() as f:
            cached = json.load(f)
        # JSON serialises int keys as strings — convert them back.
        for key in ("relation_bucket", "head_out_degree_bucket", "tail_in_degree_bucket"):
            cached[key] = {int(k): v for k, v in cached[key].items()}
        return cached

    print(f"Computing slice buckets from training split ...")
    buckets = compute_buckets(train_triples)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with BUCKETS_PATH.open("w") as f:
        json.dump(buckets, f, indent=2)
    print(f"Slice buckets cached at {BUCKETS_PATH}")
    return buckets


# ---------------------------------------------------------------------------
# Slice evaluation
# ---------------------------------------------------------------------------

def _evaluate_subset(
    model: torch.nn.Module,
    test_triples: torch.LongTensor,
    additional_filter_triples: list[torch.LongTensor],
    device: str,
    batch_size: int = 256,
) -> dict[str, float]:
    """Run a filtered RankBasedEvaluator on a subset of test triples."""
    from pykeen.evaluation import RankBasedEvaluator

    evaluator = RankBasedEvaluator(filtered=True)
    result = evaluator.evaluate(
        model=model,
        mapped_triples=test_triples,
        additional_filter_triples=additional_filter_triples,
        device=device,
        batch_size=batch_size,
        use_tqdm=False,
    )
    return {
        "n": int(test_triples.shape[0]),
        "mrr": float(result.get_metric("both.realistic.inverse_harmonic_mean_rank")),
        "hits_at_1": float(result.get_metric("both.realistic.hits_at_1")),
        "hits_at_3": float(result.get_metric("both.realistic.hits_at_3")),
        "hits_at_10": float(result.get_metric("both.realistic.hits_at_10")),
        "mrr_head": float(result.get_metric("head.realistic.inverse_harmonic_mean_rank")),
        "hits_at_10_head": float(result.get_metric("head.realistic.hits_at_10")),
        "mrr_tail": float(result.get_metric("tail.realistic.inverse_harmonic_mean_rank")),
        "hits_at_10_tail": float(result.get_metric("tail.realistic.hits_at_10")),
    }


def slice_indices(
    test_triples: np.ndarray,
    buckets: dict,
    axis: str,
) -> dict[str, np.ndarray]:
    """Return {bucket_label: row_indices into test_triples} for the requested axis."""
    if axis == "relation_frequency":
        key = "relation_bucket"
        col = 1
    elif axis == "head_degree":
        key = "head_out_degree_bucket"
        col = 0
    elif axis == "tail_degree":
        key = "tail_in_degree_bucket"
        col = 2
    else:
        raise ValueError(f"Unknown slice axis: {axis}")

    lookup = buckets[key]
    labels = np.array(
        [lookup.get(int(value), "unknown") for value in test_triples[:, col]],
        dtype=object,
    )
    return {
        bucket: np.flatnonzero(labels == bucket)
        for bucket in ("low", "mid", "high")
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
    limit: int | None = None,
    batch_size: int = 256,
) -> dict:
    """Compute global + sliced metrics for one trained checkpoint."""
    from pykeen.datasets import FB15k237

    print(f"\n=== Slicing {checkpoint_path.relative_to(REPO_ROOT)} ===")

    print("Loading FB15k-237 ...")
    dataset = FB15k237()
    train_triples_np = dataset.training.mapped_triples.cpu().numpy()
    buckets = load_or_compute_buckets(train_triples_np)

    print(f"Loading model from {checkpoint_path} ...")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    test_mapped = dataset.testing.mapped_triples
    if limit is not None:
        test_mapped = test_mapped[:limit]
        print(f"  (limited to {limit} test triples for smoke check)")
    test_np = test_mapped.cpu().numpy()

    additional = [dataset.training.mapped_triples, dataset.validation.mapped_triples]

    print(f"Evaluating global ({test_mapped.shape[0]} triples) ...")
    output: dict = {
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "num_test_triples": int(test_mapped.shape[0]),
        "global": _evaluate_subset(model, test_mapped, additional, device, batch_size),
    }

    for axis in ("relation_frequency", "head_degree", "tail_degree"):
        print(f"Evaluating axis: {axis} ...")
        index_map = slice_indices(test_np, buckets, axis)
        output[axis] = {}
        for bucket_label, row_idx in index_map.items():
            if row_idx.size == 0:
                output[axis][bucket_label] = {"n": 0}
                continue
            subset = test_mapped[torch.as_tensor(row_idx, dtype=torch.long)]
            output[axis][bucket_label] = _evaluate_subset(
                model, subset, additional, device, batch_size
            )

    out_path = checkpoint_path.parent / "slices.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Slice metrics saved to {out_path}")

    # Pretty-print key numbers
    g = output["global"]
    print(
        f"  global: MRR={g['mrr']:.4f}  H@10={g['hits_at_10']:.4f}  "
        f"head-MRR={g['mrr_head']:.4f}  tail-MRR={g['mrr_tail']:.4f}"
    )
    for axis in ("relation_frequency", "head_degree", "tail_degree"):
        parts = []
        for bucket_label in ("low", "mid", "high"):
            entry = output[axis][bucket_label]
            if entry.get("n", 0) == 0:
                parts.append(f"{bucket_label}:n=0")
            else:
                parts.append(f"{bucket_label}:MRR={entry['mrr']:.3f}(n={entry['n']})")
        print(f"  {axis}: " + " | ".join(parts))

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_all_checkpoints() -> list[Path]:
    """Scan artifacts/ for trained_model.pkl files (excludes the *_old backup)."""
    found: list[Path] = []
    for path in sorted(ARTIFACTS_DIR.rglob("trained_model.pkl")):
        if "baseline_old_" in path.as_posix():
            continue
        found.append(path)
    return found


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(description="Sliced evaluation of trained KGE models.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Single model checkpoint to evaluate.")
    p.add_argument("--all", action="store_true",
                   help="Evaluate every trained_model.pkl found under artifacts/.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of test triples (smoke test).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None,
                   help="Device override: cpu | cuda | mps. Auto if omitted.")
    args = p.parse_args()

    if not args.checkpoint and not args.all:
        p.error("Provide --checkpoint or --all.")

    device = args.device or best_device()
    # RotatE complex ops are unsupported on MPS; downgrade silently.
    if device == "mps":
        print("Note: RotatE complex ops not supported on MPS — falling back to CPU.")
        device = "cpu"

    checkpoints: Iterable[Path]
    if args.all:
        checkpoints = _find_all_checkpoints()
        if not checkpoints:
            print("No trained_model.pkl files found under artifacts/.", file=sys.stderr)
            return 1
        print(f"Found {len(checkpoints)} checkpoint(s): {[str(c.relative_to(REPO_ROOT)) for c in checkpoints]}")
    else:
        checkpoints = [args.checkpoint]

    for ckpt in checkpoints:
        if not ckpt.exists():
            print(f"Skip (missing): {ckpt}", file=sys.stderr)
            continue
        try:
            evaluate_checkpoint(ckpt, device=device, limit=args.limit, batch_size=args.batch_size)
        except Exception as exc:
            print(f"FAILED on {ckpt}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
