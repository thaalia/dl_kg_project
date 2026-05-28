#!/usr/bin/env python3
"""
Qualitative comparison of trained KGE checkpoints on a handful of test triples.

For each sampled test triple (h, r, t) we compute, on every checkpoint:
  - the filtered rank of the gold tail when predicting (h, r, ?)
  - the filtered rank of the gold head when predicting (?, r, t)
  - the top-K predicted entities on both sides (filtering known true triples)

The output is a single human-readable Markdown report saved under
``artifacts/qualitative.md`` (or wherever ``--output`` points). This is meant
to power the qualitative analysis section of the report.

Usage:
    python code/qualitative_examples.py
    python code/qualitative_examples.py --num-triples 5 --top-k 10
    python code/qualitative_examples.py \
        --checkpoints artifacts/baseline/pykeen_RotatE/trained_model.pkl \
                      artifacts/custom/RotatE_hard/trained_model.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

DEFAULT_CHECKPOINTS = [
    ARTIFACTS_DIR / "baseline" / "pykeen_TransE" / "trained_model.pkl",
    ARTIFACTS_DIR / "baseline" / "pykeen_RotatE" / "trained_model.pkl",
    ARTIFACTS_DIR / "custom" / "RotatE_random" / "trained_model.pkl",
    ARTIFACTS_DIR / "custom" / "RotatE_hard" / "trained_model.pkl",
    ARTIFACTS_DIR / "custom" / "RotatE_mixed_50_50" / "trained_model.pkl",
]


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "cpu"  # MPS lacks complex ops needed by RotatE
    return "cpu"


def _short_label(label: str | None, ent_id: int) -> str:
    """FB15k-237 entity labels are Freebase MIDs like /m/02h40lc — keep them as-is."""
    if label is None or label == "":
        return f"#{ent_id}"
    return label


def _relation_label(label: str | None, rel_id: int) -> str:
    if label is None or label == "":
        return f"r{rel_id}"
    return label


def _build_known_set(*triple_arrays: torch.Tensor) -> set[tuple[int, int, int]]:
    known: set[tuple[int, int, int]] = set()
    for arr in triple_arrays:
        for row in arr.cpu().numpy():
            known.add((int(row[0]), int(row[1]), int(row[2])))
    return known


def _filtered_top_k_tail(
    model: torch.nn.Module,
    h: int,
    r: int,
    gold_t: int,
    known: set[tuple[int, int, int]],
    num_entities: int,
    top_k: int,
    device: str,
) -> tuple[int, list[tuple[int, float]]]:
    """Return (filtered_rank_of_gold, [(ent_id, score)] top_k after filtering)."""
    hr_batch = torch.as_tensor([[h, r]], dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model.score_t(hr_batch).squeeze(0).cpu().numpy()  # (num_entities,)

    # Filter out known triples (other than the gold target itself).
    masked_scores = scores.copy()
    for ent in range(num_entities):
        if ent == gold_t:
            continue
        if (h, r, ent) in known:
            masked_scores[ent] = -np.inf

    # Filtered rank of the gold tail (1-based).
    gold_score = masked_scores[gold_t]
    rank = int((masked_scores > gold_score).sum()) + 1

    # Top-K predictions (descending) from the filtered scores.
    order = np.argsort(-masked_scores)
    top_k_list = [(int(idx), float(masked_scores[idx])) for idx in order[:top_k]]
    return rank, top_k_list


def _filtered_top_k_head(
    model: torch.nn.Module,
    r: int,
    t: int,
    gold_h: int,
    known: set[tuple[int, int, int]],
    num_entities: int,
    top_k: int,
    device: str,
) -> tuple[int, list[tuple[int, float]]]:
    rt_batch = torch.as_tensor([[r, t]], dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model.score_h(rt_batch).squeeze(0).cpu().numpy()  # (num_entities,)

    masked_scores = scores.copy()
    for ent in range(num_entities):
        if ent == gold_h:
            continue
        if (ent, r, t) in known:
            masked_scores[ent] = -np.inf

    gold_score = masked_scores[gold_h]
    rank = int((masked_scores > gold_score).sum()) + 1

    order = np.argsort(-masked_scores)
    top_k_list = [(int(idx), float(masked_scores[idx])) for idx in order[:top_k]]
    return rank, top_k_list


def _format_top_list(
    top_list: list[tuple[int, float]],
    gold_id: int,
    id_to_label: dict[int, str],
) -> str:
    parts = []
    for ent, score in top_list:
        label = _short_label(id_to_label.get(ent), ent)
        marker = " ←gold" if ent == gold_id else ""
        parts.append(f"  - `{label}` (score={score:.3f}){marker}")
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Qualitative side-by-side predictions.")
    p.add_argument("--checkpoints", nargs="*", type=Path, default=None,
                   help="Paths to trained_model.pkl files. Defaults to the curated set.")
    p.add_argument("--num-triples", type=int, default=5,
                   help="How many test triples to sample.")
    p.add_argument("--top-k", type=int, default=10,
                   help="How many top predictions to show per side.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for triple sampling.")
    p.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "qualitative.md")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    try:
        from pykeen.datasets import FB15k237
    except ImportError:
        print("Install PyKEEN: pip install -r requirements.txt", file=sys.stderr)
        return 1

    device = args.device or best_device()
    print(f"Device: {device}")

    checkpoints = args.checkpoints or DEFAULT_CHECKPOINTS
    available = [p for p in checkpoints if p.exists()]
    missing = [p for p in checkpoints if not p.exists()]
    if missing:
        print("Skipping missing checkpoints:", *missing, sep="\n  ")
    if not available:
        print("No checkpoints available; nothing to do.", file=sys.stderr)
        return 1

    print("Loading FB15k-237 ...")
    dataset = FB15k237()
    train = dataset.training.mapped_triples
    val = dataset.validation.mapped_triples
    test = dataset.testing.mapped_triples
    num_entities = int(dataset.training.num_entities)

    entity_id_to_label: dict[int, str] = {
        int(idx): label for label, idx in dataset.training.entity_to_id.items()
    }
    relation_id_to_label: dict[int, str] = {
        int(idx): label for label, idx in dataset.training.relation_to_id.items()
    }

    known = _build_known_set(train, val, test)

    rng = np.random.default_rng(args.seed)
    sampled_idx = rng.choice(test.shape[0], size=args.num_triples, replace=False)
    sampled_triples = test[torch.as_tensor(sampled_idx, dtype=torch.long)]

    # Preload all models.
    models = []
    for ckpt in available:
        print(f"Loading {ckpt.relative_to(REPO_ROOT)} ...")
        m = torch.load(ckpt.resolve(), map_location=device, weights_only=False)
        m = m.to(device)
        m.eval()
        models.append((ckpt.relative_to(REPO_ROOT), m))

    out_lines: list[str] = [
        "# Qualitative predictions on FB15k-237",
        "",
        f"_Sampled {args.num_triples} test triples (seed={args.seed}). "
        f"Top-{args.top_k} candidates per side, filtered against the train+val+test triples._",
        "",
    ]

    for triple_idx, triple in enumerate(sampled_triples, start=1):
        h, r, t = (int(x) for x in triple)
        h_label = _short_label(entity_id_to_label.get(h), h)
        t_label = _short_label(entity_id_to_label.get(t), t)
        r_label = _relation_label(relation_id_to_label.get(r), r)

        out_lines.append(f"## Triple {triple_idx}: ({h_label}, {r_label}, {t_label})")
        out_lines.append("")

        for model_label, model in models:
            tail_rank, tail_top = _filtered_top_k_tail(
                model, h, r, t, known, num_entities, args.top_k, device
            )
            head_rank, head_top = _filtered_top_k_head(
                model, r, t, h, known, num_entities, args.top_k, device
            )

            out_lines.append(f"### {model_label}")
            out_lines.append(
                f"- **Tail prediction**: filtered rank of gold tail "
                f"`{t_label}` = **{tail_rank}**"
            )
            out_lines.append(_format_top_list(tail_top, t, entity_id_to_label))
            out_lines.append(
                f"- **Head prediction**: filtered rank of gold head "
                f"`{h_label}` = **{head_rank}**"
            )
            out_lines.append(_format_top_list(head_top, h, entity_id_to_label))
            out_lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nQualitative report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
