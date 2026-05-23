#!/usr/bin/env python3
"""Smoke test: generate candidates, score with RotatE, rank by hardness."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from negative_sampling import build_sampling_context, generate_candidates_bernoulli  # noqa: E402
from score_candidates import (  # noqa: E402
    DEFAULT_ROTATE_CHECKPOINT,
    best_device,
    load_rotate_model,
    rank_candidates,
    score_triples,
)


def _label(factory, idx: int) -> str:
    return factory.entity_id_to_label[idx]


def _relation_label(factory, idx: int) -> str:
    return factory.relation_id_to_label[idx]


def _format_triple(factory, triple: tuple[int, int, int]) -> str:
    h, r, t = triple
    return f"({_label(factory, h)}, {_relation_label(factory, r)}, {_label(factory, t)})"


def _print_ranked(title: str, ranked, factory, limit: int = 5) -> None:
    print(title)
    for item in ranked[:limit]:
        print(f"  score={item.score:7.3f}  {_format_triple(factory, item.triple)}")


def main() -> int:
    try:
        from pykeen.datasets import FB15k237
    except ImportError:
        print("Install PyKEEN: pip install -r requirements.txt", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Score and rank negative candidates with RotatE.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_ROTATE_CHECKPOINT)
    p.add_argument("--n", type=int, default=32, help="Candidates per positive triple.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu | cuda (auto-detected if omitted).")
    args = p.parse_args()
    args.device = args.device or best_device()
    if args.device == "mps":
        args.device = "cpu"

    os.chdir(REPO_ROOT)
    rng = np.random.default_rng(args.seed)

    print("Loading FB15k-237 and RotatE checkpoint ...")
    training = FB15k237().training
    triple_index, bernoulli, num_entities = build_sampling_context(training)
    model = load_rotate_model(args.checkpoint, device=args.device)

    mapped = training.mapped_triples.cpu().numpy()
    demo_idx = 0
    for i in range(len(mapped)):
        h, r, t = map(int, mapped[i])
        if _relation_label(training, r) == "/people/person/place_of_birth":
            demo_idx = i
            break

    h, r, t = map(int, mapped[demo_idx])
    positive = (h, r, t)

    side, candidates = generate_candidates_bernoulli(
        h,
        r,
        t,
        n=args.n,
        num_entities=num_entities,
        bernoulli=bernoulli,
        triple_index=triple_index,
        rng=rng,
    )

    positive_score = float(score_triples(model, [positive], device=args.device)[0])
    candidate_scores = score_triples(model, candidates, device=args.device)
    ranked = rank_candidates(candidates, candidate_scores)

    print(f"\n=== Demo triple ({side.value} corruption, n={args.n}) ===")
    print(f"Positive score={positive_score:7.3f}  {_format_triple(training, positive)}")
    print(f"Device: {args.device}")

    _print_ranked("\nHardest negatives (highest scores):", ranked, training)
    _print_ranked("\nEasiest negatives (lowest scores):", list(reversed(ranked)), training)

    assert len(candidate_scores) == len(candidates)
    assert np.all(np.isfinite(candidate_scores))
    ranked_scores = [item.score for item in ranked]
    assert ranked_scores == sorted(ranked_scores, reverse=True)
    assert len(set(item.triple for item in ranked)) == len(candidates)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
