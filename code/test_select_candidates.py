#!/usr/bin/env python3
"""Smoke test: compare random, hard, and mixed negative selection."""
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
from select_candidates import SelectionStrategy, select_negatives  # noqa: E402


def _label(factory, idx: int) -> str:
    return factory.entity_id_to_label[idx]


def _relation_label(factory, idx: int) -> str:
    return factory.relation_id_to_label[idx]


def _format_triple(factory, triple: tuple[int, int, int]) -> str:
    h, r, t = triple
    return f"({_label(factory, h)}, {_relation_label(factory, r)}, {_label(factory, t)})"


def _score_map(candidates, scores) -> dict[tuple[int, int, int], float]:
    return {triple: float(score) for triple, score in zip(candidates, scores)}


def _print_selection(title: str, selected, score_by_triple, factory) -> None:
    print(title)
    for triple in selected:
        print(f"  score={score_by_triple[triple]:7.3f}  {_format_triple(factory, triple)}")


def main() -> int:
    try:
        from pykeen.datasets import FB15k237
    except ImportError:
        print("Install PyKEEN: pip install -r requirements.txt", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Smoke test for negative selection strategies.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_ROTATE_CHECKPOINT)
    p.add_argument("--pool-size", type=int, default=32, help="Candidate pool size n.")
    p.add_argument("--k", type=int, default=4, help="Negatives to select for training.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    args.device = args.device or best_device()
    if args.device == "mps":
        args.device = "cpu"

    if args.k > args.pool_size:
        print("--k cannot exceed --pool-size", file=sys.stderr)
        return 1

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
        n=args.pool_size,
        num_entities=num_entities,
        bernoulli=bernoulli,
        triple_index=triple_index,
        rng=rng,
    )
    scores = score_triples(model, candidates, device=args.device)
    ranked = rank_candidates(candidates, scores)
    score_by_triple = _score_map(candidates, scores)
    top_k_hard = {item.triple for item in ranked[: args.k]}

    print(f"\n=== Demo triple ({side.value} corruption, pool={args.pool_size}, k={args.k}) ===")
    print(f"Positive: {_format_triple(training, positive)}")

    strategy_seeds = {
        SelectionStrategy.RANDOM: args.seed + 1,
        SelectionStrategy.HARD: args.seed + 2,
        SelectionStrategy.MIXED: args.seed + 3,
    }
    selections = {}
    for strategy in SelectionStrategy:
        selected = select_negatives(
            strategy,
            candidates,
            scores,
            args.k,
            np.random.default_rng(strategy_seeds[strategy]),
        )
        selections[strategy] = selected
        _print_selection(f"\n{strategy.value}-only selection:", selected, score_by_triple, training)

        assert len(selected) == args.k
        assert len(set(selected)) == args.k

        if strategy is SelectionStrategy.HARD:
            assert set(selected) == top_k_hard
        elif strategy is SelectionStrategy.MIXED:
            n_hard = int(round(args.k * 0.5))
            n_random = args.k - n_hard
            hard_part = {item.triple for item in ranked[:n_hard]}
            assert set(selected[:n_hard]) == hard_part
            assert len(set(selected[n_hard:])) == n_random

    assert set(selections[SelectionStrategy.RANDOM]) != set(selections[SelectionStrategy.HARD])

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
