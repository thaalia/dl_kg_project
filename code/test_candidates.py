#!/usr/bin/env python3
"""Smoke test for negative candidate generation on FB15k-237."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from negative_sampling import (  # noqa: E402
    CorruptionSide,
    build_sampling_context,
    generate_candidates,
    generate_candidates_bernoulli,
)


def _label(factory, idx: int) -> str:
    return factory.entity_id_to_label[idx]


def _relation_label(factory, idx: int) -> str:
    return factory.relation_id_to_label[idx]


def _format_triple(factory, triple: tuple[int, int, int]) -> str:
    h, r, t = triple
    return f"({_label(factory, h)}, {_relation_label(factory, r)}, {_label(factory, t)})"


def _assert_candidates_valid(
    positive: tuple[int, int, int],
    side: CorruptionSide,
    candidates: list[tuple[int, int, int]],
    triple_index,
) -> None:
    h, r, t = positive
    for cand in candidates:
        ch, cr, ct = cand
        assert cand != positive, "candidate equals positive triple"
        assert cr == r, "relation changed during corruption"
        if side is CorruptionSide.HEAD:
            assert ct == t and ch != h, f"invalid head corruption: {cand}"
        else:
            assert ch == h and ct != t, f"invalid tail corruption: {cand}"
        assert not triple_index.contains(*cand), f"unfiltered train triple: {cand}"

    assert len(candidates) == len(set(candidates)), "duplicate candidates"


def main() -> int:
    try:
        from pykeen.datasets import FB15k237
    except ImportError:
        print("Install PyKEEN: pip install -r requirements.txt", file=sys.stderr)
        return 1

    os.chdir(REPO_ROOT)
    print("Loading FB15k-237 training split ...")
    training = FB15k237().training
    triple_index, bernoulli, num_entities = build_sampling_context(training)

    mapped = training.mapped_triples.cpu().numpy()
    rng = np.random.default_rng(42)

    demo_idx = 0
    for i in range(len(mapped)):
        h, r, t = map(int, mapped[i])
        if _relation_label(training, r) == "/people/person/place_of_birth":
            demo_idx = i
            break

    h, r, t = map(int, mapped[demo_idx])
    positive = (h, r, t)
    n = 10

    print("\n=== Demo triple ===")
    print(f"Positive: {_format_triple(training, positive)}")

    side, candidates = generate_candidates_bernoulli(
        h,
        r,
        t,
        n=n,
        num_entities=num_entities,
        bernoulli=bernoulli,
        triple_index=triple_index,
        rng=rng,
    )
    print(f"Corruption side: {side.value}")
    print(f"Bernoulli P(corrupt head) for this relation: {bernoulli.corrupt_head_probability[r]:.3f}")
    print(f"\nGenerated {len(candidates)} filtered candidates:")
    for i, cand in enumerate(candidates, start=1):
        print(f"  {i:2d}. {_format_triple(training, cand)}")

    _assert_candidates_valid(positive, side, candidates, triple_index)

    tail_candidates = generate_candidates(
        h,
        r,
        t,
        n=5,
        side=CorruptionSide.TAIL,
        num_entities=num_entities,
        triple_index=triple_index,
        rng=rng,
    )
    _assert_candidates_valid(positive, CorruptionSide.TAIL, tail_candidates, triple_index)

    sample_size = 200
    sample_indices = rng.choice(len(mapped), size=sample_size, replace=False)
    for idx in sample_indices:
        sh, sr, st = map(int, mapped[idx])
        batch_side, batch_candidates = generate_candidates_bernoulli(
            sh,
            sr,
            st,
            n=32,
            num_entities=num_entities,
            bernoulli=bernoulli,
            triple_index=triple_index,
            rng=rng,
        )
        _assert_candidates_valid((sh, sr, st), batch_side, batch_candidates, triple_index)

    print(f"\nAll checks passed ({sample_size} extra triples x n=32 candidates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
