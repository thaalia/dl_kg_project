"""Select negatives from a scored candidate pool (random / hard / mixed)."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from negative_sampling import (
    BernoulliCorruptionProbs,
    TrainTripleIndex,
    choose_corruption_side,
    generate_candidates,
)
from score_candidates import RankedCandidate, rank_candidates, score_triples

if TYPE_CHECKING:
    import torch


class SelectionStrategy(str, Enum):
    RANDOM = "random"
    HARD = "hard"
    MIXED = "mixed"


def select_random(
    candidates: list[tuple[int, int, int]],
    k: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Pick k negatives uniformly at random from the candidate pool."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(candidates):
        raise ValueError(f"Cannot select k={k} negatives from a pool of size {len(candidates)}.")

    indices = rng.choice(len(candidates), size=k, replace=False)
    return [candidates[int(i)] for i in indices]


def select_hard(
    ranked: list[RankedCandidate],
    k: int,
) -> list[tuple[int, int, int]]:
    """Pick the k highest-scored (hardest) negatives."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(ranked):
        raise ValueError(f"Cannot select k={k} negatives from a pool of size {len(ranked)}.")

    return [item.triple for item in ranked[:k]]


def select_mixed(
    candidates: list[tuple[int, int, int]],
    ranked: list[RankedCandidate],
    k: int,
    rng: np.random.Generator,
    *,
    hard_fraction: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Pick a mix of random and hard negatives (default 50/50)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be between 0 and 1.")
    if k > len(candidates):
        raise ValueError(f"Cannot select k={k} negatives from a pool of size {len(candidates)}.")

    n_hard = int(round(k * hard_fraction))
    n_hard = min(max(n_hard, 0), k)
    n_random = k - n_hard

    hard_triples = select_hard(ranked, n_hard) if n_hard else []
    hard_set = set(hard_triples)

    remaining = [triple for triple in candidates if triple not in hard_set]
    if n_random > len(remaining):
        raise ValueError(
            f"Not enough remaining candidates for mixed selection: "
            f"need {n_random} random negatives but only {len(remaining)} left after picking hard ones."
        )

    random_triples = select_random(remaining, n_random, rng) if n_random else []
    return hard_triples + random_triples


def select_negatives(
    strategy: SelectionStrategy,
    candidates: list[tuple[int, int, int]],
    scores: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    hard_fraction: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Select k training negatives using the requested strategy."""
    ranked = rank_candidates(candidates, scores)

    if strategy is SelectionStrategy.RANDOM:
        return select_random(candidates, k, rng)
    if strategy is SelectionStrategy.HARD:
        return select_hard(ranked, k)
    if strategy is SelectionStrategy.MIXED:
        return select_mixed(candidates, ranked, k, rng, hard_fraction=hard_fraction)

    raise ValueError(f"Unknown strategy: {strategy}")


def sample_training_negatives_batch(
    positive_batch: np.ndarray,
    strategy: SelectionStrategy,
    model: torch.nn.Module,
    *,
    pool_size: int,
    k: int,
    num_entities: int,
    bernoulli: BernoulliCorruptionProbs,
    triple_index: TrainTripleIndex,
    rng: np.random.Generator,
    device: str,
    hard_fraction: float = 0.5,
) -> np.ndarray:
    """Sample k negatives per positive triple using the two-stage pipeline."""
    if positive_batch.ndim != 2 or positive_batch.shape[1] != 3:
        raise ValueError("positive_batch must have shape (batch_size, 3).")

    batch_size = positive_batch.shape[0]
    negatives: list[list[tuple[int, int, int]]] = []
    all_candidates: list[tuple[int, int, int]] = []
    candidate_slices: list[slice] = []

    for row in positive_batch:
        head, relation, tail = map(int, row)
        side = choose_corruption_side(relation, rng, bernoulli)
        candidates = generate_candidates(
            head,
            relation,
            tail,
            n=pool_size,
            side=side,
            num_entities=num_entities,
            triple_index=triple_index,
            rng=rng,
        )
        start = len(all_candidates)
        all_candidates.extend(candidates)
        candidate_slices.append(slice(start, start + len(candidates)))

    if strategy is SelectionStrategy.RANDOM:
        for slc in candidate_slices:
            pool = all_candidates[slc]
            negatives.append(select_random(pool, k, rng))
        return np.asarray(negatives, dtype=np.int64)

    pool_scores = score_triples(model, all_candidates, device=device)
    for slc in candidate_slices:
        pool = all_candidates[slc]
        pool_score_array = pool_scores[slc]
        negatives.append(
            select_negatives(
                strategy,
                pool,
                pool_score_array,
                k,
                rng,
                hard_fraction=hard_fraction,
            )
        )
    return np.asarray(negatives, dtype=np.int64)
