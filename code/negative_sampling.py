"""Negative candidate generation for custom hard-negative experiments.

Matches PyKEEN's Bernoulli head/tail corruption probabilities and filters
candidates that are known true triples in the training split.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pykeen.triples import TriplesFactory


class CorruptionSide(str, Enum):
    HEAD = "head"
    TAIL = "tail"


@dataclass(frozen=True)
class BernoulliCorruptionProbs:
    """Per-relation probability of corrupting the head (tail otherwise)."""

    corrupt_head_probability: np.ndarray  # shape: (num_relations,)


@dataclass(frozen=True)
class TrainTripleIndex:
    """Fast membership test for known training triples."""

    known: frozenset[tuple[int, int, int]]

    @classmethod
    def from_mapped_triples(cls, mapped_triples: np.ndarray) -> TrainTripleIndex:
        rows = mapped_triples.astype(np.int64, copy=False)
        return cls(frozenset(map(tuple, rows.tolist())))

    def contains(self, h: int, r: int, t: int) -> bool:
        return (int(h), int(r), int(t)) in self.known


def compute_bernoulli_probs(
    mapped_triples: np.ndarray,
    num_relations: int,
) -> BernoulliCorruptionProbs:
    """Compute head-corruption probabilities as in PyKEEN's BernoulliNegativeSampler."""
    triples = mapped_triples.astype(np.int64, copy=False)
    corrupt_head_probability = np.zeros(num_relations, dtype=np.float64)

    head_rel_pairs, tail_counts = np.unique(triples[:, :2], axis=0, return_counts=True)
    rel_tail_pairs, head_counts = np.unique(triples[:, 1:], axis=0, return_counts=True)

    for relation in range(num_relations):
        tail_mask = head_rel_pairs[:, 1] == relation
        head_mask = rel_tail_pairs[:, 0] == relation

        tph = tail_counts[tail_mask].astype(np.float64).mean() if tail_mask.any() else 0.0
        hpt = head_counts[head_mask].astype(np.float64).mean() if head_mask.any() else 0.0

        if tph + hpt == 0.0:
            corrupt_head_probability[relation] = 0.5
        else:
            corrupt_head_probability[relation] = tph / (tph + hpt)

    return BernoulliCorruptionProbs(corrupt_head_probability=corrupt_head_probability)


def choose_corruption_side(
    relation: int,
    rng: np.random.Generator,
    bernoulli: BernoulliCorruptionProbs,
) -> CorruptionSide:
    """Sample head vs tail corruption for one positive triple."""
    if rng.random() < bernoulli.corrupt_head_probability[relation]:
        return CorruptionSide.HEAD
    return CorruptionSide.TAIL


def random_entity_excluding(
    current: int,
    num_entities: int,
    rng: np.random.Generator,
) -> int:
    """Uniform random entity != current (same trick as PyKEEN random_replacement_)."""
    if num_entities <= 1:
        raise ValueError("num_entities must be > 1")
    replacement = int(rng.integers(0, num_entities - 1))
    if replacement >= current:
        replacement += 1
    return replacement


def corrupt_once(
    head: int,
    relation: int,
    tail: int,
    side: CorruptionSide,
    num_entities: int,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Create one corrupted triple by replacing head or tail."""
    if side is CorruptionSide.HEAD:
        return random_entity_excluding(head, num_entities, rng), relation, tail
    return head, relation, random_entity_excluding(tail, num_entities, rng)


def generate_candidates(
    head: int,
    relation: int,
    tail: int,
    *,
    n: int,
    side: CorruptionSide,
    num_entities: int,
    triple_index: TrainTripleIndex,
    rng: np.random.Generator,
    max_attempts_per_candidate: int = 100,
) -> list[tuple[int, int, int]]:
    """Generate n unique filtered negative candidates on a fixed corruption side."""
    if n <= 0:
        raise ValueError("n must be positive")

    positive = (int(head), int(relation), int(tail))
    candidates: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    max_attempts = n * max_attempts_per_candidate
    attempts = 0

    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        candidate = corrupt_once(head, relation, tail, side, num_entities, rng)
        if candidate == positive or candidate in seen or triple_index.contains(*candidate):
            continue
        seen.add(candidate)
        candidates.append(candidate)

    if len(candidates) < n:
        raise RuntimeError(
            f"Could only sample {len(candidates)}/{n} filtered candidates for "
            f"{positive} ({side.value} corruption) after {attempts} attempts."
        )
    return candidates


def generate_candidates_bernoulli(
    head: int,
    relation: int,
    tail: int,
    *,
    n: int,
    num_entities: int,
    bernoulli: BernoulliCorruptionProbs,
    triple_index: TrainTripleIndex,
    rng: np.random.Generator,
    max_attempts_per_candidate: int = 100,
) -> tuple[CorruptionSide, list[tuple[int, int, int]]]:
    """Bernoulli head/tail choice, then n filtered candidates on that side."""
    side = choose_corruption_side(relation, rng, bernoulli)
    candidates = generate_candidates(
        head,
        relation,
        tail,
        n=n,
        side=side,
        num_entities=num_entities,
        triple_index=triple_index,
        rng=rng,
        max_attempts_per_candidate=max_attempts_per_candidate,
    )
    return side, candidates


def build_sampling_context(training: TriplesFactory) -> tuple[TrainTripleIndex, BernoulliCorruptionProbs, int]:
    """Build index and Bernoulli probabilities from a PyKEEN training factory."""
    mapped = training.mapped_triples.cpu().numpy()
    num_entities = int(training.num_entities)
    num_relations = int(training.num_relations)
    return (
        TrainTripleIndex.from_mapped_triples(mapped),
        compute_bernoulli_probs(mapped, num_relations),
        num_entities,
    )
