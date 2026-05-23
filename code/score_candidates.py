"""Score negative candidate triples with a trained KGE model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROTATE_CHECKPOINT = REPO_ROOT / "artifacts/baseline/pykeen_RotatE/trained_model.pkl"


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def load_rotate_model(
    checkpoint: Path | str = DEFAULT_ROTATE_CHECKPOINT,
    device: str | None = None,
) -> torch.nn.Module:
    """Load a PyKEEN RotatE model saved by train_baseline_kge.py."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"RotatE checkpoint not found at {path}. "
            "Run: python code/train_baseline_kge.py --model RotatE --epochs 50 --patience 10"
        )

    device = device or best_device()
    if device == "mps":
        device = "cpu"

    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model.to(device)


def score_triples(
    model: torch.nn.Module,
    triples: list[tuple[int, int, int]],
    *,
    device: str | None = None,
) -> np.ndarray:
    """Return plausibility scores; higher means the model finds the triple more plausible."""
    if not triples:
        return np.array([], dtype=np.float64)

    device = device or next(model.parameters()).device.type
    batch = torch.tensor(triples, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model.score_hrt(batch).view(-1)
    return scores.detach().cpu().numpy().astype(np.float64)


@dataclass(frozen=True)
class RankedCandidate:
    triple: tuple[int, int, int]
    score: float


def rank_candidates(
    candidates: list[tuple[int, int, int]],
    scores: np.ndarray,
) -> list[RankedCandidate]:
    """Rank candidates by score descending (hardest / most plausible first)."""
    if len(candidates) != len(scores):
        raise ValueError(
            f"Expected {len(candidates)} scores for {len(candidates)} candidates, got {len(scores)}."
        )

    order = np.argsort(-scores)
    return [
        RankedCandidate(triple=candidates[i], score=float(scores[i]))
        for i in order
    ]
