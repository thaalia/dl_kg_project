#!/usr/bin/env python3
"""
Minimal FB15k-237 baseline using PyKEEN (TransE / RotatE / DistMult).
Install:  pip install -r requirements.txt
Run from repo root:  python code/train_baseline_kge.py --model TransE --epochs 3
Saves:    artifacts/baseline/<model>_summary.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        from pykeen.pipeline import pipeline
    except ImportError as e:
        print("Install PyKEEN:  pip install pykeen torch", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["TransE", "RotatE", "DistMult"], default="TransE")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = REPO_ROOT / "artifacts" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model}_summary.txt"

    # Dataset id varies slightly by PyKEEN version; try a few spellings
    last_err: Optional[Exception] = None
    result = None
    for ds in ("FB15k237", "fb15k237"):
        try:
            result = pipeline(
                model=args.model,
                dataset=ds,
                model_kwargs={"embedding_dim": args.dim},
                training_kwargs={"num_epochs": args.epochs, "batch_size": args.batch_size},
                optimizer_kwargs={"lr": args.lr},
                random_seed=args.seed,
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            result = None
    if result is None:
        print("Failed to open FB15k-237. Try: pip install -U pykeen  (see PyKEEN dataset name for your version).", file=sys.stderr)
        if last_err is not None:
            raise last_err
        return 1

    # Persist everything we can; API differs slightly across PyKEEN minor versions
    lines = [repr(result)]
    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"model={args.model} epochs={args.epochs} dim={args.dim} seed={args.seed}\n\n")
        f.write("\n".join(lines))

    if hasattr(result, "save_to_directory"):
        result.save_to_directory(str(out_dir / f"pykeen_{args.model}"))
    try:
        print(result)  # often prints metrics table
    except Exception:
        pass
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    raise SystemExit(main())
