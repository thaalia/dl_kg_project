#!/usr/bin/env python3
"""
Crop multi-panel artifact figures to single panels for poster use.

Run from repo root:
    python3 poster/crop_figures.py

Output: poster/figures/{mrr_bar,val_mrr_curves,slice_rel_freq}.png
"""
from pathlib import Path
from PIL import Image

ROOT   = Path(__file__).resolve().parent.parent
ARTS   = ROOT / "artifacts"
OUT    = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def crop_panel(src: Path, l: float, r: float,
               t: float = 0.0, b: float = 1.0,
               out: str | None = None) -> None:
    """Crop [l,r] x [t,b] (fractions of image size) and save."""
    img = Image.open(src).convert("RGB")
    W, H = img.size
    box  = (int(l * W), int(t * H), int(r * W), int(b * H))
    crop = img.crop(box)
    name = out or (src.stem + "_panel.png")
    dest = OUT / name
    crop.save(dest, dpi=(300, 300))
    print(f"  {src.name}  [{l:.2f},{r:.2f}]×[{t:.2f},{b:.2f}]"
          f"  → {name}  {crop.size}")


print("=== Cropping poster figures ===\n")

# ── Global filtered MRR bar chart ─────────────────────────────────────────
# d3_headline_figures has 3 panels (global MRR | MRR by tail in-degree |
# MRR by relation frequency).  The leftmost panel runs 0 → ~34 % of width.
crop_panel(
    ARTS / "d3_headline_figures.png",
    l=0.00, r=0.34,
    out="mrr_bar.png",
)

# ── Validation MRR learning curves ────────────────────────────────────────
# custom_learning_curves has 2 panels (training loss | validation MRR).
# Right (validation MRR) panel: 50 % → 100 % of width.
# Trim top ~7 % to remove the truncated figure-level title bleed.
crop_panel(
    ARTS / "custom_learning_curves.png",
    l=0.50, r=1.00, t=0.07,
    out="val_mrr_curves.png",
)

# ── Slice evaluation — relation frequency ─────────────────────────────────
# slice_bar_plots has 3 equal panels (relation freq | head out-deg | tail in-deg).
# Left panel (relation frequency): 0 → 32 % to avoid next panel's y-axis label.
crop_panel(
    ARTS / "slice_bar_plots.png",
    l=0.00, r=0.32,
    out="slice_rel_freq.png",
)

# ── FB15k-237 relation-frequency distribution ──────────────────────────────
# dataset_distributions has 3 panels (relation freq | out-degree | in-degree).
# Left panel (relation frequency, log scale): 0 → ~34 % of width.
crop_panel(
    ARTS / "dataset_distributions.png",
    l=0.00, r=0.34,
    out="dataset_rel_freq.png",
)

print(f"\nDone.  Figures written to: {OUT}")
