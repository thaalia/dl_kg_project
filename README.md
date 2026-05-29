# Hard Negative Mining for Knowledge Graph Link Prediction

**RC2526 — Knowledge Graphs — Deliverable 3**
Joint project with Deep Learning (Project 425282 — Milestone 2)

**Team:** Thalia Delhaise, Lenny Briclet, André Fonseca, Rafael Gufler

---

## Final Report

The final report (PDF) is located at:

```
submissions/final_report.pdf
```

It is also seperately uploaded in the hand-in.

---

## Code

All code is in this repository: https://github.com/thaalia/dl_kg_project

The **main entry point** is the Jupyter notebook:

```
code/train_pipeline_colab.ipynb
```

It is self-contained and reproduces every result in the report — baselines,
custom training, slice evaluation, and qualitative analysis — on a Google Colab
T4 GPU. Run Section A once to install dependencies, then follow the sections in order.

### Module overview

| File | Role |
|------|------|
| `code/negative_sampling.py` | Stage 1: Bernoulli corruption + filtered candidate pool |
| `code/score_candidates.py` | Stage 2: scoring candidates with the current model |
| `code/select_candidates.py` | Stage 2: random / hard / mixed selection strategies |
| `code/train_baseline_kge.py` | Baseline training (TransE + RotatE via PyKEEN) |
| `code/train_rotate_custom.py` | Custom training loop (two-stage pipeline) |
| `code/evaluate_slices.py` | Slice evaluation by relation frequency and entity degree |
| `code/qualitative_examples.py` | Per-triple filtered rank and top-K predictions |

### Artifacts

Pre-computed results (summaries, learning curves, slice JSONs, qualitative report)
are committed under `artifacts/` so all analysis sections of the notebook run
without retraining.
