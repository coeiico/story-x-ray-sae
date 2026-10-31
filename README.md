# story-x-ray-sae

**Sparse autoencoders.** Sparse autoencoders on a pretrained TinyStories model's residual stream, pulling out human-readable features and steering with them.

**Status:** not started.

## Question

Can we pull human-readable features out of a real, if small, language model?

## Plan

- **Architecture:** a sparse autoencoder (one hidden layer 8–32× wider than the input, ReLU with an L1 penalty, or TopK) trained on residual-stream activations of a pretrained TinyStories model loaded with Hugging Face Transformers, such as `roneneldan/TinyStories-33M`. Optional second target: a tiny-transformer model trained on TinyStories.
- **Data:** TinyStories via Hugging Face `datasets`.

## Experiments

- Reconstruction quality versus sparsity.
- Dead-feature rate.
- Top-activating text for each feature.
- Hand-label 50 features.
- Steer generation by adding a feature's direction.

## Done when

- There is a gallery of clearly interpretable features and one working steering demo.

Plus the general checklist:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Notes

- The only repo that uses Hugging Face Transformers, to load the pretrained TinyStories model.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just bg train ...       # the same, in a detached tmux session; any recipe works
just analyze <run>      # baselines, probes, interventions and plots
just cloud <sweep>      # the same runs on Modal, one GPU each; see .agents/instructions/cloud.md
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
story-x-ray-sae/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/story_x_ray_sae/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── cloud.py                 the same runs on Modal, one GPU per run
├── figures/                 README figures: LaTeX sources, data and built PNGs
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Bricken et al. 2023, Towards monosemanticity
- Eldan and Li 2023, TinyStories

## License

MIT
