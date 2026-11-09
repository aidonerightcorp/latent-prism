# LatentPrism (`v1.0.0`)

**Mechanistic Interpretability, Matryoshka Sparse Autoencoders, Multi-Layer Crosscoders, MLP Transcoders, Conformal Sheaf Concept Probes, Nullspace-Guarded Activation Steering, and 10-Format Portable Cartridges.**

`LatentPrism` is an open-source Python toolkit (Apache-2.0) addressing three central failure modes in contemporary sparse-dictionary interpretability and residual-stream steering:

1. **Collateral Capability & Fluency Drift under Naive Residual Steering**: Adding raw Sparse Autoencoder (`SAE`) decoder vectors (`h + alpha * W_dec[i]`) or raw Contrastive Activation Addition (`CAA`) difference vectors directly into transformer hidden states perturbs high-energy syntactic and general-capability anchor directions. `LatentPrism` projects candidate steering directions into the orthogonal complement of the empirical anchor subspace (`d_clean = (I - P_anchor) d`), folds the resulting intervention directly into `mlp.fc2.bias` or a rank-1 `LoRA-XS` weight update (`0` runtime hooks), and serializes zero-dependency **GGUF v3 `controlvector`** bundles.
2. **Correlational Co-Firing vs. Input-Invariant Causal Circuits**: Cross-layer activation cosine (`CoFire`) alone cannot separate persistent shared residual representations from downstream MLP computation. `LatentPrism` combines a joint **Multi-Layer Sparse Crosscoder** (`[L2, L5, L8, L11]`), **Orthogonal Procrustes 2D Manifold Alignment** (`SVD(A^T B)`), and an **L8 MLP Sparse Transcoder** (`x_pre_mlp -> sparse z -> y_post_mlp`) evaluated via **Activation Patching Indirect Effect**.
3. **Heavy Runtime Hooks & Uncalibrated Contradiction Detection**: Instead of paid external judge calls or heavyweight runtime hooks, `LatentPrism` pairs **12-Layer Calibrated Concept Probes** (`Mass-Mean`, shrinkage `Fisher LDA`, and `Ridge Logistic`) with a **Layerwise Sheaf Laplacian + Split Conformal Contradiction (`ELK`) Detector** (`>= 90%` finite-sample clean coverage) and compiles `JumpReLU` SAE + probe cartridges into **10 zero-dependency formats** (`safetensors`, `python`, `js`, `sql`, `c99`, `rust`, `wasm`, `wat`, `wgsl`, `glsl`) matching within `< 3e-7` maximum absolute discrepancy.

---

## Architecture Overview

```text
12-Layer Pre-LN Causal Transformer Hidden States (512-d, L0..L11 + L8 pre/post-MLP)
   |
   +-- [1] MatryoshkaTopKAndJumpReluSuite (layers L2, L5, L8, L11)
   |       * Unit-norm decoder rows + parallel-gradient projection removal
   |       * Nested Matryoshka prefixes (256 < 512 < 1024) + AuxK dead-latent revival
   |       * Calibrated JumpReLU thresholds + automated SAEBench / CE-Bench selectivity
   |
   +-- [2] MultiLayerSparseCrosscoderAndTranscoderSuite
   |       * Joint 4-layer Sparse Crosscoder separating shared vs layer-localized latents
   |       * CoFire partner cosines + Orthogonal Procrustes 2D Manifold Atlas (240 points)
   |       * L8 MLP Sparse Transcoder + weight-space attribution + causal patching
   |
   +-- [3] CalibratedLatentConceptProbeSuite (all 12 layers L0..L11)
   |       * Mass-Mean, Shrinkage Fisher LDA, and L2 Ridge Logistic probes
   |       * Leave-One-Out Ridge Sheaf Laplacian + Split Conformal ELK contradiction detector
   |
   +-- [4] NullspaceGuardedActivationSteerer + Zero-Hook Folding + GGUF v3 Writer
   |       * Raw SAE vs CAA vs Nullspace-Guarded surgical steering across downstream blocks
   |       * Closed-form mlp.fc2.bias and rank-1 LoRA-XS weight folding (0 runtime hooks)
   |       * Zero-dependency GGUF v3 controlvector binary serializer & parser
   |
   +-- [5] TenTargetPortableExporter
           * Compiles 512-d -> JumpReLU SAE -> 2-class cartridges across:
             safetensors | python | js | sql | c99 | rust | wasm | wat | wgsl | glsl
```

---

## Installation & Quickstart

```bash
pip install -e ".[dev]"

# Run the full 12-layer benchmark and generate JSON reports under results/
latent-prism benchmark --output-dir results

# Export GGUF v3 controlvector and portable source/shader cartridges to a directory
latent-prism export-artifacts --output-dir /tmp/latent_prism_exports

# Run the automated conformance, mathematical, and multi-runtime test suite
pytest -q
```

When `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32` is present locally, `LatentPrism` extracts real 12-layer `512-d` hidden states and `L8` pre/post-MLP activations completely offline (`local_files_only=True`, `$0.00` API spend). On air-gapped CI runners without a local cache, `TransformerBackboneRunner` falls back cleanly to a deterministic 12-layer Pre-LN causal transformer fixture with identical module topology.

---

## Configuration Settings Grid & Mechanical Conformance Gates

All architectural dimensions, Matryoshka prefix widths, `TopK`/`AuxK` sparsity levels, learning rates, shrinkage and ridge regularizers, conformal significance `alpha`, steering strength grids, GGUF v3 binary type codes, and WebAssembly opcodes live in [`config/latent_prism_grid.json`](config/latent_prism_grid.json):

- **`NodeGrid` + `ParameterBounds` + Boundary Auto-Expansion**: Validates cross-parameter invariants (`coarse_prefix < mid_prefix < full_dict`, `top_k < coarse_prefix`, `top_k <= aux_k`) and automatically expands saturated parameter axes into a new grid generation.
- **`GridLedger` (`maker != checker`)**: Tracks configuration cells across lifecycle stages and enforces disjoint maker/auditor identities prior to promotion.
- **`ZeroHardcodedValuesAstGate`**: Scans `src/latent_prism/*.py` AST nodes to ensure only canonical structural identities (`{0, 1, -1, 0.0, 1.0, True, False, None}`) appear inline.
- **`DomainAgnosticNamingGate`**: Enforces domain-independent systems and mathematical naming across `config/`, `src/`, and `tests/`.

---

## Repository Layout

- `config/latent_prism_grid.json` — Declarative multi-regime Configuration Settings Grid.
- `src/latent_prism/grid_and_conformance.py` — `NodeGrid`, `GridLedger`, `OperationCostLedger`, `ZeroHardcodedValuesAstGate`, and `DomainAgnosticNamingGate`.
- `src/latent_prism/activation_corpus.py` — `328`-prompt contrastive/persona/anchor corpus, `80`-prompt `ELK` calibration/pressure corpus, and 12-layer `CLIPTextModel` extractor.
- `src/latent_prism/matryoshka_sae.py` — `MatryoshkaTopKAndJumpReluSuite` (`TopK + AuxK + JumpReLU`, unit-norm decoder, parallel-gradient removal, nested prefix evaluation).
- `src/latent_prism/crosscoder_and_transcoder.py` — 4-layer Sparse Crosscoder, Orthogonal Procrustes 2D alignment, and `L8` MLP Sparse Transcoder causal patching.
- `src/latent_prism/concept_probes_and_elk.py` — 12-layer `Mass-Mean`, `Fisher LDA`, and `Ridge Logistic` probes + LOO Sheaf-Conformal `ELK` detector.
- `src/latent_prism/nullspace_steering_and_gguf.py` — Downstream `CLIPTextModel` steering evaluation, closed-form bias/`LoRA-XS` folding, and GGUF v3 `controlvector` binary IO.
- `src/latent_prism/portable_exporter.py` — Zero-dependency 10-target compiler (`safetensors`, `python`, `js`, `sql`, `c99`, `rust`, `wasm`, `wat`, `wgsl`, `glsl`).
- `results/latent_prism_benchmark_summary.json` — Full real 12-layer `CLIPTextModel` benchmark metrics.
- `results/atlas_and_steering_ui_payload.json` — Compact 240-point Procrustes 2D atlas, cross-layer threads, transcoder circuit edges, 12-layer probe/ELK curves, and steering curves.

---

## License

Apache License 2.0 — Copyright 2026 `aidonerightcorp`.
