"""Automated conformance, mathematical, multi-layer benchmark, GGUF v3, and 10-target parity tests for LatentPrism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latent_prism.activation_corpus import extract_multilayer_activation_dataset
from latent_prism.benchmark_runner import run_full_latent_prism_benchmark
from latent_prism.concept_probes_and_elk import LinearConceptProbe
from latent_prism.grid_and_conformance import (
    DomainAgnosticNamingGate,
    GridLedger,
    NodeGrid,
    ZeroHardcodedValuesAstGate,
)
from latent_prism.matryoshka_sae import TrainedLayerSAE
from latent_prism.nullspace_steering_and_gguf import (
    compute_anchor_subspace_basis,
    parse_gguf_v3_control_vector,
    project_out_anchor_subspace,
    serialize_gguf_v3_control_vector,
)
from latent_prism.portable_exporter import TenTargetPortableExporter, build_sae_probe_cartridge

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_conformance_gates_and_grid_lifecycle(tmp_path: Path) -> None:
    src_files = sorted((REPO_ROOT / "src" / "latent_prism").glob("*.py"))
    ast_res = ZeroHardcodedValuesAstGate().assert_clean(src_files)
    assert ast_res["violations"] == 0
    assert ast_res["checked_files"] >= 8

    bad_py = tmp_path / "bad_magic.py"
    bad_py.write_text("def compute_val(x: float) -> float:\n    return x * 42.0\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        ZeroHardcodedValuesAstGate().assert_clean([bad_py])

    naming_res = DomainAgnosticNamingGate().assert_clean(REPO_ROOT)
    assert naming_res["violations"] == 0

    bad_ws = tmp_path / "bad_ws"
    bad_ws.mkdir()
    forbidden_file = bad_ws / ("health" + "care_router.py")
    forbidden_file.write_text("def route() -> int:\n    return 1\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        DomainAgnosticNamingGate().assert_clean(bad_ws)

    grid = NodeGrid.from_json_file()
    ledger = GridLedger(stages=grid.lifecycle_stages)
    cid, cell = grid.resolve_regime_cell("surgical_nullspace_steering")
    ledger.register_cell(cid, "surgical_nullspace_steering", cell, maker_id="maker_alpha")
    for st in grid.lifecycle_stages[1:-2]:
        ledger.advance(cid, st, actor_id="maker_alpha")
    audit_stage = grid.lifecycle_stages[-2]
    with pytest.raises(PermissionError):
        ledger.advance(cid, audit_stage, actor_id="maker_alpha")
    ledger.advance(cid, audit_stage, actor_id="checker_beta", metrics={"ok": 1.0})
    ledger.advance(cid, grid.lifecycle_stages[-1], actor_id="checker_beta")
    assert ledger.summary()["promoted_cells"] == 1

    expanded, axes = grid.auto_expand_saturated_boundaries([cell])
    assert "anchor_nullspace_rank" in axes
    assert expanded.generation == grid.generation + 1
    assert expanded.parameters["anchor_nullspace_rank"].bounds.max_value > grid.parameters[
        "anchor_nullspace_rank"
    ].bounds.max_value


def test_offline_procedural_fallback_extractor() -> None:
    grid = NodeGrid.from_json_file()
    dataset, runner = extract_multilayer_activation_dataset(grid, force_offline_fallback=True)
    assert dataset.backbone_source == "procedural_offline_fallback"
    assert runner.backbone_source == "procedural_offline_fallback"
    assert len(dataset.records) >= 240
    assert dataset.pooled_all_layers.shape == (len(dataset.records), 12, 512)
    assert dataset.l8_pre_mlp_tokens.shape == dataset.l8_post_mlp_tokens.shape
    assert dataset.l8_pre_mlp_tokens.shape[1] == 512


def test_benchmark_metrics_and_ui_payload_contracts() -> None:
    summary_path = REPO_ROOT / "results" / "latent_prism_benchmark_summary.json"
    ui_path = REPO_ROOT / "results" / "atlas_and_steering_ui_payload.json"
    if not (summary_path.is_file() and ui_path.is_file()):
        run_full_latent_prism_benchmark(output_dir=REPO_ROOT / "results")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ui = json.loads(ui_path.read_text(encoding="utf-8"))

    assert summary["backbone"]["num_layers"] == 12
    assert summary["backbone"]["hidden_dim"] == 512
    assert summary["backbone"]["prompt_corpus_count"] >= 240

    sae_sum = summary["matryoshka_sae_suite"]
    assert sae_sum["mean_heldout_r2"] > 0.65
    comp = sae_sum["matryoshka_vs_flat_topk"]
    assert comp["coarse_prefix_r2_gain"] > 0.02

    cc_sum = summary["crosscoder_and_transcoder_suite"]
    assert cc_sum["procrustes_alignment"]["total_atlas_points"] == 240
    assert len(ui["atlas_2d_points"]) == 240
    assert cc_sum["procrustes_alignment"]["procrustes_rmse_reduction_ratio"] > 0.0
    assert (
        cc_sum["mlp_transcoder"]["causal_patching_indirect_effect"]
        > cc_sum["mlp_transcoder"]["random_control_indirect_effect"]
    )

    probe_sum = summary["concept_probes_and_elk_suite"]["summary"]
    assert len(ui["layerwise_probe_and_elk_curves"]) == 12
    assert probe_sum["peak_ridge_logistic_auroc"] > 0.85
    elk = probe_sum["elk_contradiction_detector"]
    assert elk["clean_empirical_coverage"] >= 0.90
    assert elk["contradiction_detection_auroc"] > 0.90
    assert elk["mid_layer_truth_invariance_abs_gap"] < 0.05

    st_sum = summary["nullspace_steering_and_gguf_suite"]
    assert st_sum["nullspace_max_intervened_anchor_drift"] < 1e-6
    assert st_sum["raw_sae_mean_intervened_anchor_drift"] > 0.05
    assert st_sum["caa_mean_intervened_anchor_drift"] > 0.05
    assert st_sum["high_strength_nullspace_anchor_retention"] > st_sum["high_strength_caa_anchor_retention"]
    assert st_sum["peak_clean_steerability_layer"] in {5, 8}
    assert st_sum["zero_hook_weight_folding"]["runtime_hooks_after_folding"] == 0
    assert st_sum["zero_hook_weight_folding"]["bias_fold_vs_hook_max_abs_diff"] < 1e-5
    assert st_sum["zero_hook_weight_folding"]["rank1_lora_xs_vs_adapter_max_abs_diff"] < 1e-5


def test_gguf_v3_and_ten_target_portable_exporter_parity(tmp_path: Path) -> None:
    grid = NodeGrid.from_json_file()
    rng = np.random.default_rng(2026)

    anchor_mat = rng.standard_normal((40, 512)).astype(np.float32)
    basis = compute_anchor_subspace_basis(anchor_mat, 32)
    raw_dir = rng.standard_normal(512).astype(np.float32)
    clean_dir = project_out_anchor_subspace(raw_dir, basis, 1e-8)
    assert float(np.linalg.norm(clean_dir @ basis)) < 1e-6

    dirs = {3: clean_dir, 6: clean_dir, 9: clean_dir, 12: clean_dir}
    gguf_blob = serialize_gguf_v3_control_vector(
        dirs,
        model_hint="clip-vit-base-patch32",
        total_layers=12,
        grid=grid,
    )
    gguf_file = tmp_path / "test_control.gguf"
    gguf_file.write_bytes(gguf_blob)
    parsed = parse_gguf_v3_control_vector(gguf_file, grid)
    assert parsed["version"] == 3
    assert parsed["metadata"]["general.architecture"] == "controlvector"
    assert parsed["metadata"]["controlvector.layer_count"] == 12
    for l_key, vec in dirs.items():
        assert float(np.max(np.abs(parsed["tensors"][f"direction.{l_key}"] - vec))) == 0.0

    w_enc = (rng.standard_normal((512, 64)) * 0.08).astype(np.float32)
    w_dec = w_enc.T.copy()
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    dummy_sae = TrainedLayerSAE(
        layer_idx=8,
        variant="matryoshka_topk_auxk",
        input_dim=512,
        dict_size=64,
        top_k=8,
        prefixes=(16, 32, 64),
        b_pre=(rng.standard_normal(512) * 0.05).astype(np.float32),
        w_enc=w_enc,
        b_enc=np.zeros(64, dtype=np.float32),
        w_dec=w_dec,
        jumprelu_theta=np.full(64, 0.05, dtype=np.float32),
        metrics={},
    )
    dummy_probe = LinearConceptProbe(
        method="ridge_logistic",
        layer_idx=8,
        regime_name="formal_logic",
        weight=(rng.standard_normal(512) * 0.1).astype(np.float32),
        bias=0.02,
        logit_clip=25.0,
    )
    cart = build_sae_probe_cartridge(dummy_sae, dummy_probe, grid)
    x_eval = rng.standard_normal((5, 512)).astype(np.float32)

    exporter = TenTargetPortableExporter(grid)
    report = exporter.evaluate_cross_runtime_parity(cart, x_eval, scratch_dir=tmp_path / "compile_box")
    assert report["within_tolerance"] is True
    assert report["overall_max_abs_diff"] < 3e-7
    assert all(report["shader_and_text_checks"].values())

    forbidden_binaries = [
        p
        for p in REPO_ROOT.rglob("*")
        if p.is_file() and p.suffix in {".wasm", ".rlib", ".so", ".dylib", ".dll", ".o", ".exe"}
    ]
    assert forbidden_binaries == []
