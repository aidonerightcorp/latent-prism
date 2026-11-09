"""End-to-end LatentPrism benchmark orchestrator and UI payload exporter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from latent_prism.activation_corpus import extract_multilayer_activation_dataset
from latent_prism.concept_probes_and_elk import CalibratedLatentConceptProbeSuite
from latent_prism.crosscoder_and_transcoder import MultiLayerSparseCrosscoderAndTranscoderSuite
from latent_prism.grid_and_conformance import (
    DomainAgnosticNamingGate,
    GridLedger,
    NodeGrid,
    OperationCostLedger,
    ZeroHardcodedValuesAstGate,
    round_nested_floats,
)
from latent_prism.matryoshka_sae import MatryoshkaTopKAndJumpReluSuite
from latent_prism.nullspace_steering_and_gguf import NullspaceGuardedActivationSteerer
from latent_prism.portable_exporter import TenTargetPortableExporter, build_sae_probe_cartridge


def run_full_latent_prism_benchmark(
    *,
    grid_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    force_offline_fallback: bool = False,
) -> dict[str, Any]:
    """Execute full 12-layer extraction, Matryoshka SAEs, Crosscoder/Transcoder, 12-layer Probes/ELK, Steering, and 10-target export."""
    t_start = time.perf_counter()
    grid = NodeGrid.from_json_file(grid_path)
    num = grid.raw_spec["numerical_constants"]
    round_digits = int(num["float_round_digits"])
    us_per_s = float(num["microseconds_per_second"])

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    ast_gate_summary = ZeroHardcodedValuesAstGate().assert_clean(sorted(pkg_dir.glob("*.py")))
    naming_gate_summary = DomainAgnosticNamingGate().assert_clean(repo_root)

    grid_ledger = GridLedger(stages=grid.lifecycle_stages)
    cost_ledger = OperationCostLedger()

    sae_cell_id, sae_cell = grid.resolve_regime_cell("residual_dictionary_learning")
    cc_cell_id, cc_cell = grid.resolve_regime_cell("causal_circuit_discovery")
    steer_cell_id, steer_cell = grid.resolve_regime_cell("surgical_nullspace_steering")

    t_ext0 = time.perf_counter()
    dataset, runner = extract_multilayer_activation_dataset(grid, force_offline_fallback=force_offline_fallback)
    ext_sec = time.perf_counter() - t_ext0

    t_sae0 = time.perf_counter()
    sae_out = MatryoshkaTopKAndJumpReluSuite(grid).run(dataset, sae_cell)
    sae_sec = time.perf_counter() - t_sae0
    saes_by_layer = sae_out["saes_by_layer"]

    grid_ledger.promote_full_lifecycle(
        sae_cell_id,
        "residual_dictionary_learning",
        sae_cell,
        maker_id="sae_trainer",
        checker_id="heldout_sae_auditor",
        metrics={"mean_heldout_r2": sae_out["summary"]["mean_heldout_r2"]},
    )
    cost_ledger.record(
        "residual_dictionary_learning",
        dataset.backbone_source,
        sae_cell_id,
        sae_out["summary"]["mean_heldout_r2"],
        sae_sec * us_per_s,
        int(sae_cell["sae_dictionary_width"]),
    )

    t_cc0 = time.perf_counter()
    cc_out = MultiLayerSparseCrosscoderAndTranscoderSuite(grid).run(dataset, saes_by_layer, cc_cell)
    cc_sec = time.perf_counter() - t_cc0

    grid_ledger.promote_full_lifecycle(
        cc_cell_id,
        "causal_circuit_discovery",
        cc_cell,
        maker_id="circuit_builder",
        checker_id="causal_patch_auditor",
        metrics={"transcoder_r2": cc_out.transcoder_metrics["heldout_r2"]},
    )
    cost_ledger.record(
        "causal_circuit_discovery",
        dataset.backbone_source,
        cc_cell_id,
        cc_out.transcoder_metrics["heldout_r2"],
        cc_sec * us_per_s,
        len(cc_out.atlas_points),
    )

    probe_out = CalibratedLatentConceptProbeSuite(grid).run(dataset)

    t_st0 = time.perf_counter()
    steer_out = NullspaceGuardedActivationSteerer(grid).run(
        dataset,
        runner,
        saes_by_layer,
        probe_out.probes_by_regime_layer,
        steer_cell,
    )
    steer_sec = time.perf_counter() - t_st0

    grid_ledger.promote_full_lifecycle(
        steer_cell_id,
        "surgical_nullspace_steering",
        steer_cell,
        maker_id="nullspace_steerer",
        checker_id="downstream_anchor_auditor",
        metrics={"peak_clean_steerability": steer_out.summary["peak_clean_steerability_score"]},
    )
    cost_ledger.record(
        "surgical_nullspace_steering",
        dataset.backbone_source,
        steer_cell_id,
        steer_out.summary["peak_clean_steerability_score"],
        steer_sec * us_per_s,
        len(steer_out.gguf_bytes),
    )

    expanded_grid, expanded_axes = grid.auto_expand_saturated_boundaries([sae_cell, cc_cell, steer_cell])

    export_layer = int(grid.raw_spec["backbone_and_corpus_spec"]["transcoder_layer"])
    export_regime = str(grid.raw_spec["steering_and_gguf_spec"]["steering_regime"])
    export_probe = probe_out.probes_by_regime_layer[(export_regime, export_layer)]["ridge_logistic"]
    cartridge = build_sae_probe_cartridge(saes_by_layer[export_layer], export_probe, grid)
    n_parity = int(grid.raw_spec["portable_exporter_spec"]["parity_eval_samples"])
    parity_samples = dataset.pooled_all_layers[:n_parity, export_layer, :]
    exporter = TenTargetPortableExporter(grid)
    parity_report = exporter.evaluate_cross_runtime_parity(cartridge, parity_samples)

    total_sec = time.perf_counter() - t_start

    summary_payload = round_nested_floats(
        {
            "suite": "latent_prism",
            "version": "1.0.0",
            "backbone": {
                "source": dataset.backbone_source,
                "model_id": dataset.backbone_id,
                "num_layers": dataset.num_layers,
                "hidden_dim": dataset.hidden_dim,
                "target_layers": list(dataset.target_layers),
                "prompt_corpus_count": len(dataset.records),
                "elk_prompt_count": len(dataset.elk_records),
                "token_vectors_per_layer": int(dataset.token_states_by_layer[dataset.target_layers[0]].shape[0]),
                "extraction_seconds": ext_sec,
            },
            "conformance_gates": {
                "zero_hardcoded_values_ast_gate": ast_gate_summary,
                "domain_agnostic_naming_gate": naming_gate_summary,
                "grid_ledger": grid_ledger.summary(),
                "boundary_auto_expansion": {
                    "initial_generation": grid.generation,
                    "expanded_generation": expanded_grid.generation,
                    "expanded_axes": expanded_axes,
                    "expanded_anchor_nullspace_max": expanded_grid.parameters["anchor_nullspace_rank"].bounds.max_value,
                },
                "contextual_cost_observations": len(cost_ledger.observations),
            },
            "matryoshka_sae_suite": sae_out["summary"],
            "crosscoder_and_transcoder_suite": {
                "crosscoder": cc_out.crosscoder_metrics,
                "procrustes_alignment": cc_out.procrustes_metrics,
                "mlp_transcoder": cc_out.transcoder_metrics,
            },
            "concept_probes_and_elk_suite": {
                "summary": probe_out.summary,
                "layerwise_curves": probe_out.layerwise_curves,
            },
            "nullspace_steering_and_gguf_suite": steer_out.summary,
            "portable_ten_target_exporter": parity_report,
            "total_runtime_seconds": total_sec,
        },
        round_digits,
    )

    ui_payload = round_nested_floats(
        {
            "schema": "latent_prism.atlas_and_steering_ui.v1",
            "backbone": summary_payload["backbone"],
            "atlas_2d_points": cc_out.atlas_points,
            "cross_layer_threads": cc_out.cross_layer_threads,
            "transcoder_circuit_edges": cc_out.transcoder_circuit_edges,
            "layerwise_probe_and_elk_curves": probe_out.layerwise_curves,
            "elk_contradiction_summary": probe_out.elk_metrics,
            "raw_vs_nullspace_steering_curves": steer_out.steering_curves,
            "zero_hook_weight_folding": steer_out.zero_hook_folding,
            "gguf_control_vector": steer_out.gguf_summary,
            "portable_exporter_parity": parity_report,
        },
        round_digits,
    )

    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "latent_prism_benchmark_summary.json").write_text(
            json.dumps(summary_payload, indent=int(num["two"])) + "\n",
            encoding="utf-8",
        )
        (out_root / "atlas_and_steering_ui_payload.json").write_text(
            json.dumps(ui_payload, indent=int(num["two"])) + "\n",
            encoding="utf-8",
        )

    return {
        "summary": summary_payload,
        "ui_payload": ui_payload,
    }
