"""Command-line interface for LatentPrism benchmarks, GGUF v3 control vectors, and 10-target cartridge exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_prism.activation_corpus import extract_multilayer_activation_dataset
from latent_prism.benchmark_runner import run_full_latent_prism_benchmark
from latent_prism.concept_probes_and_elk import CalibratedLatentConceptProbeSuite
from latent_prism.grid_and_conformance import NodeGrid
from latent_prism.matryoshka_sae import MatryoshkaTopKAndJumpReluSuite
from latent_prism.nullspace_steering_and_gguf import NullspaceGuardedActivationSteerer
from latent_prism.portable_exporter import TenTargetPortableExporter, build_sae_probe_cartridge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="latent-prism", description="LatentPrism mechanistic interpretability toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="Run full 12-layer benchmark and emit summary + UI JSON payloads")
    p_bench.add_argument("--output-dir", type=Path, default=Path("results"))
    p_bench.add_argument("--offline-fallback", action="store_true")

    p_export = sub.add_parser("export-artifacts", help="Export GGUF v3 controlvector and 10-target cartridges to a folder")
    p_export.add_argument("--output-dir", type=Path, required=True)
    p_export.add_argument("--offline-fallback", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "benchmark":
        res = run_full_latent_prism_benchmark(
            output_dir=args.output_dir,
            force_offline_fallback=bool(args.offline_fallback),
        )
        print(
            json.dumps(
                {
                    "backbone": res["summary"]["backbone"]["source"],
                    "mean_sae_r2": res["summary"]["matryoshka_sae_suite"]["mean_heldout_r2"],
                    "peak_probe_auroc": res["summary"]["concept_probes_and_elk_suite"]["summary"][
                        "peak_ridge_logistic_auroc"
                    ],
                    "elk_auroc": res["summary"]["concept_probes_and_elk_suite"]["summary"][
                        "elk_contradiction_detector"
                    ]["contradiction_detection_auroc"],
                    "portable_max_diff": res["summary"]["portable_ten_target_exporter"]["overall_max_abs_diff"],
                }
            )
        )
        return 0

    if args.command == "export-artifacts":
        grid = NodeGrid.from_json_file()
        _, sae_cell = grid.resolve_regime_cell("residual_dictionary_learning")
        _, steer_cell = grid.resolve_regime_cell("surgical_nullspace_steering")
        dataset, runner = extract_multilayer_activation_dataset(grid, force_offline_fallback=bool(args.offline_fallback))
        sae_out = MatryoshkaTopKAndJumpReluSuite(grid).run(dataset, sae_cell)
        probe_out = CalibratedLatentConceptProbeSuite(grid).run(dataset)
        steer_out = NullspaceGuardedActivationSteerer(grid).run(
            dataset, runner, sae_out["saes_by_layer"], probe_out.probes_by_regime_layer, steer_cell
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "nullspace_control_vector.gguf").write_bytes(steer_out.gguf_bytes)

        exp_layer = int(grid.raw_spec["backbone_and_corpus_spec"]["transcoder_layer"])
        exp_regime = str(grid.raw_spec["steering_and_gguf_spec"]["steering_regime"])
        cart = build_sae_probe_cartridge(
            sae_out["saes_by_layer"][exp_layer],
            probe_out.probes_by_regime_layer[(exp_regime, exp_layer)]["ridge_logistic"],
            grid,
        )
        artifacts = TenTargetPortableExporter(grid).export_all(cart)
        (out_dir / "cartridge.safetensors").write_bytes(artifacts["safetensors"])
        (out_dir / "cartridge.py").write_text(artifacts["python"], encoding="utf-8")
        (out_dir / "cartridge.mjs").write_text(artifacts["js"], encoding="utf-8")
        (out_dir / "cartridge.sql").write_text(artifacts["sql"], encoding="utf-8")
        (out_dir / "cartridge.c").write_text(artifacts["c99"], encoding="utf-8")
        (out_dir / "cartridge.rs").write_text(artifacts["rust"], encoding="utf-8")
        (out_dir / "cartridge.wat").write_text(artifacts["wat"], encoding="utf-8")
        (out_dir / "cartridge.wgsl").write_text(artifacts["wgsl"], encoding="utf-8")
        (out_dir / "cartridge.glsl").write_text(artifacts["glsl"], encoding="utf-8")
        print(json.dumps({"exported_dir": str(out_dir), "gguf_bytes": len(steer_out.gguf_bytes)}))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
