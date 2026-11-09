"""Nullspace-Guarded Activation Steering, Zero-Hook Weight Folding, and Zero-Dependency GGUF v3 Control Vector IO."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_prism.activation_corpus import MultiLayerActivationDataset, TransformerBackboneRunner
from latent_prism.concept_probes_and_elk import LinearConceptProbe
from latent_prism.grid_and_conformance import NodeGrid
from latent_prism.matryoshka_sae import TrainedLayerSAE


def compute_anchor_subspace_basis(anchor_states: np.ndarray, rank: int) -> np.ndarray:
    """Extract orthonormal top-r basis (d, r) spanning the general capability and syntax anchor subspace."""
    _, _, vt = np.linalg.svd(anchor_states.astype(np.float64), full_matrices=False)
    r_eff = min(max(1, int(rank)), int(vt.shape[0]))
    return vt[:r_eff].T.astype(np.float32)


def project_out_anchor_subspace(direction: np.ndarray, anchor_basis: np.ndarray, eps: float) -> np.ndarray:
    """Project steering vector into the orthogonal complement (I - P_anchor) and normalize to unit L2 norm."""
    d64 = direction.astype(np.float64)
    u64 = anchor_basis.astype(np.float64)
    clean = d64 - u64 @ (u64.T @ d64)
    norm = float(np.linalg.norm(clean))
    if norm <= eps:
        return np.zeros_like(direction, dtype=np.float32)
    return (clean / norm).astype(np.float32)


def _align_offset(offset: int, alignment: int) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _pack_gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def serialize_gguf_v3_control_vector(
    directions_1indexed: dict[int, np.ndarray],
    *,
    model_hint: str,
    total_layers: int,
    grid: NodeGrid,
) -> bytes:
    """Serialize per-layer float32 control vectors into zero-dependency GGUF v3 binary format."""
    gspec = grid.raw_spec["steering_and_gguf_spec"]
    magic = bytes(int(b) for b in gspec["gguf_magic_bytes"])
    version = int(gspec["gguf_version"])
    alignment = int(gspec["gguf_alignment"])
    t_u32 = int(gspec["gguf_type_uint32"])
    t_str = int(gspec["gguf_type_string"])
    ggml_f32 = int(gspec["ggml_type_f32"])
    kv_count = int(gspec["gguf_kv_count"])
    n_dims = int(gspec["gguf_tensor_dims"])
    f32_bytes = int(gspec["f32_byte_size"])

    sorted_layers = sorted(directions_1indexed.keys())
    buf = bytearray()
    buf.extend(magic)
    buf.extend(struct.pack("<I", version))
    buf.extend(struct.pack("<Q", len(sorted_layers)))
    buf.extend(struct.pack("<Q", kv_count))

    buf.extend(_pack_gguf_string("general.architecture"))
    buf.extend(struct.pack("<I", t_str))
    buf.extend(_pack_gguf_string("controlvector"))

    buf.extend(_pack_gguf_string("controlvector.model_hint"))
    buf.extend(struct.pack("<I", t_str))
    buf.extend(_pack_gguf_string(model_hint))

    buf.extend(_pack_gguf_string("controlvector.layer_count"))
    buf.extend(struct.pack("<I", t_u32))
    buf.extend(struct.pack("<I", int(total_layers)))

    data_chunks: list[bytes] = []
    running_data_offset = 0
    for layer_1idx in sorted_layers:
        vec = np.ascontiguousarray(directions_1indexed[layer_1idx], dtype="<f4").ravel()
        aligned_off = _align_offset(running_data_offset, alignment)
        pad_len = aligned_off - running_data_offset
        if pad_len > 0:
            data_chunks.append(bytes(pad_len))
        raw_vec = vec.tobytes()
        data_chunks.append(raw_vec)
        running_data_offset = aligned_off + len(vec) * f32_bytes

        buf.extend(_pack_gguf_string(f"direction.{layer_1idx}"))
        buf.extend(struct.pack("<I", n_dims))
        buf.extend(struct.pack("<Q", int(vec.shape[0])))
        buf.extend(struct.pack("<I", ggml_f32))
        buf.extend(struct.pack("<Q", aligned_off))

    header_aligned = _align_offset(len(buf), alignment)
    if header_aligned > len(buf):
        buf.extend(bytes(header_aligned - len(buf)))
    for chunk in data_chunks:
        buf.extend(chunk)
    return bytes(buf)


def parse_gguf_v3_control_vector(blob: bytes | Path, grid: NodeGrid) -> dict[str, Any]:
    """Parse and validate a GGUF v3 controlvector binary payload without external dependencies."""
    raw = blob.read_bytes() if isinstance(blob, Path) else bytes(blob)
    gspec = grid.raw_spec["steering_and_gguf_spec"]
    expected_magic = bytes(int(b) for b in gspec["gguf_magic_bytes"])
    alignment = int(gspec["gguf_alignment"])
    t_u32 = int(gspec["gguf_type_uint32"])
    t_str = int(gspec["gguf_type_string"])
    f32_bytes = int(gspec["f32_byte_size"])
    u64_bytes = int(gspec["u64_byte_size"])

    pos = 0
    magic = raw[pos : pos + len(expected_magic)]
    pos += len(expected_magic)
    if magic != expected_magic:
        raise ValueError("Invalid GGUF magic header")

    (version,) = struct.unpack_from("<I", raw, pos)
    pos += f32_bytes
    (tensor_count,) = struct.unpack_from("<Q", raw, pos)
    pos += u64_bytes
    (kv_count,) = struct.unpack_from("<Q", raw, pos)
    pos += u64_bytes

    def _read_str(cursor: int) -> tuple[str, int]:
        (slen,) = struct.unpack_from("<Q", raw, cursor)
        cursor += u64_bytes
        sval = raw[cursor : cursor + slen].decode("utf-8")
        return sval, cursor + slen

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key, pos = _read_str(pos)
        (vtype,) = struct.unpack_from("<I", raw, pos)
        pos += f32_bytes
        if vtype == t_str:
            val, pos = _read_str(pos)
            metadata[key] = val
        elif vtype == t_u32:
            (uval,) = struct.unpack_from("<I", raw, pos)
            pos += f32_bytes
            metadata[key] = int(uval)
        else:
            raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")

    descriptors: list[tuple[str, int, int]] = []
    for _ in range(tensor_count):
        tname, pos = _read_str(pos)
        (ndims,) = struct.unpack_from("<I", raw, pos)
        pos += f32_bytes
        total_elems = 1
        for _ in range(ndims):
            (dim_n,) = struct.unpack_from("<Q", raw, pos)
            pos += u64_bytes
            total_elems *= int(dim_n)
        (_ggml_t,) = struct.unpack_from("<I", raw, pos)
        pos += f32_bytes
        (rel_off,) = struct.unpack_from("<Q", raw, pos)
        pos += u64_bytes
        descriptors.append((tname, total_elems, int(rel_off)))

    data_base = _align_offset(pos, alignment)
    tensors: dict[str, np.ndarray] = {}
    for tname, total_elems, rel_off in descriptors:
        start = data_base + rel_off
        end = start + total_elems * f32_bytes
        tensors[tname] = np.frombuffer(raw[start:end], dtype="<f4").copy()

    return {
        "version": int(version),
        "tensor_count": int(tensor_count),
        "metadata": metadata,
        "tensors": tensors,
    }


@dataclass
class SteeringBenchmarkResult:
    """Empirical steering curves, zero-hook weight folding metrics, and GGUF v3 controlvector summary."""

    steering_curves: list[dict[str, Any]]
    zero_hook_folding: dict[str, Any]
    gguf_summary: dict[str, Any]
    gguf_bytes: bytes
    summary: dict[str, Any]


class NullspaceGuardedActivationSteerer:
    """Benchmarks Raw SAE vs CAA vs Nullspace-Guarded Steering on real downstream Transformer passes + Zero-Hook Folding."""

    def __init__(self, grid: NodeGrid) -> None:
        self.grid = grid

    def run(
        self,
        dataset: MultiLayerActivationDataset,
        runner: TransformerBackboneRunner,
        saes_by_layer: dict[int, TrainedLayerSAE],
        probes_by_regime_layer: dict[tuple[str, int], dict[str, LinearConceptProbe]],
        cell: dict[str, Any],
    ) -> SteeringBenchmarkResult:
        gspec = self.grid.raw_spec["steering_and_gguf_spec"]
        cspec = self.grid.raw_spec["backbone_and_corpus_spec"]
        num = self.grid.raw_spec["numerical_constants"]
        eps = float(num["eps"])
        two = int(num["two"])

        steer_layers = [int(x) for x in gspec["steering_layers"]]
        multipliers = [float(x) for x in gspec["strength_multipliers"]]
        target_regime = str(gspec["steering_regime"])
        anchor_rank = int(cell["anchor_nullspace_rank"])
        eval_n = int(gspec["steering_eval_count"])
        anchor_eval_n = int(gspec["anchor_eval_count"])
        penalty_scale = float(gspec["collateral_penalty_scale"])
        final_layer = int(cspec["elk_late_layer"])

        neg_records = [
            r for r in dataset.records if r.group_kind == "contrastive" and r.regime_name == target_regime and r.label == 0
        ][:eval_n]
        pos_ids = [
            r.prompt_id
            for r in dataset.records
            if r.group_kind == "contrastive" and r.regime_name == target_regime and r.label == 1
        ]
        neg_ids = [
            r.prompt_id
            for r in dataset.records
            if r.group_kind == "contrastive" and r.regime_name == target_regime and r.label == 0
        ]
        anchor_records = [r for r in dataset.records if r.group_kind == "anchor"][:anchor_eval_n]

        neg_texts = [r.text for r in neg_records]
        anchor_texts = [r.text for r in anchor_records]
        n_neg = len(neg_texts)
        combined_texts = neg_texts + anchor_texts

        base_comb_fwd = runner.forward_batch(combined_texts)
        base_neg_pooled = base_comb_fwd["pooled_all_layers"][:n_neg]
        base_anc_pooled = base_comb_fwd["pooled_all_layers"][n_neg:]

        readout_probe = probes_by_regime_layer[(target_regime, final_layer)]["ridge_logistic"]
        base_target_probs = readout_probe.predict_proba(base_neg_pooled[:, final_layer, :])
        base_anc_final = base_comb_fwd["final_normed_eos"][n_neg:]
        base_anc_final_norms = np.linalg.norm(base_anc_final, axis=1) + eps
        anchor_basis_final = compute_anchor_subspace_basis(base_anc_final, anchor_rank)

        steering_curves: list[dict[str, Any]] = []
        nullspace_dirs_by_layer: dict[int, np.ndarray] = {}

        for l_idx in steer_layers:
            sae = saes_by_layer[l_idx]
            pooled_l = dataset.pooled_all_layers[:, l_idx, :]
            med_norm = float(np.median(np.linalg.norm(dataset.token_states_by_layer[l_idx], axis=1)))

            z_pos = sae.encode_topk(pooled_l[pos_ids])
            z_neg = sae.encode_topk(pooled_l[neg_ids])
            feat_diff = z_pos.mean(axis=0) - z_neg.mean(axis=0)
            best_feat = int(np.argmax(feat_diff))
            d_raw_sae = sae.w_dec[best_feat].astype(np.float32)
            d_raw_sae = d_raw_sae / (float(np.linalg.norm(d_raw_sae)) + eps)

            caa_diff = pooled_l[pos_ids].mean(axis=0) - pooled_l[neg_ids].mean(axis=0)
            d_caa = (caa_diff / (float(np.linalg.norm(caa_diff)) + eps)).astype(np.float32)

            lda_w = probes_by_regime_layer[(target_regime, l_idx)]["fisher_lda"].weight
            lda_unit = (lda_w / (float(np.linalg.norm(lda_w)) + eps)).astype(np.float32)
            hybrid = d_caa + lda_unit

            anc_states_l = base_anc_pooled[:, l_idx, :]
            anchor_basis_l = compute_anchor_subspace_basis(anc_states_l, anchor_rank)
            d_null = project_out_anchor_subspace(hybrid, anchor_basis_l, eps)
            nullspace_dirs_by_layer[l_idx] = d_null

            methods = {
                "raw_sae": d_raw_sae,
                "caa": d_caa,
                "nullspace_guarded": d_null,
            }

            sae_selectivity = float(sae.metrics.get("contrastive_selectivity_score", 1.0))

            for mult in multipliers:
                alpha = float(mult) * med_norm
                for method_name, unit_vec in methods.items():
                    delta_vec = (alpha * unit_vec).astype(np.float32)
                    steered_comb = runner.forward_batch(combined_texts, intervene_layer=l_idx, intervene_delta=delta_vec)
                    steered_neg_pooled = steered_comb["pooled_all_layers"][:n_neg]
                    steered_anc_pooled = steered_comb["pooled_all_layers"][n_neg:]

                    steered_probs = readout_probe.predict_proba(steered_neg_pooled[:, final_layer, :])
                    prob_shift = float(np.mean(steered_probs - base_target_probs))

                    anc_diff_intervened = steered_anc_pooled[:, l_idx, :] - base_anc_pooled[:, l_idx, :]
                    proj_intervened = anc_diff_intervened @ anchor_basis_l
                    intervened_anchor_drift = float(np.mean(np.linalg.norm(proj_intervened, axis=1)) / (alpha + eps))

                    steered_anc_final = steered_comb["final_normed_eos"][n_neg:]
                    dots = np.sum(steered_anc_final * base_anc_final, axis=1)
                    denom = (np.linalg.norm(steered_anc_final, axis=1) + eps) * base_anc_final_norms
                    downstream_cos_retention = float(np.mean(dots / denom))

                    downstream_diff = steered_anc_final - base_anc_final
                    downstream_anchor_drift = float(
                        np.mean(np.linalg.norm(downstream_diff @ anchor_basis_final, axis=1))
                        / (float(np.mean(base_anc_final_norms)) + eps)
                    )

                    clean_steerability = (
                        max(0.0, prob_shift)
                        * max(0.0, downstream_cos_retention)
                        * sae_selectivity
                        * (1.0 / (1.0 + penalty_scale * intervened_anchor_drift + penalty_scale * downstream_anchor_drift))
                    )

                    steering_curves.append(
                        {
                            "layer": int(l_idx),
                            "strength_multiplier": float(mult),
                            "alpha_norm": alpha,
                            "method": method_name,
                            "target_prob_shift": prob_shift,
                            "intervened_anchor_subspace_drift": intervened_anchor_drift,
                            "downstream_anchor_cosine_retention": downstream_cos_retention,
                            "downstream_anchor_subspace_drift": downstream_anchor_drift,
                            "clean_steerability_score": clean_steerability,
                        }
                    )

        null_rows = [r for r in steering_curves if r["method"] == "nullspace_guarded"]
        raw_rows = [r for r in steering_curves if r["method"] == "raw_sae"]
        caa_rows = [r for r in steering_curves if r["method"] == "caa"]

        internal_null_rows = [r for r in null_rows if r["layer"] < final_layer]
        best_null_row = max(internal_null_rows or null_rows, key=lambda r: r["clean_steerability_score"])
        fold_layer = int(best_null_row["layer"])
        fold_mult = float(gspec["default_fold_multiplier"])
        med_fold = float(np.median(np.linalg.norm(dataset.token_states_by_layer[fold_layer], axis=1)))
        fold_delta = (fold_mult * med_fold * nullspace_dirs_by_layer[fold_layer]).astype(np.float32)

        hook_ref = runner.forward_batch(neg_texts, intervene_layer=fold_layer, intervene_delta=fold_delta)
        blk = runner.core.encoder.layers[fold_layer]
        orig_bias = blk.mlp.fc2.bias.data.clone()
        try:
            with torch.no_grad():
                blk.mlp.fc2.bias.data.add_(torch.from_numpy(fold_delta).to(orig_bias.device, dtype=orig_bias.dtype))
            folded_bias_out = runner.forward_batch(neg_texts)
        finally:
            with torch.no_grad():
                blk.mlp.fc2.bias.data.copy_(orig_bias)

        bias_fold_max_diff = float(
            np.max(np.abs(hook_ref["pooled_all_layers"][:, final_layer, :] - folded_bias_out["pooled_all_layers"][:, final_layer, :]))
        )

        enc_fold = runner.encode_texts(neg_texts)
        mid_acts: dict[str, torch.Tensor] = {}

        def _cap_act(_mod: Any, _inp: Any, out: Any) -> None:
            mid_acts["post_act"] = out.detach().cpu()

        act_handle = blk.mlp.activation_fn.register_forward_hook(_cap_act)
        try:
            with torch.no_grad():
                _ = runner.model(**enc_fold, output_hidden_states=False)
        finally:
            act_handle.remove()

        post_act = mid_acts["post_act"]
        mean_gate = post_act.mean(dim=(0, 1))
        gate_unit = mean_gate / mean_gate.norm().clamp_min(eps)
        mean_proj = float((post_act @ gate_unit).mean().item())
        scale_factor = 1.0 / (mean_proj if abs(mean_proj) > eps else 1.0)
        delta_t = torch.from_numpy(fold_delta).to(torch.float32)
        rank1_update = scale_factor * torch.outer(delta_t, gate_unit)

        orig_weight = blk.mlp.fc2.weight.data.clone()

        def _explicit_lora_hook(_mod: Any, inp: Any, out: Any) -> Any:
            h_in = inp[0]
            lora_term = h_in @ rank1_update.t().to(h_in.device, dtype=h_in.dtype)
            return out + lora_term

        lora_handle = blk.mlp.fc2.register_forward_hook(_explicit_lora_hook)
        try:
            explicit_lora_out = runner.forward_batch(neg_texts)
        finally:
            lora_handle.remove()

        try:
            with torch.no_grad():
                blk.mlp.fc2.weight.data.add_(rank1_update.to(orig_weight.device, dtype=orig_weight.dtype))
            folded_lora_out = runner.forward_batch(neg_texts)
        finally:
            with torch.no_grad():
                blk.mlp.fc2.weight.data.copy_(orig_weight)

        lora_fold_max_diff = float(
            np.max(
                np.abs(
                    explicit_lora_out["pooled_all_layers"][:, final_layer, :]
                    - folded_lora_out["pooled_all_layers"][:, final_layer, :]
                )
            )
        )

        zero_hook_folding = {
            "folded_layer": fold_layer,
            "runtime_hooks_after_folding": 0,
            "bias_fold_vs_hook_max_abs_diff": bias_fold_max_diff,
            "rank1_lora_xs_vs_adapter_max_abs_diff": lora_fold_max_diff,
            "tolerance": float(gspec["weight_folding_tolerance"]),
        }

        gguf_dirs = {int(l_idx) + 1: nullspace_dirs_by_layer[l_idx] for l_idx in steer_layers}
        gguf_bytes = serialize_gguf_v3_control_vector(
            gguf_dirs,
            model_hint=str(cspec["model_hint"]),
            total_layers=dataset.num_layers,
            grid=self.grid,
        )
        parsed_gguf = parse_gguf_v3_control_vector(gguf_bytes, self.grid)
        roundtrip_diffs = [
            float(np.max(np.abs(parsed_gguf["tensors"][f"direction.{k}"] - v))) for k, v in gguf_dirs.items()
        ]
        gguf_summary = {
            "format_version": parsed_gguf["version"],
            "architecture": parsed_gguf["metadata"]["general.architecture"],
            "model_hint": parsed_gguf["metadata"]["controlvector.model_hint"],
            "layer_count": parsed_gguf["metadata"]["controlvector.layer_count"],
            "exported_tensors": sorted(parsed_gguf["tensors"].keys()),
            "payload_bytes": len(gguf_bytes),
            "roundtrip_max_abs_diff": max(roundtrip_diffs) if roundtrip_diffs else 0.0,
        }

        max_null_intervened_drift = max(r["intervened_anchor_subspace_drift"] for r in null_rows)
        mean_raw_intervened_drift = float(np.mean([r["intervened_anchor_subspace_drift"] for r in raw_rows]))
        mean_caa_intervened_drift = float(np.mean([r["intervened_anchor_subspace_drift"] for r in caa_rows]))

        high_mult = max(multipliers)
        null_high = [r for r in null_rows if r["strength_multiplier"] == high_mult]
        raw_high = [r for r in raw_rows if r["strength_multiplier"] == high_mult]
        caa_high = [r for r in caa_rows if r["strength_multiplier"] == high_mult]

        summary = {
            "peak_clean_steerability_layer": int(best_null_row["layer"]),
            "peak_clean_steerability_score": float(best_null_row["clean_steerability_score"]),
            "nullspace_max_intervened_anchor_drift": max_null_intervened_drift,
            "raw_sae_mean_intervened_anchor_drift": mean_raw_intervened_drift,
            "caa_mean_intervened_anchor_drift": mean_caa_intervened_drift,
            "high_strength_nullspace_anchor_retention": float(
                np.mean([r["downstream_anchor_cosine_retention"] for r in null_high])
            ),
            "high_strength_raw_sae_anchor_retention": float(
                np.mean([r["downstream_anchor_cosine_retention"] for r in raw_high])
            ),
            "high_strength_caa_anchor_retention": float(
                np.mean([r["downstream_anchor_cosine_retention"] for r in caa_high])
            ),
            "high_strength_nullspace_downstream_drift": float(
                np.mean([r["downstream_anchor_subspace_drift"] for r in null_high])
            ),
            "high_strength_caa_downstream_drift": float(
                np.mean([r["downstream_anchor_subspace_drift"] for r in caa_high])
            ),
            "zero_hook_weight_folding": zero_hook_folding,
            "gguf_control_vector": gguf_summary,
        }

        return SteeringBenchmarkResult(
            steering_curves=steering_curves,
            zero_hook_folding=zero_hook_folding,
            gguf_summary=gguf_summary,
            gguf_bytes=gguf_bytes,
            summary=summary,
        )
