"""Multi-Layer Sparse Crosscoder, Orthogonal Procrustes 2D Manifold Alignment, and L8 MLP Sparse Transcoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from latent_prism.activation_corpus import MultiLayerActivationDataset
from latent_prism.grid_and_conformance import NodeGrid
from latent_prism.matryoshka_sae import TrainedLayerSAE


class _TorchMultiLayerCrosscoder(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, dict_size: int, seed: int, eps: float) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.eps = float(eps)
        w_enc = torch.randn((num_layers, input_dim, dict_size), generator=gen, dtype=torch.float32)
        w_enc = w_enc / (w_enc.norm(dim=1, keepdim=True).clamp_min(self.eps) * float(num_layers) ** float(1 / (1 + 1)))
        self.w_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(torch.zeros(dict_size, dtype=torch.float32))
        self.w_dec = nn.Parameter(w_enc.transpose(1, 1 + 1).contiguous().clone())
        self.b_dec = nn.Parameter(torch.zeros((num_layers, input_dim), dtype=torch.float32))

    def encode_topk(self, x_stack: torch.Tensor, k: int) -> torch.Tensor:
        agg = torch.einsum("bld,ldm->bm", x_stack, self.w_enc) + self.b_enc
        pre = torch.relu(agg)
        k_eff = min(max(1, int(k)), int(pre.shape[1]))
        vals, idx = torch.topk(pre, k=k_eff, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bm,lmd->bld", z, self.w_dec) + self.b_dec.unsqueeze(0)


class _TorchMlpTranscoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dict_size: int,
        x_mean: torch.Tensor,
        y_mean: torch.Tensor,
        seed: int,
        eps: float,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.eps = float(eps)
        w_enc = torch.randn((input_dim, dict_size), generator=gen, dtype=torch.float32)
        w_enc = w_enc / w_enc.norm(dim=0, keepdim=True).clamp_min(self.eps)
        w_dec = torch.randn((dict_size, output_dim), generator=gen, dtype=torch.float32)
        w_dec = w_dec / w_dec.norm(dim=1, keepdim=True).clamp_min(self.eps)

        self.b_in = nn.Parameter(x_mean.clone().to(torch.float32))
        self.b_out = nn.Parameter(y_mean.clone().to(torch.float32))
        self.w_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(torch.zeros(dict_size, dtype=torch.float32))
        self.w_dec = nn.Parameter(w_dec)

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            norms = self.w_dec.data.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            self.w_dec.data.div_(norms)

    def remove_parallel_decoder_grad_(self) -> None:
        if self.w_dec.grad is None:
            return
        with torch.no_grad():
            parallel = (self.w_dec.grad * self.w_dec.data).sum(dim=-1, keepdim=True) * self.w_dec.data
            self.w_dec.grad.sub_(parallel)

    def encode_topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        pre = torch.relu((x - self.b_in) @ self.w_enc + self.b_enc)
        k_eff = min(max(1, int(k)), int(pre.shape[1]))
        vals, idx = torch.topk(pre, k=k_eff, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def forward(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_topk(x, k)
        y_hat = self.b_out + z @ self.w_dec
        return y_hat, z


@dataclass
class CrosscoderAndTranscoderResult:
    """Outputs and metrics for joint multi-layer crosscoder, Procrustes atlas, and L8 MLP transcoder."""

    crosscoder_metrics: dict[str, Any]
    procrustes_metrics: dict[str, Any]
    transcoder_metrics: dict[str, Any]
    atlas_points: list[dict[str, Any]]
    cross_layer_threads: list[dict[str, Any]]
    transcoder_circuit_edges: list[dict[str, Any]]


def orthogonal_procrustes_2d(source_anchor: np.ndarray, target_anchor: np.ndarray) -> np.ndarray:
    """Solve min_{R^T R = I} ||source_anchor @ R - target_anchor||_F via exact SVD(A^T B)."""
    cross = source_anchor.T @ target_anchor
    u_mat, _, vt_mat = np.linalg.svd(cross, full_matrices=False)
    return (u_mat @ vt_mat).astype(np.float32)


def _intrinsic_2d_coords(feature_profiles: np.ndarray, two: int, eps: float) -> np.ndarray:
    """Project (n_features, n_prompts) activation profiles into centered 2D SVD coordinates."""
    centered = feature_profiles - feature_profiles.mean(axis=0, keepdims=True)
    u_mat, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u_mat[:, :two] * s_vals[:two][None, :]
    scale = float(np.sqrt(np.mean(coords ** two)) + eps)
    return (coords / scale).astype(np.float32)


class MultiLayerSparseCrosscoderAndTranscoderSuite:
    """Trains the 4-layer Sparse Crosscoder, aligns 2D feature manifolds via Procrustes, and fits L8 MLP Transcoder."""

    def __init__(self, grid: NodeGrid) -> None:
        self.grid = grid

    def run(
        self,
        dataset: MultiLayerActivationDataset,
        saes_by_layer: dict[int, TrainedLayerSAE],
        cell: dict[str, Any],
    ) -> CrosscoderAndTranscoderResult:
        cspec = self.grid.raw_spec["backbone_and_corpus_spec"]
        ctspec = self.grid.raw_spec["crosscoder_transcoder_spec"]
        num = self.grid.raw_spec["numerical_constants"]
        eps = float(num["eps"])
        two = int(num["two"])
        seed = int(cspec["default_seed"])
        train_frac = float(cspec["train_fraction"])
        torch.set_num_threads(int(num["torch_threads"]))

        layers = list(dataset.target_layers)
        unique_prompts = np.unique(dataset.token_prompt_indices)
        rng = np.random.default_rng(seed)
        perm_prompts = rng.permutation(unique_prompts)
        n_train_p = max(1, int(round(train_frac * float(len(perm_prompts)))))
        train_set = set(perm_prompts[:n_train_p].tolist())
        train_mask = np.array([int(pid) in train_set for pid in dataset.token_prompt_indices], dtype=bool)
        test_mask = ~train_mask

        raw_stack = np.stack([dataset.token_states_by_layer[l_idx] for l_idx in layers], axis=1).astype(np.float32)
        mean_stack = raw_stack[train_mask].mean(axis=0, keepdims=True)
        rms_stack = np.sqrt(np.mean((raw_stack[train_mask] - mean_stack) ** two, axis=(0, two), keepdims=True)) + eps
        norm_stack = (raw_stack - mean_stack) / rms_stack

        cc_dict = int(cell["crosscoder_dictionary_width"])
        cc_k = int(ctspec["crosscoder_top_k"])
        cc_lr = float(ctspec["crosscoder_lr"])
        cc_epochs = int(ctspec["crosscoder_epochs"])
        cc_bs = int(ctspec["crosscoder_batch_size"])

        torch.manual_seed(seed)
        cc_model = _TorchMultiLayerCrosscoder(len(layers), dataset.hidden_dim, cc_dict, seed, eps)
        cc_opt = torch.optim.Adam(cc_model.parameters(), lr=cc_lr)
        x_cc_train = torch.from_numpy(norm_stack[train_mask])
        gen = torch.Generator().manual_seed(seed)

        for _ in range(cc_epochs):
            order = torch.randperm(x_cc_train.shape[0], generator=gen)
            for start in range(0, int(x_cc_train.shape[0]), cc_bs):
                batch = x_cc_train[order[start : start + cc_bs]]
                z = cc_model.encode_topk(batch, cc_k)
                hat = cc_model.decode(z)
                loss = ((batch - hat) ** two).mean()
                cc_opt.zero_grad()
                loss.backward()
                cc_opt.step()

        with torch.no_grad():
            x_cc_test = torch.from_numpy(norm_stack[test_mask])
            z_cc_test = cc_model.encode_topk(x_cc_test, cc_k)
            hat_cc_test = cc_model.decode(z_cc_test).numpy()
            w_dec_cc = cc_model.w_dec.detach().cpu().numpy()

        test_norm = norm_stack[test_mask]
        cc_var = float(np.mean((test_norm - test_norm.mean(axis=0, keepdims=True)) ** two)) + eps
        cc_mse = float(np.mean((test_norm - hat_cc_test) ** two))
        cc_r2 = 1.0 - cc_mse / cc_var

        layer_norms = np.linalg.norm(w_dec_cc, axis=-1)
        norm_sq = layer_norms ** two
        part_ratio = ((norm_sq.sum(axis=0) ** two) / (np.sum(norm_sq ** two, axis=0) + eps)) / float(len(layers))
        shared_fraction = float(np.mean(part_ratio >= float(num["half"])))

        per_layer_r2: dict[str, float] = {}
        for idx, l_num in enumerate(layers):
            v_l = float(np.mean((test_norm[:, idx] - test_norm[:, idx].mean(axis=0, keepdims=True)) ** two)) + eps
            m_l = float(np.mean((test_norm[:, idx] - hat_cc_test[:, idx]) ** two))
            per_layer_r2[f"L{l_num}"] = 1.0 - m_l / v_l

        crosscoder_metrics = {
            "joint_heldout_r2": cc_r2,
            "joint_heldout_fvu": cc_mse / cc_var,
            "per_layer_heldout_r2": per_layer_r2,
            "mean_layer_participation_ratio": float(np.mean(part_ratio)),
            "shared_backbone_latent_fraction": shared_fraction,
            "layer_specific_latent_fraction": 1.0 - shared_fraction,
        }

        feat_per_layer = int(ctspec["atlas_features_per_layer"])
        min_cos = float(ctspec["cofire_min_cosine"])
        max_threads = int(ctspec["max_ui_threads"])

        selected_indices: dict[int, np.ndarray] = {}
        traces_by_layer: dict[int, np.ndarray] = {}
        raw_coords_by_layer: dict[int, np.ndarray] = {}

        for l_num in layers:
            sae = saes_by_layer[l_num]
            z_pooled = sae.encode_topk(dataset.pooled_all_layers[:, l_num, :])
            energy = np.mean(z_pooled ** two, axis=0)
            chosen = np.argsort(energy)[-feat_per_layer:][::-1]
            selected_indices[l_num] = chosen
            sub_trace = z_pooled[:, chosen]
            col_norms = np.linalg.norm(sub_trace, axis=0, keepdims=True) + eps
            norm_trace = (sub_trace / col_norms).astype(np.float32)
            traces_by_layer[l_num] = norm_trace
            raw_coords_by_layer[l_num] = _intrinsic_2d_coords(norm_trace.T, two, eps)

        aligned_coords: dict[int, np.ndarray] = {layers[0]: raw_coords_by_layer[layers[0]]}
        pre_rmses: list[float] = []
        post_rmses: list[float] = []
        candidate_threads: list[dict[str, Any]] = []

        for step_i in range(len(layers) - 1):
            l_prev = layers[step_i]
            l_curr = layers[step_i + 1]
            cos_mat = traces_by_layer[l_prev].T @ traces_by_layer[l_curr]
            partner_prev = np.argmax(cos_mat, axis=0)
            partner_cos = cos_mat[partner_prev, np.arange(cos_mat.shape[1])]

            target_anchor = aligned_coords[l_prev][partner_prev]
            source_raw = raw_coords_by_layer[l_curr]
            weights = np.maximum(partner_cos, 0.0)[:, None] + eps
            src_c = source_raw - np.sum(weights * source_raw, axis=0, keepdims=True) / np.sum(weights)
            tgt_c = target_anchor - np.sum(weights * target_anchor, axis=0, keepdims=True) / np.sum(weights)

            rot = orthogonal_procrustes_2d(src_c * np.sqrt(weights), tgt_c * np.sqrt(weights))
            aligned_curr = src_c @ rot + np.sum(weights * target_anchor, axis=0, keepdims=True) / np.sum(weights)
            aligned_coords[l_curr] = aligned_curr.astype(np.float32)

            pre_rmse = float(np.sqrt(np.mean((src_c - tgt_c) ** two)))
            post_rmse = float(np.sqrt(np.mean(((src_c @ rot) - tgt_c) ** two)))
            pre_rmses.append(pre_rmse)
            post_rmses.append(post_rmse)

            for curr_local_idx in range(cos_mat.shape[1]):
                c_val = float(partner_cos[curr_local_idx])
                if c_val >= min_cos:
                    prev_local_idx = int(partner_prev[curr_local_idx])
                    candidate_threads.append(
                        {
                            "from_layer": int(l_prev),
                            "from_feature": int(selected_indices[l_prev][prev_local_idx]),
                            "from_local_idx": prev_local_idx,
                            "to_layer": int(l_curr),
                            "to_feature": int(selected_indices[l_curr][curr_local_idx]),
                            "to_local_idx": curr_local_idx,
                            "cofire_cosine": c_val,
                        }
                    )

        candidate_threads.sort(key=lambda item: item["cofire_cosine"], reverse=True)
        cross_layer_threads = candidate_threads[:max_threads]

        regime_names = sorted({r.regime_name for r in dataset.records if r.group_kind == "contrastive"})
        atlas_points: list[dict[str, Any]] = []
        for l_num in layers:
            chosen = selected_indices[l_num]
            coords = aligned_coords[l_num]
            sae = saes_by_layer[l_num]
            z_pooled = sae.encode_topk(dataset.pooled_all_layers[:, l_num, :])[:, chosen]
            for local_i, feat_id in enumerate(chosen):
                best_reg = "general"
                best_sel = 0.0
                col = z_pooled[:, local_i]
                for rname in regime_names:
                    pos_ids = [
                        r.prompt_id
                        for r in dataset.records
                        if r.group_kind == "contrastive" and r.regime_name == rname and r.label == 1
                    ]
                    neg_ids = [
                        r.prompt_id
                        for r in dataset.records
                        if r.group_kind == "contrastive" and r.regime_name == rname and r.label == 0
                    ]
                    m_pos = float(np.mean(col[pos_ids]))
                    m_neg = float(np.mean(col[neg_ids]))
                    sel = max(0.0, (m_pos - m_neg) / (m_pos + m_neg + float(np.std(col)) + eps))
                    if sel > best_sel:
                        best_sel = sel
                        best_reg = rname
                atlas_points.append(
                    {
                        "layer": int(l_num),
                        "local_index": int(local_i),
                        "feature_id": int(feat_id),
                        "x": float(coords[local_i, 0]),
                        "y": float(coords[local_i, 1]),
                        "dominant_regime": best_reg,
                        "selectivity": best_sel,
                        "mean_activation": float(np.mean(col)),
                    }
                )

        procrustes_metrics = {
            "total_atlas_points": len(atlas_points),
            "features_per_layer": feat_per_layer,
            "mean_pre_procrustes_rmse": float(np.mean(pre_rmses)),
            "mean_post_procrustes_rmse": float(np.mean(post_rmses)),
            "procrustes_rmse_reduction_ratio": 1.0 - (float(np.mean(post_rmses)) / (float(np.mean(pre_rmses)) + eps)),
            "discovered_cofire_threads": len(cross_layer_threads),
        }

        tr_dict = int(cell["transcoder_dictionary_width"])
        tr_k = int(ctspec["transcoder_top_k"])
        tr_lr = float(ctspec["transcoder_lr"])
        tr_epochs = int(ctspec["transcoder_epochs"])
        tr_bs = int(ctspec["transcoder_batch_size"])

        x_pre_tr = torch.from_numpy(dataset.l8_pre_mlp_tokens[train_mask])
        y_post_tr = torch.from_numpy(dataset.l8_post_mlp_tokens[train_mask])
        tr_model = _TorchMlpTranscoder(
            dataset.hidden_dim,
            dataset.hidden_dim,
            tr_dict,
            x_pre_tr.mean(dim=0),
            y_post_tr.mean(dim=0),
            seed + two,
            eps,
        )
        tr_opt = torch.optim.Adam(tr_model.parameters(), lr=tr_lr)
        gen_tr = torch.Generator().manual_seed(seed + two)

        for _ in range(tr_epochs):
            order = torch.randperm(x_pre_tr.shape[0], generator=gen_tr)
            for start in range(0, int(x_pre_tr.shape[0]), tr_bs):
                bx = x_pre_tr[order[start : start + tr_bs]]
                by = y_post_tr[order[start : start + tr_bs]]
                yhat, _ = tr_model(bx, tr_k)
                loss = ((by - yhat) ** two).mean()
                tr_opt.zero_grad()
                loss.backward()
                tr_model.remove_parallel_decoder_grad_()
                tr_opt.step()
                tr_model.normalize_decoder_()

        with torch.no_grad():
            x_pre_te = torch.from_numpy(dataset.l8_pre_mlp_tokens[test_mask])
            y_post_te = dataset.l8_post_mlp_tokens[test_mask]
            yhat_te_t, z_tr_te_t = tr_model(x_pre_te, tr_k)
            yhat_te = yhat_te_t.numpy()
            z_tr_te = z_tr_te_t.numpy()
            w_enc_tr = tr_model.w_enc.detach().cpu().numpy()
            w_dec_tr = tr_model.w_dec.detach().cpu().numpy()
            b_out_tr = tr_model.b_out.detach().cpu().numpy()

        y_var = float(np.mean((y_post_te - y_post_te.mean(axis=0, keepdims=True)) ** two)) + eps
        tr_mse = float(np.mean((y_post_te - yhat_te) ** two))
        tr_r2 = 1.0 - tr_mse / y_var

        mlp_layer = int(cspec["transcoder_layer"])
        late_layer = int(cspec["elk_late_layer"])
        sae_l8 = saes_by_layer[mlp_layer]
        sae_l11 = saes_by_layer[late_layer]

        in_attr = sae_l8.w_dec @ w_enc_tr
        out_attr = w_dec_tr @ sae_l11.w_enc

        max_edges = int(ctspec["max_ui_circuit_edges"])
        flat_order = np.argsort(np.abs(in_attr).ravel())[-max_edges:][::-1]
        circuit_edges: list[dict[str, Any]] = []
        for flat_idx in flat_order:
            src_f = int(flat_idx // in_attr.shape[1])
            tr_f = int(flat_idx % in_attr.shape[1])
            dst_f = int(np.argmax(np.abs(out_attr[tr_f])))
            circuit_edges.append(
                {
                    "source_layer": mlp_layer,
                    "source_sae_feature": src_f,
                    "transcoder_latent": tr_f,
                    "target_layer": late_layer,
                    "target_sae_feature": dst_f,
                    "input_attribution_weight": float(in_attr[src_f, tr_f]),
                    "downstream_attribution_weight": float(out_attr[tr_f, dst_f]),
                }
            )

        n_patch_pairs = min(int(ctspec["patching_pair_count"]), int(z_tr_te.shape[0]) // two)
        patch_k = int(ctspec["patching_top_latents"])
        causal_recoveries: list[float] = []
        random_recoveries: list[float] = []
        rng_patch = np.random.default_rng(seed + int(num["three"]))

        for p_idx in range(n_patch_pairs):
            i_src = p_idx * two
            i_base = p_idx * two + 1
            delta_true = y_post_te[i_src] - y_post_te[i_base]
            d_norm = float(np.linalg.norm(delta_true))
            if d_norm <= eps:
                continue
            unit_dir = delta_true / d_norm
            z_diff = z_tr_te[i_src] - z_tr_te[i_base]
            contrib = z_diff * (w_dec_tr @ unit_dir)
            top_causal = np.argsort(np.abs(contrib))[-patch_k:]

            z_patched_causal = z_tr_te[i_base].copy()
            z_patched_causal[top_causal] = z_tr_te[i_src, top_causal]
            y_base_hat = b_out_tr + z_tr_te[i_base] @ w_dec_tr
            y_causal_hat = b_out_tr + z_patched_causal @ w_dec_tr
            causal_ie = float(np.dot(y_causal_hat - y_base_hat, unit_dir) / d_norm)
            causal_recoveries.append(causal_ie)

            rand_idx = rng_patch.choice(tr_dict, size=patch_k, replace=False)
            z_patched_rand = z_tr_te[i_base].copy()
            z_patched_rand[rand_idx] = z_tr_te[i_src, rand_idx]
            y_rand_hat = b_out_tr + z_patched_rand @ w_dec_tr
            rand_ie = float(np.dot(y_rand_hat - y_base_hat, unit_dir) / d_norm)
            random_recoveries.append(rand_ie)

        transcoder_metrics = {
            "layer": mlp_layer,
            "heldout_r2": tr_r2,
            "heldout_fvu": tr_mse / y_var,
            "heldout_l0": float(np.mean((z_tr_te > 0.0).sum(axis=1))),
            "causal_patching_indirect_effect": float(np.mean(causal_recoveries)) if causal_recoveries else 0.0,
            "random_control_indirect_effect": float(np.mean(random_recoveries)) if random_recoveries else 0.0,
            "causal_vs_random_selectivity_ratio": (
                (float(np.mean(causal_recoveries)) + eps) / (abs(float(np.mean(random_recoveries))) + eps)
            ),
            "exported_circuit_edges": len(circuit_edges),
        }

        return CrosscoderAndTranscoderResult(
            crosscoder_metrics=crosscoder_metrics,
            procrustes_metrics=procrustes_metrics,
            transcoder_metrics=transcoder_metrics,
            atlas_points=atlas_points,
            cross_layer_threads=cross_layer_threads,
            transcoder_circuit_edges=circuit_edges,
        )
