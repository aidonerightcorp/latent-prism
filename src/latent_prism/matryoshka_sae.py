"""Matryoshka TopK + AuxK + JumpReLU Sparse Autoencoders with parallel-gradient removal and SAEBench/CE-Bench metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from latent_prism.activation_corpus import MultiLayerActivationDataset
from latent_prism.grid_and_conformance import NodeGrid


@dataclass
class TrainedLayerSAE:
    """Trained Sparse Autoencoder parameters, JumpReLU thresholds, and held-out evaluation metrics."""

    layer_idx: int
    variant: str
    input_dim: int
    dict_size: int
    top_k: int
    prefixes: tuple[int, ...]
    b_pre: np.ndarray
    w_enc: np.ndarray
    b_enc: np.ndarray
    w_dec: np.ndarray
    jumprelu_theta: np.ndarray
    metrics: dict[str, Any]

    def pre_activations(self, x: np.ndarray) -> np.ndarray:
        centered = x.astype(np.float32) - self.b_pre[None, :]
        raw = centered @ self.w_enc + self.b_enc[None, :]
        return np.maximum(raw, 0.0)

    def encode_topk(self, x: np.ndarray, *, prefix_width: int | None = None, custom_k: int | None = None) -> np.ndarray:
        m = int(prefix_width) if prefix_width is not None else self.dict_size
        k_eff = (
            int(custom_k)
            if custom_k is not None
            else max(1, int(round(float(self.top_k) * float(m) / float(self.dict_size))))
        )
        k_eff = min(k_eff, m)
        pre = self.pre_activations(x)[:, :m]
        out = np.zeros_like(pre, dtype=np.float32)
        if k_eff <= 0:
            return out
        top_idx = np.argpartition(pre, -k_eff, axis=1)[:, -k_eff:]
        rows = np.arange(pre.shape[0])[:, None]
        out[rows, top_idx] = pre[rows, top_idx]
        return out

    def encode_jumprelu(self, x: np.ndarray) -> np.ndarray:
        centered = x.astype(np.float32) - self.b_pre[None, :]
        raw = centered @ self.w_enc + self.b_enc[None, :]
        mask = raw > self.jumprelu_theta[None, :]
        return np.where(mask, np.maximum(raw, 0.0), 0.0).astype(np.float32)

    def reconstruct_topk(self, x: np.ndarray, *, prefix_width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        m = int(prefix_width) if prefix_width is not None else self.dict_size
        z = self.encode_topk(x, prefix_width=m)
        x_hat = self.b_pre[None, :] + z @ self.w_dec[:m]
        return x_hat.astype(np.float32), z


class _TorchMatryoshkaSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        init_mean: torch.Tensor,
        seed: int,
        eps: float,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.eps = float(eps)
        w_init = torch.randn((input_dim, dict_size), generator=gen, dtype=torch.float32)
        w_init = w_init / w_init.norm(dim=0, keepdim=True).clamp_min(self.eps)

        self.b_pre = nn.Parameter(init_mean.clone().to(torch.float32))
        self.w_enc = nn.Parameter(w_init.clone())
        self.b_enc = nn.Parameter(torch.zeros(dict_size, dtype=torch.float32))
        self.w_dec = nn.Parameter(w_init.t().contiguous().clone())

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

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_pre) @ self.w_enc + self.b_enc)

    @staticmethod
    def sparse_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
        k_eff = min(max(1, int(k)), int(pre.shape[1]))
        vals, idx = torch.topk(pre, k=k_eff, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z


def _calibrate_jumprelu_thresholds(pre_train: np.ndarray, target_k: int, eps: float) -> np.ndarray:
    """Calibrate featurewise JumpReLU thresholds matching expected TopK L0 sparsity."""
    n_samples, dict_size = pre_train.shape
    total_target_fires = max(1, n_samples * int(target_k))
    flat = pre_train.reshape(-1)
    if total_target_fires >= flat.size:
        return np.full(dict_size, eps, dtype=np.float32)
    global_cut = float(np.partition(flat, -total_target_fires)[-total_target_fires])
    thresh = max(eps, global_cut)
    return np.full(dict_size, thresh, dtype=np.float32)


def _compute_ce_bench_selectivity_and_splitting(
    sae: TrainedLayerSAE,
    pooled_layer: np.ndarray,
    dataset: MultiLayerActivationDataset,
    grid: NodeGrid,
    *,
    prefix_limit: int | None = None,
) -> dict[str, float]:
    """Compute automated CE-Bench contrastive monosemanticity selectivity and feature-splitting cosine."""
    eps = float(grid.raw_spec["numerical_constants"]["eps"])
    sspec = grid.raw_spec["sae_spec"]
    top_pool = int(sspec["splitting_top_pool"])
    z_all = sae.encode_topk(pooled_layer, prefix_width=prefix_limit)
    w_dec_slice = sae.w_dec[: z_all.shape[1]]

    regime_names = sorted({r.regime_name for r in dataset.records if r.group_kind == "contrastive"})
    best_selectivities: list[float] = []
    splitting_cosines: list[float] = []

    for rname in regime_names:
        pos_idx = [r.prompt_id for r in dataset.records if r.group_kind == "contrastive" and r.regime_name == rname and r.label == 1]
        neg_idx = [r.prompt_id for r in dataset.records if r.group_kind == "contrastive" and r.regime_name == rname and r.label == 0]
        z_pos = z_all[pos_idx]
        z_neg = z_all[neg_idx]
        mu_pos = z_pos.mean(axis=0)
        mu_neg = z_neg.mean(axis=0)
        std_pool = np.sqrt(z_pos.var(axis=0) + z_neg.var(axis=0) + eps)
        fire_pos = (z_pos > 0.0).mean(axis=0)
        fire_neg = (z_neg > 0.0).mean(axis=0)
        precision_like = fire_pos / (fire_pos + fire_neg + eps)
        effect = (mu_pos - mu_neg) / (mu_pos + mu_neg + std_pool + eps)
        selectivity_vec = np.clip(precision_like * np.maximum(effect, 0.0), 0.0, 1.0)
        best_selectivities.append(float(selectivity_vec.max()))

        pool_k = min(top_pool, int(selectivity_vec.shape[0]))
        if pool_k > 1:
            top_feat_idx = np.argsort(selectivity_vec)[-pool_k:]
            atoms = w_dec_slice[top_feat_idx]
            gram = np.abs(atoms @ atoms.T)
            np.fill_diagonal(gram, 0.0)
            splitting_cosines.append(float(gram.max()))

    return {
        "contrastive_selectivity_score": float(np.mean(best_selectivities)) if best_selectivities else 0.0,
        "feature_splitting_max_cosine": float(np.mean(splitting_cosines)) if splitting_cosines else 0.0,
    }


def train_and_evaluate_layer_sae(
    layer_idx: int,
    tokens_layer: np.ndarray,
    token_prompt_indices: np.ndarray,
    pooled_layer: np.ndarray,
    dataset: MultiLayerActivationDataset,
    grid: NodeGrid,
    cell: dict[str, Any],
    *,
    enable_matryoshka: bool = True,
    seed_offset: int = 0,
) -> TrainedLayerSAE:
    """Train a unit-norm-decoder TopK + AuxK + JumpReLU SAE (Matryoshka or Flat TopK) and evaluate on held-out split."""
    cspec = grid.raw_spec["backbone_and_corpus_spec"]
    sspec = grid.raw_spec["sae_spec"]
    num = grid.raw_spec["numerical_constants"]
    eps = float(num["eps"])
    torch.set_num_threads(int(num["torch_threads"]))

    dict_size = int(cell["sae_dictionary_width"])
    m_coarse = int(cell["matryoshka_coarse_prefix"])
    m_mid = int(cell["matryoshka_mid_prefix"])
    prefixes = (m_coarse, m_mid, dict_size)
    weights = [float(w) for w in sspec["matryoshka_weights"]] if enable_matryoshka else [0.0, 0.0, 1.0]

    top_k = int(cell["sae_top_k"])
    aux_k = int(cell["aux_k_dead_latents"])
    aux_weight = float(sspec["aux_loss_weight"])
    dead_thresh = int(sspec["dead_step_threshold"])
    lr = float(sspec["learning_rate"])
    epochs = int(sspec["epochs"])
    batch_size = int(sspec["batch_size"])
    grad_clip = float(sspec["grad_clip_norm"])
    parallel_proj = bool(cell["enable_parallel_grad_projection"])
    seed = int(cspec["default_seed"]) + int(layer_idx) + int(seed_offset)
    train_frac = float(cspec["train_fraction"])

    unique_prompts = np.unique(token_prompt_indices)
    rng = np.random.default_rng(seed)
    perm_prompts = rng.permutation(unique_prompts)
    n_train_prompts = max(1, int(round(train_frac * float(len(perm_prompts)))))
    train_prompt_set = set(perm_prompts[:n_train_prompts].tolist())
    train_mask = np.array([int(pid) in train_prompt_set for pid in token_prompt_indices], dtype=bool)
    test_mask = ~train_mask

    x_train_np = tokens_layer[train_mask].astype(np.float32)
    x_test_np = tokens_layer[test_mask].astype(np.float32)

    torch.manual_seed(seed)
    x_train_t = torch.from_numpy(x_train_np)
    init_mean = x_train_t.mean(dim=0)

    model = _TorchMatryoshkaSAE(
        input_dim=int(tokens_layer.shape[1]),
        dict_size=dict_size,
        init_mean=init_mean,
        seed=seed,
        eps=eps,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps_since_fired = torch.zeros(dict_size, dtype=torch.long)

    gen = torch.Generator().manual_seed(seed)
    n_train = int(x_train_t.shape[0])

    for _ in range(epochs):
        order = torch.randperm(n_train, generator=gen)
        for start in range(0, n_train, batch_size):
            batch = x_train_t[order[start : start + batch_size]]
            pre = model.pre_acts(batch)

            recon_loss = torch.tensor(0.0, dtype=torch.float32)
            full_hat = batch
            fired_any = torch.zeros(dict_size, dtype=torch.bool)

            for p_width, p_weight in zip(prefixes, weights):
                if p_weight <= 0.0 and p_width != dict_size:
                    continue
                k_p = max(1, int(round(float(top_k) * float(p_width) / float(dict_size))))
                z_p = model.sparse_topk(pre[:, :p_width], k_p)
                hat_p = model.b_pre + z_p @ model.w_dec[:p_width]
                if p_width == dict_size:
                    full_hat = hat_p
                    fired_any = (z_p > 0.0).any(dim=0)
                if p_weight > 0.0:
                    recon_loss = recon_loss + float(p_weight) * ((batch - hat_p) ** int(num["two"])).mean()

            steps_since_fired = torch.where(fired_any, torch.zeros_like(steps_since_fired), steps_since_fired + 1)
            dead_mask = steps_since_fired >= dead_thresh
            aux_loss = torch.tensor(0.0, dtype=torch.float32)
            n_dead = int(dead_mask.sum().item())
            if n_dead > 0 and aux_weight > 0.0:
                resid = (batch - full_hat).detach()
                dead_pre = pre[:, dead_mask]
                z_dead = model.sparse_topk(dead_pre, min(aux_k, n_dead))
                aux_hat = z_dead @ model.w_dec[dead_mask]
                aux_loss = ((resid - aux_hat) ** int(num["two"])).mean()

            loss = recon_loss + aux_weight * aux_loss
            optimizer.zero_grad()
            loss.backward()
            if parallel_proj:
                model.remove_parallel_decoder_grad_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.normalize_decoder_()

    with torch.no_grad():
        b_pre_np = model.b_pre.detach().cpu().numpy().astype(np.float32)
        w_enc_np = model.w_enc.detach().cpu().numpy().astype(np.float32)
        b_enc_np = model.b_enc.detach().cpu().numpy().astype(np.float32)
        w_dec_np = model.w_dec.detach().cpu().numpy().astype(np.float32)

    temp_sae = TrainedLayerSAE(
        layer_idx=layer_idx,
        variant="matryoshka_topk_auxk" if enable_matryoshka else "flat_topk_baseline",
        input_dim=int(tokens_layer.shape[1]),
        dict_size=dict_size,
        top_k=top_k,
        prefixes=prefixes,
        b_pre=b_pre_np,
        w_enc=w_enc_np,
        b_enc=b_enc_np,
        w_dec=w_dec_np,
        jumprelu_theta=np.zeros(dict_size, dtype=np.float32),
        metrics={},
    )
    pre_train_np = temp_sae.pre_activations(x_train_np)
    theta_np = _calibrate_jumprelu_thresholds(pre_train_np, top_k, eps)
    temp_sae.jumprelu_theta = theta_np

    total_var = float(np.mean((x_test_np - x_test_np.mean(axis=0, keepdims=True)) ** int(num["two"]))) + eps
    x_hat_full, z_test = temp_sae.reconstruct_topk(x_test_np)
    mse_full = float(np.mean((x_test_np - x_hat_full) ** int(num["two"])))
    fvu = mse_full / total_var
    r2_full = 1.0 - fvu
    l0_topk = float(np.mean((z_test > 0.0).sum(axis=1)))

    z_all_eval = temp_sae.encode_topk(tokens_layer)
    dead_rate = float(np.mean((z_all_eval > 0.0).sum(axis=0) == 0))

    z_jump = temp_sae.encode_jumprelu(x_test_np)
    x_hat_jump = temp_sae.b_pre[None, :] + z_jump @ temp_sae.w_dec
    fvu_jump = float(np.mean((x_test_np - x_hat_jump) ** int(num["two"]))) / total_var
    r2_jump = 1.0 - fvu_jump
    l0_jump = float(np.mean((z_jump > 0.0).sum(axis=1)))

    prefix_r2: dict[str, float] = {}
    for p_width in prefixes:
        hat_p, _ = temp_sae.reconstruct_topk(x_test_np, prefix_width=p_width)
        mse_p = float(np.mean((x_test_np - hat_p) ** int(num["two"])))
        prefix_r2[f"prefix_{p_width}"] = 1.0 - (mse_p / total_var)

    ce_full = _compute_ce_bench_selectivity_and_splitting(temp_sae, pooled_layer, dataset, grid)
    ce_coarse = _compute_ce_bench_selectivity_and_splitting(
        temp_sae, pooled_layer, dataset, grid, prefix_limit=m_coarse
    )
    decoder_unit_norm_err = float(np.max(np.abs(np.linalg.norm(temp_sae.w_dec, axis=1) - 1.0)))

    temp_sae.metrics = {
        "layer": int(layer_idx),
        "variant": temp_sae.variant,
        "heldout_fvu": fvu,
        "heldout_r2": r2_full,
        "heldout_l0": l0_topk,
        "dead_latent_rate": dead_rate,
        "jumprelu_r2": r2_jump,
        "jumprelu_l0": l0_jump,
        "nested_prefix_r2": prefix_r2,
        "contrastive_selectivity_score": ce_full["contrastive_selectivity_score"],
        "coarse_prefix_selectivity_score": ce_coarse["contrastive_selectivity_score"],
        "feature_splitting_max_cosine": ce_full["feature_splitting_max_cosine"],
        "coarse_prefix_splitting_cosine": ce_coarse["feature_splitting_max_cosine"],
        "decoder_unit_norm_max_error": decoder_unit_norm_err,
    }
    return temp_sae


class MatryoshkaTopKAndJumpReluSuite:
    """Trains Matryoshka TopK+AuxK+JumpReLU SAEs across [L2, L5, L8, L11] and compares against Flat TopK."""

    def __init__(self, grid: NodeGrid) -> None:
        self.grid = grid

    def run(self, dataset: MultiLayerActivationDataset, cell: dict[str, Any]) -> dict[str, Any]:
        saes_by_layer: dict[int, TrainedLayerSAE] = {}
        layer_reports: dict[str, Any] = {}

        for l_idx in dataset.target_layers:
            sae = train_and_evaluate_layer_sae(
                layer_idx=l_idx,
                tokens_layer=dataset.token_states_by_layer[l_idx],
                token_prompt_indices=dataset.token_prompt_indices,
                pooled_layer=dataset.pooled_all_layers[:, l_idx, :],
                dataset=dataset,
                grid=self.grid,
                cell=cell,
                enable_matryoshka=True,
            )
            saes_by_layer[l_idx] = sae
            layer_reports[f"L{l_idx}"] = sae.metrics

        comp_layer = int(self.grid.raw_spec["backbone_and_corpus_spec"]["transcoder_layer"])
        flat_sae = train_and_evaluate_layer_sae(
            layer_idx=comp_layer,
            tokens_layer=dataset.token_states_by_layer[comp_layer],
            token_prompt_indices=dataset.token_prompt_indices,
            pooled_layer=dataset.pooled_all_layers[:, comp_layer, :],
            dataset=dataset,
            grid=self.grid,
            cell=cell,
            enable_matryoshka=False,
        )
        mat_sae = saes_by_layer[comp_layer]
        m_coarse = int(cell["matryoshka_coarse_prefix"])
        coarse_key = f"prefix_{m_coarse}"

        comparison = {
            "comparison_layer": comp_layer,
            "matryoshka_coarse_prefix_r2": mat_sae.metrics["nested_prefix_r2"][coarse_key],
            "flat_topk_coarse_prefix_r2": flat_sae.metrics["nested_prefix_r2"][coarse_key],
            "coarse_prefix_r2_gain": (
                mat_sae.metrics["nested_prefix_r2"][coarse_key] - flat_sae.metrics["nested_prefix_r2"][coarse_key]
            ),
            "matryoshka_coarse_selectivity": mat_sae.metrics["coarse_prefix_selectivity_score"],
            "flat_topk_coarse_selectivity": flat_sae.metrics["coarse_prefix_selectivity_score"],
            "matryoshka_splitting_cosine": mat_sae.metrics["coarse_prefix_splitting_cosine"],
            "flat_topk_splitting_cosine": flat_sae.metrics["coarse_prefix_splitting_cosine"],
        }

        return {
            "saes_by_layer": saes_by_layer,
            "flat_baseline_sae": flat_sae,
            "summary": {
                "layers": layer_reports,
                "mean_heldout_r2": float(np.mean([s.metrics["heldout_r2"] for s in saes_by_layer.values()])),
                "mean_heldout_fvu": float(np.mean([s.metrics["heldout_fvu"] for s in saes_by_layer.values()])),
                "mean_dead_latent_rate": float(np.mean([s.metrics["dead_latent_rate"] for s in saes_by_layer.values()])),
                "matryoshka_vs_flat_topk": comparison,
            },
        }
