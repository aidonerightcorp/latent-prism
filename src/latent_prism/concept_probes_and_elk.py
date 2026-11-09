"""12-Layer Concept Probe Suite (Mass-Mean, Fisher LDA, Ridge Logistic) and Sheaf-Conformal ELK Contradiction Detector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from latent_prism.activation_corpus import MultiLayerActivationDataset
from latent_prism.grid_and_conformance import NodeGrid


@dataclass
class LinearConceptProbe:
    """Calibrated linear probe mapping d-dimensional hidden states to scalar logits and probabilities."""

    method: str
    layer_idx: int
    regime_name: str
    weight: np.ndarray
    bias: float
    logit_clip: float

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float32) @ self.weight + np.float32(self.bias)).astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = np.clip(self.decision_function(x), -self.logit_clip, self.logit_clip)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC supporting ties without external dependencies."""
    y = labels.astype(np.int64).ravel()
    s = scores.astype(np.float64).ravel()
    pos_scores = s[y == 1]
    neg_scores = s[y == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0
    diffs = pos_scores[:, None] - neg_scores[None, :]
    wins = float(np.sum(diffs > 0.0))
    ties = float(np.sum(diffs == 0.0))
    half_val = 1.0 / (1.0 + 1.0)
    return (wins + half_val * ties) / float(pos_scores.size * neg_scores.size)


def fit_mass_mean_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    layer_idx: int,
    regime_name: str,
    eps: float,
    logit_clip: float,
) -> LinearConceptProbe:
    x64 = x_train.astype(np.float64)
    mu_pos = x64[y_train == 1].mean(axis=0)
    mu_neg = x64[y_train == 0].mean(axis=0)
    diff = mu_pos - mu_neg
    var_scale = float(np.mean(x64.var(axis=0)) + eps)
    w = diff / var_scale
    half_val = 1.0 / (1.0 + 1.0)
    b = float(-half_val * np.dot(w, mu_pos + mu_neg))
    return LinearConceptProbe("mass_mean", layer_idx, regime_name, w.astype(np.float32), b, logit_clip)


def fit_fisher_lda_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    layer_idx: int,
    regime_name: str,
    shrinkage: float,
    eps: float,
    logit_clip: float,
) -> LinearConceptProbe:
    x64 = x_train.astype(np.float64)
    pos = x64[y_train == 1]
    neg = x64[y_train == 0]
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    c_pos = pos - mu_pos[None, :]
    c_neg = neg - mu_neg[None, :]
    denom = max(1, int(x64.shape[0]) - (1 + 1))
    cov = (c_pos.T @ c_pos + c_neg.T @ c_neg) / float(denom)
    dim = int(x64.shape[1])
    trace_mean = float(np.trace(cov)) / float(dim) + eps
    cov_shrunk = (1.0 - shrinkage) * cov + shrinkage * trace_mean * np.eye(dim, dtype=np.float64)
    w = np.linalg.solve(cov_shrunk, mu_pos - mu_neg)
    half_val = 1.0 / (1.0 + 1.0)
    b = float(-half_val * np.dot(w, mu_pos + mu_neg))
    return LinearConceptProbe("fisher_lda", layer_idx, regime_name, w.astype(np.float32), b, logit_clip)


def fit_ridge_logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    layer_idx: int,
    regime_name: str,
    ridge_lambda: float,
    lr: float,
    steps: int,
    eps: float,
    logit_clip: float,
) -> LinearConceptProbe:
    x64 = x_train.astype(np.float64)
    y64 = y_train.astype(np.float64)
    mean = x64.mean(axis=0)
    std = np.sqrt(x64.var(axis=0) + eps)
    xn = (x64 - mean[None, :]) / std[None, :]

    n_samples, dim = xn.shape
    w_std = np.zeros(dim, dtype=np.float64)
    b_std = 0.0

    for _ in range(steps):
        logits = np.clip(xn @ w_std + b_std, -logit_clip, logit_clip)
        probs = 1.0 / (1.0 + np.exp(-logits))
        err = probs - y64
        grad_w = (xn.T @ err) / float(n_samples) + (ridge_lambda / float(n_samples)) * w_std
        grad_b = float(np.mean(err))
        w_std = w_std - lr * grad_w
        b_std = b_std - lr * grad_b

    w_raw = w_std / std
    b_raw = float(b_std - np.dot(w_raw, mean))
    return LinearConceptProbe("ridge_logistic", layer_idx, regime_name, w_raw.astype(np.float32), b_raw, logit_clip)


def _fit_loo_ridge_map(
    src_cal: np.ndarray,
    tgt_cal: np.ndarray,
    ridge_lambda: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit linear restriction map Y ≈ [X, 1] W and return exact leave-one-out calibration residuals."""
    n_cal = int(src_cal.shape[0])
    ones = np.ones((n_cal, 1), dtype=np.float64)
    x_aug = np.concatenate([src_cal.astype(np.float64), ones], axis=1)
    y_mat = tgt_cal.astype(np.float64)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge_lambda)
    reg[-1, -1] = eps
    gram_inv = np.linalg.inv(x_aug.T @ x_aug + reg)
    w_aug = gram_inv @ (x_aug.T @ y_mat)
    y_hat = x_aug @ w_aug
    hat_diag = np.sum((x_aug @ gram_inv) * x_aug, axis=1)
    denom = np.maximum(1.0 - hat_diag, eps)[:, None]
    loo_resid = (y_mat - y_hat) / denom
    return w_aug[:-1].astype(np.float32), w_aug[-1].astype(np.float32), loo_resid.astype(np.float32)


@dataclass
class ProbeAndElkSuiteResult:
    """12-layer probe curves, fitted probe dictionary, and Sheaf-Conformal ELK contradiction metrics."""

    layerwise_curves: list[dict[str, Any]]
    probes_by_regime_layer: dict[tuple[str, int], dict[str, LinearConceptProbe]]
    elk_metrics: dict[str, Any]
    summary: dict[str, Any]


class CalibratedLatentConceptProbeSuite:
    """Evaluates Mass-Mean, Fisher LDA, and Ridge Logistic across L0..L11 + Sheaf-Conformal ELK detection."""

    def __init__(self, grid: NodeGrid) -> None:
        self.grid = grid

    def run(self, dataset: MultiLayerActivationDataset) -> ProbeAndElkSuiteResult:
        cspec = self.grid.raw_spec["backbone_and_corpus_spec"]
        pspec = self.grid.raw_spec["probe_and_elk_spec"]
        num = self.grid.raw_spec["numerical_constants"]
        eps = float(num["eps"])
        two = int(num["two"])
        logit_clip = float(num["logit_clip"])
        train_frac = float(cspec["train_fraction"])
        seed = int(cspec["default_seed"])

        shrinkage = float(pspec["lda_shrinkage"])
        ridge_lam = float(pspec["ridge_lambda"])
        log_lr = float(pspec["logistic_lr"])
        log_steps = int(pspec["logistic_steps"])

        regimes = sorted({r.regime_name for r in dataset.records if r.group_kind == "contrastive"})
        probes_store: dict[tuple[str, int], dict[str, LinearConceptProbe]] = {}
        per_layer_method_auroc: dict[int, dict[str, list[float]]] = {
            l_idx: {"mass_mean": [], "fisher_lda": [], "ridge_logistic": []} for l_idx in range(dataset.num_layers)
        }
        per_layer_method_acc: dict[int, dict[str, list[float]]] = {
            l_idx: {"mass_mean": [], "fisher_lda": [], "ridge_logistic": []} for l_idx in range(dataset.num_layers)
        }

        rng = np.random.default_rng(seed)
        for rname in regimes:
            r_recs = [r for r in dataset.records if r.group_kind == "contrastive" and r.regime_name == rname]
            pair_ids = sorted({r.pair_id for r in r_recs})
            perm_pairs = rng.permutation(pair_ids)
            n_tr_pairs = max(1, int(round(train_frac * float(len(perm_pairs)))))
            tr_pair_set = set(perm_pairs[:n_tr_pairs].tolist())

            tr_ids = [r.prompt_id for r in r_recs if r.pair_id in tr_pair_set]
            te_ids = [r.prompt_id for r in r_recs if r.pair_id not in tr_pair_set]
            y_tr = np.array([dataset.records[i].label for i in tr_ids], dtype=np.int64)
            y_te = np.array([dataset.records[i].label for i in te_ids], dtype=np.int64)

            for l_idx in range(dataset.num_layers):
                x_tr = dataset.pooled_all_layers[tr_ids, l_idx, :]
                x_te = dataset.pooled_all_layers[te_ids, l_idx, :]

                p_mm = fit_mass_mean_probe(
                    x_tr, y_tr, layer_idx=l_idx, regime_name=rname, eps=eps, logit_clip=logit_clip
                )
                p_lda = fit_fisher_lda_probe(
                    x_tr, y_tr, layer_idx=l_idx, regime_name=rname, shrinkage=shrinkage, eps=eps, logit_clip=logit_clip
                )
                p_log = fit_ridge_logistic_probe(
                    x_tr,
                    y_tr,
                    layer_idx=l_idx,
                    regime_name=rname,
                    ridge_lambda=ridge_lam,
                    lr=log_lr,
                    steps=log_steps,
                    eps=eps,
                    logit_clip=logit_clip,
                )
                probes_store[(rname, l_idx)] = {
                    "mass_mean": p_mm,
                    "fisher_lda": p_lda,
                    "ridge_logistic": p_log,
                }
                for m_key, probe in probes_store[(rname, l_idx)].items():
                    probs = probe.predict_proba(x_te)
                    preds = (probs >= float(num["half"])).astype(np.int64)
                    per_layer_method_auroc[l_idx][m_key].append(binary_auroc(y_te, probs))
                    per_layer_method_acc[l_idx][m_key].append(float(np.mean(preds == y_te)))

        mid_layers = [int(x) for x in cspec["elk_mid_layers"]]
        l_mid_a, l_mid_b = mid_layers[0], mid_layers[1]
        l_late = int(cspec["elk_late_layer"])

        cal_ids = [r.prompt_id for r in dataset.elk_records if r.group_kind == "elk_calibration_clean"]
        clean_eval_ids = [r.prompt_id for r in dataset.elk_records if r.group_kind == "elk_eval_clean"]
        syc_eval_ids = [r.prompt_id for r in dataset.elk_records if r.group_kind == "elk_eval_sycophantic"]

        y_cal_truth = np.array([dataset.elk_records[i].label for i in cal_ids], dtype=np.int64)
        y_clean_truth = np.array([dataset.elk_records[i].label for i in clean_eval_ids], dtype=np.int64)
        y_syc_truth = np.array([dataset.elk_records[i].label for i in syc_eval_ids], dtype=np.int64)

        mid_truth_probe = fit_ridge_logistic_probe(
            dataset.elk_prefix_states[l_mid_b][cal_ids],
            y_cal_truth,
            layer_idx=l_mid_b,
            regime_name="elk_internal_truth",
            ridge_lambda=ridge_lam,
            lr=log_lr,
            steps=log_steps,
            eps=eps,
            logit_clip=logit_clip,
        )
        late_eos_truth_probe = fit_ridge_logistic_probe(
            dataset.elk_eos_states[l_late][cal_ids],
            y_cal_truth,
            layer_idx=l_late,
            regime_name="elk_late_eos_truth",
            ridge_lambda=ridge_lam,
            lr=log_lr,
            steps=log_steps,
            eps=eps,
            logit_clip=logit_clip,
        )

        mid_auroc_clean = binary_auroc(
            y_clean_truth, mid_truth_probe.predict_proba(dataset.elk_prefix_states[l_mid_b][clean_eval_ids])
        )
        mid_auroc_syc = binary_auroc(
            y_syc_truth, mid_truth_probe.predict_proba(dataset.elk_prefix_states[l_mid_b][syc_eval_ids])
        )
        late_auroc_clean = binary_auroc(
            y_clean_truth, late_eos_truth_probe.predict_proba(dataset.elk_eos_states[l_late][clean_eval_ids])
        )
        late_auroc_syc = binary_auroc(
            y_syc_truth, late_eos_truth_probe.predict_proba(dataset.elk_eos_states[l_late][syc_eval_ids])
        )

        proj_dim = int(pspec["sheaf_proj_dim"])
        sheaf_lam = float(pspec["sheaf_ridge_lambda"])
        disagree_w = float(pspec["elk_disagreement_weight"])
        conf_alpha = float(pspec["conformal_alpha"])

        def _pca_basis(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mu = mat.mean(axis=0)
            _, _, vt = np.linalg.svd(mat - mu[None, :], full_matrices=False)
            k_eff = min(proj_dim, int(vt.shape[0]))
            return mu.astype(np.float32), vt[:k_eff].T.astype(np.float32)

        mu_a, basis_a = _pca_basis(dataset.elk_prefix_states[l_mid_a][cal_ids])
        mu_b, basis_b = _pca_basis(dataset.elk_prefix_states[l_mid_b][cal_ids])
        mu_c, basis_c = _pca_basis(dataset.elk_eos_states[l_late][cal_ids])

        za_cal = (dataset.elk_prefix_states[l_mid_a][cal_ids] - mu_a) @ basis_a
        zb_cal = (dataset.elk_prefix_states[l_mid_b][cal_ids] - mu_b) @ basis_b
        zc_cal = (dataset.elk_eos_states[l_late][cal_ids] - mu_c) @ basis_c

        r_ab, b_ab, loo_ab = _fit_loo_ridge_map(za_cal, zb_cal, sheaf_lam, eps)
        r_bc, b_bc, loo_bc = _fit_loo_ridge_map(zb_cal, zc_cal, sheaf_lam, eps)

        syc_framing_probe = fit_ridge_logistic_probe(
            dataset.eos_all_layers[
                [r.prompt_id for r in dataset.records if r.group_kind == "contrastive" and r.regime_name == "factual_vs_sycophantic"],
                l_late,
                :,
            ],
            np.array(
                [r.label for r in dataset.records if r.group_kind == "contrastive" and r.regime_name == "factual_vs_sycophantic"],
                dtype=np.int64,
            ),
            layer_idx=l_late,
            regime_name="eos_sycophancy_framing",
            ridge_lambda=ridge_lam,
            lr=log_lr,
            steps=log_steps,
            eps=eps,
            logit_clip=logit_clip,
        )
        cal_syc_prob = 1.0 - syc_framing_probe.predict_proba(dataset.elk_eos_states[l_late][cal_ids])
        cal_mid_p = mid_truth_probe.predict_proba(dataset.elk_prefix_states[l_mid_b][cal_ids])
        cal_late_p = late_eos_truth_probe.predict_proba(dataset.elk_eos_states[l_late][cal_ids])

        cal_scores = (
            np.mean(loo_ab ** two, axis=1)
            + np.mean(loo_bc ** two, axis=1)
            + disagree_w * ((cal_mid_p - cal_late_p) ** two + cal_syc_prob ** two)
        )

        n_cal = int(cal_scores.shape[0])
        rank_idx = min(n_cal - 1, max(0, int(math.ceil(float(n_cal + 1) * (1.0 - conf_alpha))) - 1))
        conformal_threshold = float(np.sort(cal_scores)[rank_idx])

        def _eval_sheaf_energy(indices: list[int]) -> np.ndarray:
            za = (dataset.elk_prefix_states[l_mid_a][indices] - mu_a) @ basis_a
            zb = (dataset.elk_prefix_states[l_mid_b][indices] - mu_b) @ basis_b
            zc = (dataset.elk_eos_states[l_late][indices] - mu_c) @ basis_c
            e_ab = np.mean((zb - (za @ r_ab + b_ab[None, :])) ** two, axis=1)
            e_bc = np.mean((zc - (zb @ r_bc + b_bc[None, :])) ** two, axis=1)
            syc_p = 1.0 - syc_framing_probe.predict_proba(dataset.elk_eos_states[l_late][indices])
            p_m = mid_truth_probe.predict_proba(dataset.elk_prefix_states[l_mid_b][indices])
            p_l = late_eos_truth_probe.predict_proba(dataset.elk_eos_states[l_late][indices])
            return (e_ab + e_bc + disagree_w * ((p_m - p_l) ** two + syc_p ** two)).astype(np.float32)

        clean_energies = _eval_sheaf_energy(clean_eval_ids)
        syc_energies = _eval_sheaf_energy(syc_eval_ids)

        empirical_clean_coverage = float(np.mean(clean_energies <= conformal_threshold))
        sycophancy_flag_rate = float(np.mean(syc_energies > conformal_threshold))
        combined_y = np.concatenate(
            [np.zeros(clean_energies.shape[0], dtype=np.int64), np.ones(syc_energies.shape[0], dtype=np.int64)]
        )
        combined_s = np.concatenate([clean_energies, syc_energies])
        elk_detection_auroc = binary_auroc(combined_y, combined_s)

        layerwise_curves: list[dict[str, Any]] = []
        for l_idx in range(dataset.num_layers):
            syc_probe_l = probes_store[("factual_vs_sycophantic", l_idx)]["ridge_logistic"]
            layerwise_curves.append(
                {
                    "layer": int(l_idx),
                    "mass_mean_auroc": float(np.mean(per_layer_method_auroc[l_idx]["mass_mean"])),
                    "fisher_lda_auroc": float(np.mean(per_layer_method_auroc[l_idx]["fisher_lda"])),
                    "ridge_logistic_auroc": float(np.mean(per_layer_method_auroc[l_idx]["ridge_logistic"])),
                    "ridge_logistic_accuracy": float(np.mean(per_layer_method_acc[l_idx]["ridge_logistic"])),
                    "sycophancy_framing_auroc": binary_auroc(
                        np.array(
                            [
                                r.label
                                for r in dataset.records
                                if r.group_kind == "contrastive" and r.regime_name == "factual_vs_sycophantic"
                            ],
                            dtype=np.int64,
                        ),
                        syc_probe_l.predict_proba(
                            dataset.pooled_all_layers[
                                [
                                    r.prompt_id
                                    for r in dataset.records
                                    if r.group_kind == "contrastive" and r.regime_name == "factual_vs_sycophantic"
                                ],
                                l_idx,
                                :,
                            ]
                        ),
                    ),
                }
            )

        elk_metrics = {
            "conformal_alpha": conf_alpha,
            "conformal_threshold": conformal_threshold,
            "clean_empirical_coverage": empirical_clean_coverage,
            "target_min_coverage": float(pspec["min_target_coverage"]),
            "contradiction_detection_auroc": elk_detection_auroc,
            "contradiction_flag_rate": sycophancy_flag_rate,
            "mid_layer_clean_truth_auroc": mid_auroc_clean,
            "mid_layer_pressured_truth_auroc": mid_auroc_syc,
            "mid_layer_truth_invariance_abs_gap": abs(mid_auroc_clean - mid_auroc_syc),
            "late_eos_clean_truth_auroc": late_auroc_clean,
            "late_eos_pressured_truth_auroc": late_auroc_syc,
            "mean_clean_sheaf_energy": float(np.mean(clean_energies)),
            "mean_pressured_sheaf_energy": float(np.mean(syc_energies)),
        }

        best_layer_entry = max(layerwise_curves, key=lambda row: row["ridge_logistic_auroc"])
        summary = {
            "num_layers_probed": dataset.num_layers,
            "num_contrastive_regimes": len(regimes),
            "peak_probe_layer": int(best_layer_entry["layer"]),
            "peak_ridge_logistic_auroc": float(best_layer_entry["ridge_logistic_auroc"]),
            "peak_fisher_lda_auroc": float(max(r["fisher_lda_auroc"] for r in layerwise_curves)),
            "peak_mass_mean_auroc": float(max(r["mass_mean_auroc"] for r in layerwise_curves)),
            "elk_contradiction_detector": elk_metrics,
        }

        return ProbeAndElkSuiteResult(
            layerwise_curves=layerwise_curves,
            probes_by_regime_layer=probes_store,
            elk_metrics=elk_metrics,
            summary=summary,
        )
