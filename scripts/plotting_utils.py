"""Reusable plotting helpers for the classifier resampling workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("science")


def plot_predictions_histogram(yhat_sim, yhat_truth, cal_sim, cal_truth, output_dir):
    """Plot raw and calibrated score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(yhat_sim, bins=50, alpha=0.5, range=(0, 1), label="Simulated", density=True)
    axes[0].hist(yhat_truth, bins=50, alpha=0.5, range=(0, 1), label="Truth", density=True)
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Raw Predictions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(cal_sim, bins=50, alpha=0.5, range=(0, 1), label="Simulated", density=True)
    axes[1].hist(cal_truth, bins=50, alpha=0.5, range=(0, 1), label="Truth", density=True)
    axes[1].set_xlabel("Calibrated Probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Calibrated Predictions")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "predictions_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curves(metrics, output_dir):
    """Plot ROC curves before and after calibration."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(metrics["fpr_raw"], metrics["tpr_raw"], label=f"Raw (AUC = {metrics['auc_raw']:.4f})", lw=2, alpha=0.7)
    ax.plot(metrics["fpr_cal"], metrics["tpr_cal"], label=f"Calibrated (AUC = {metrics['auc_cal']:.4f})", lw=2, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", lw=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_calibration_curves(yhat_sim, yhat_truth, y_sim, y_truth, cal_sim, cal_truth, output_dir):
    """Plot calibration curves (reliability diagrams)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def plot_calibration(ax, scores, labels, title):
        score_count, bins = np.histogram(scores, bins=50, range=(0, 1), density=False)
        weighted_count, _ = np.histogram(scores, weights=labels, bins=50, range=(0, 1), density=False)

        n_total = len(labels)
        n_pos = np.sum(labels)
        if n_pos > 0:
            weighted_count = weighted_count * n_total / n_pos / 2

        calibration_curve = np.divide(weighted_count, score_count, out=np.zeros_like(weighted_count), where=score_count != 0)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.plot(bin_centers, calibration_curve, "o-", alpha=0.7, lw=2, markersize=4)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Empirical Probability")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    all_labels_raw = np.concatenate([y_sim, y_truth])
    all_scores_raw = np.concatenate([yhat_sim, yhat_truth])
    all_labels_cal = np.concatenate([y_sim, y_truth])
    all_scores_cal = np.concatenate([cal_sim, cal_truth])

    plot_calibration(axes[0], all_scores_raw, all_labels_raw, "Raw Predictions")
    plot_calibration(axes[1], all_scores_cal, all_labels_cal, "Calibrated Predictions")

    plt.tight_layout()
    path = Path(output_dir) / "calibration_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_hlf_comparison(X_hlf_sim, X_hlf_truth, mask, output_dir, hlf_name_map=None, X_hlf_sim_resampled=None):
    """Plot high-level features into separate PDFs by semantic group.

    Args:
        X_hlf_sim: Original simulated HLF array.
        X_hlf_truth: Truth HLF array.
        mask: Boolean mask over `X_hlf_sim` (legacy path).
        output_dir: Directory to save plots.
        hlf_name_map: Optional feature-name map.
        X_hlf_sim_resampled: Optional direct resampled HLF array. If provided,
            this is used instead of `X_hlf_sim[mask]`.
    """
    if X_hlf_sim_resampled is not None:
        X_hlf_sim_resampled = np.asarray(X_hlf_sim_resampled)
    elif mask is not None:
        X_hlf_sim_resampled = X_hlf_sim[mask]
    else:
        raise ValueError("Provide either `mask` or `X_hlf_sim_resampled`.")
    num_features = X_hlf_sim.shape[1]

    if hlf_name_map is None:
        hlf_name_map = {i: f"HLF {i}" for i in range(num_features)}

    def feature_range(i):
        """Return predefined range for known HLF indices, None for auto-detect."""
        if i == 0:
            return (0, 1.0e5)
        if i in [1, 2, 6, 10]:
            return (0, 2.0e3)
        if i in [3]:
            return (0, 7.5e3)
        if i in [7]:
            return (0, 2.5e4)
        if i in [4, 8]:
            return (0, 1.5e4)
        if i in [5, 9]:
            return (0, 5.0e3)
        if i in [11, 12]:
            return (-11, 11)
        if i in [13, 14]:
            return (-14, 14)
        if i == 15:
            return (-15, 15)
        if i == 40:
            return (0, 300.0)
        return None

    groups = {
        "energy": [],
        "eta_centers": [],
        "phi_centers": [],
        "eta_widths": [],
        "phi_widths": [],
        "sparsity": [],
        "others": [],
    }
    for i in range(num_features):
        name = hlf_name_map.get(i, "")
        if name.startswith("$E_{") or "E_{\\mathrm{tot}}" in name:
            groups["energy"].append(i)
        elif "\\langle" in name and "\\eta" in name:
            groups["eta_centers"].append(i)
        elif "\\langle" in name and "\\phi" in name:
            groups["phi_centers"].append(i)
        elif "\\sigma" in name and "\\eta" in name:
            groups["eta_widths"].append(i)
        elif "\\sigma" in name and "\\phi" in name:
            groups["phi_widths"].append(i)
        elif "\\lambda" in name:
            groups["sparsity"].append(i)
        else:
            groups["others"].append(i)

    def plot_group(indices, group_label):
        if len(indices) == 0:
            return

        ng = len(indices)
        ncols = 4
        nrows = (ng + ncols - 1) // ncols

        fig, axes = plt.subplots(
            2 * nrows,
            ncols,
            figsize=(4 * ncols, 4 * nrows),
            gridspec_kw={"height_ratios": [3, 1] * nrows},
        )

        if nrows == 1 and ncols > 1:
            axes = axes.reshape(2, ncols)
        elif ncols == 1:
            axes = axes.reshape(2 * nrows, 1)
        elif axes.ndim == 1:
            axes = axes.reshape(2 * nrows, ncols)

        for k, i in enumerate(indices):
            row = k // ncols
            col = k % ncols
            ax = axes[2 * row, col]
            ax_ratio = axes[2 * row + 1, col]

            sim_res = X_hlf_sim_resampled[:, i]
            sim_org = X_hlf_sim[:, i]
            truth = X_hlf_truth[:, i]

            r = feature_range(i)
            if r is None:
                stacked = np.concatenate([sim_res, sim_org, truth])
                stacked = stacked[np.isfinite(stacked)]
                if stacked.size == 0:
                    bins = np.linspace(0.0, 1.0, 51)
                else:
                    lo, hi = np.percentile(stacked, [0.5, 99.5])
                    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                        lo, hi = float(np.min(stacked)), float(np.max(stacked))
                        if lo == hi:
                            hi = lo + 1.0
                    bins = np.linspace(lo, hi, 51)
            else:
                bins = np.linspace(r[0], r[1], 51)

            hist_truth, edges = np.histogram(truth, bins=bins, density=True)
            hist_sim_org, _ = np.histogram(sim_org, bins=edges, density=True)
            hist_sim_res, _ = np.histogram(sim_res, bins=edges, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])

            ax.stairs(hist_sim_res, edges, label="Resampled Sim", linewidth=1.2)
            ax.stairs(hist_sim_org, edges, label="Original Sim", linestyle="dashed", linewidth=1.2)
            ax.hist(truth, bins=edges, alpha=0.35, label="Truth", density=True)

            ratio_res = np.divide(hist_sim_res, hist_truth, out=np.full_like(hist_sim_res, np.nan), where=hist_truth > 0)
            ratio_org = np.divide(hist_sim_org, hist_truth, out=np.full_like(hist_sim_org, np.nan), where=hist_truth > 0)

            ax_ratio.plot(centers, ratio_res, label="Resampled/Truth", linewidth=1.2)
            ax_ratio.plot(centers, ratio_org, label="Original/Truth", linestyle="dashed", linewidth=1.2)
            ax_ratio.axhline(1.0, color="black", linestyle=":", linewidth=1)
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.set_ylabel("Sim/Truth", fontsize=8)

            if row == nrows - 1:
                ax_ratio.set_xlabel(hlf_name_map.get(i, f"Feature {i}"), fontsize=9)

            ax.set_title(hlf_name_map.get(i, f"Feature {i}"), fontsize=10)
            if k == 0:
                ax.legend(fontsize=8, loc="upper right")
                ax_ratio.legend(fontsize=7, loc="upper right")

        for j in range(ng, nrows * ncols):
            row = j // ncols
            col = j % ncols
            if axes.ndim == 1:
                axes[2 * row].axis("off")
                axes[2 * row + 1].axis("off")
            else:
                axes[2 * row, col].axis("off")
                axes[2 * row + 1, col].axis("off")

        plt.tight_layout()
        fname = Path(output_dir) / f"hlf_comparison_{group_label}.pdf"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")

    for grp_label, indices in groups.items():
        plot_group(indices, grp_label)


def plot_latent_covariance_heatmaps(X_truth, X_sim_pre, X_sim_post, output_dir, vmax_abs=None, cmap="RdBu_r"):
    """Plot three-panel heatmap of empirical covariance minus identity for latent features."""
    X_truth = np.asarray(X_truth, dtype=np.float64)
    X_sim_pre = np.asarray(X_sim_pre, dtype=np.float64)
    X_sim_post = np.asarray(X_sim_post, dtype=np.float64)

    def emp_cov_minus_I(X):
        if X.size == 0:
            return np.zeros((0, 0), dtype=float)
        Xc = X - np.mean(X, axis=0, keepdims=True)
        cov = np.cov(Xc, rowvar=False, bias=False)
        d = cov.shape[0]
        return cov - np.eye(d, dtype=cov.dtype)

    cov_t = emp_cov_minus_I(X_truth)
    cov_pre = emp_cov_minus_I(X_sim_pre)
    cov_post = emp_cov_minus_I(X_sim_post)

    def max_abs(mat):
        if mat.size == 0:
            return 0.0
        return float(np.nanmax(np.abs(mat)))

    if vmax_abs is None:
        vmax_abs = max(max_abs(cov_t), max_abs(cov_pre), max_abs(cov_post))
        vmax_abs = max(vmax_abs, 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mats = [(cov_t, "Truth (cov - I)"), (cov_pre, "Sim Before (cov - I)"), (cov_post, "Sim After (cov - I)")]
    im = None

    for ax, (mat, title) in zip(axes, mats):
        if mat.size == 0:
            ax.text(0.5, 0.5, "Empty", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            continue

        im = ax.imshow(mat, cmap=cmap, vmin=-vmax_abs, vmax=vmax_abs, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("Covariance minus Identity")

    plt.tight_layout()
    path = Path(output_dir) / "latent_covariance_heatmaps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
