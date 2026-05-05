"""
Resample simulated latent-feature datasets to match truth distribution using a trained latent MLP.

This script mirrors `resample_classifier.py` but operates on latent features.
It loads an `MLPLatentClassifier`, evaluates truth and simulated latent datasets,
calibrates predictions with isotonic regression, resamples simulated events,
and saves resampled latent features and incident energies. Diagnostic plots
include prediction histograms, ROC, calibration curves and latent covariance
heatmaps (cov - I) for truth, sim before, and sim after resampling.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
import h5py
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve

from module.lightning import MLPLatentClassifier
from data.datamodule import SimpleMLPLatentDataModule
from plotting_utils import (
    plot_calibration_curves,
    plot_predictions_histogram,
    plot_roc_curves,
    plot_latent_covariance_heatmaps,
)

N_MAX_COV = 200000
N_DIM_COV = 100

def load_config_and_paths(config_file, truth_path=None, sim_path=None):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if truth_path:
        config["data"]["init_args"]["truth_data_path"] = truth_path
    if sim_path:
        config["data"]["init_args"]["gen_data_path"] = sim_path

    return config


def load_model_and_data(config, checkpoint_path, device="cuda"):
    model = MLPLatentClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    datamodule = SimpleMLPLatentDataModule(**config["data"]["init_args"])
    datamodule.prepare_data()
    datamodule.setup("test")

    truth_dataloader = datamodule.test_dataloader()

    # Re-initialize with sim data for second pass
    config_sim = config.copy()
    config_sim["data"]["init_args"]["test_data_path"] = config["data"]["init_args"]["gen_data_path"]
    datamodule_sim = SimpleMLPLatentDataModule(**config_sim["data"]["init_args"])
    datamodule_sim.prepare_data()
    datamodule_sim.setup("test")
    sim_dataloader = datamodule_sim.test_dataloader()

    return model, truth_dataloader, sim_dataloader


def collect_predictions(model, truth_dataloader, sim_dataloader, device="cuda"):
    sim_preds = []
    truth_preds = []
    e_sim = []
    e_truth = []
    X_latent_sim = []
    X_latent_truth = []

    with torch.no_grad():
        for truth_batch in tqdm(truth_dataloader, desc="Evaluating truth data"):
            X_proc, cond_proc, y = truth_batch
            X, y_tensor = model.get_input_from_batch(truth_batch)
            y_hat = torch.sigmoid(model(X.to(device)))

            truth_preds.append(y_hat.cpu())
            e_truth.append(cond_proc.cpu())
            X_latent_truth.append(X_proc.cpu().numpy())

        for sim_batch in tqdm(sim_dataloader, desc="Evaluating sim data"):
            X_proc, cond_proc, y = sim_batch
            X, y_tensor = model.get_input_from_batch(sim_batch)
            y_hat = torch.sigmoid(model(X.to(device)))

            sim_preds.append(y_hat.cpu())
            e_sim.append(cond_proc.cpu())
            X_latent_sim.append(X_proc.cpu().numpy())

    yhat_sim = torch.cat(sim_preds, dim=0).numpy().flatten()
    yhat_truth = torch.cat(truth_preds, dim=0).numpy().flatten()
    e_sim = torch.cat(e_sim, dim=0).numpy().flatten()
    e_truth = torch.cat(e_truth, dim=0).numpy().flatten()
    X_latent_sim = np.concatenate(X_latent_sim, axis=0)
    X_latent_truth = np.concatenate(X_latent_truth, axis=0)

    y_sim = np.zeros_like(yhat_sim)
    y_truth = np.ones_like(yhat_truth)

    return {
        "yhat_sim": yhat_sim,
        "yhat_truth": yhat_truth,
        "y_sim": y_sim,
        "y_truth": y_truth,
        "e_sim": e_sim,
        "e_truth": e_truth,
        "X_latent_sim": X_latent_sim,
        "X_latent_truth": X_latent_truth,
    }


def calibrate_scores(yhat_sim, yhat_truth, y_sim, y_truth):
    all_scores = np.concatenate([yhat_sim, yhat_truth])
    all_labels = np.concatenate([y_sim, y_truth])

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(all_scores, all_labels)
    calibrated_scores = ir.transform(all_scores)

    n_sim = len(y_sim)
    cal_sim = calibrated_scores[:n_sim]
    cal_truth = calibrated_scores[n_sim:]

    return cal_sim, cal_truth, ir


def compute_metrics(yhat_sim, yhat_truth, y_sim, y_truth, cal_sim, cal_truth):
    all_scores_raw = np.concatenate([yhat_sim, yhat_truth])
    all_labels = np.concatenate([y_sim, y_truth])
    all_scores_cal = np.concatenate([cal_sim, cal_truth])

    auc_raw = roc_auc_score(all_labels, all_scores_raw)
    auc_cal = roc_auc_score(all_labels, all_scores_cal)

    fpr_raw, tpr_raw, _ = roc_curve(all_labels, all_scores_raw)
    fpr_cal, tpr_cal, _ = roc_curve(all_labels, all_scores_cal)

    return {
        "auc_raw": auc_raw,
        "auc_cal": auc_cal,
        "fpr_raw": fpr_raw,
        "tpr_raw": tpr_raw,
        "fpr_cal": fpr_cal,
        "tpr_cal": tpr_cal,
    }


def compute_resampling_mask(cal_sim, max_score=0.95, seed=None):
    if seed is not None:
        np.random.seed(seed)

    density_ratio = cal_sim / (1 - cal_sim + 1e-8)
    max_density_ratio = max_score / (1 - max_score + 1e-8)
    density_ratio = np.clip(density_ratio, 0, max_density_ratio)
    normalized_density_ratio = density_ratio / np.max(density_ratio)

    mask = np.random.binomial(1, normalized_density_ratio) == 1

    return mask, density_ratio, normalized_density_ratio


def save_resampled_latent(X_latent_sim, e_sim, mask, output_path):
    X_resampled = X_latent_sim[mask]
    e_resampled = e_sim[mask]

    # incident energies stored in MeV
    e_resampled_mev = e_resampled * 1.0e3

    with h5py.File(output_path, "w") as f:
        f.create_dataset("latent_features", data=X_resampled.astype(np.float32))
        f.create_dataset("incident_energies", data=e_resampled_mev.astype(np.float32))

    print(f"Saved resampled latent dataset to {output_path}")
    print(f"  Original sim events: {len(e_sim)}")
    print(f"  Resampled events: {len(e_resampled)}")
    print(f"  Retention rate: {100 * len(e_resampled) / len(e_sim):.1f}%")


def save_metadata(output_path, predictions_data, metrics, mask, seed=None):
    import json

    metadata = {
        "n_original_sim": int(len(predictions_data["e_sim"])),
        "n_resampled": int(np.sum(mask)),
        "retention_rate": float(np.sum(mask) / len(predictions_data["e_sim"])),
        "auc_raw": float(metrics["auc_raw"]),
        "auc_calibrated": float(metrics["auc_cal"]),
        "seed": seed,
    }

    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Resample simulated latent features using a trained MLP latent classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--truth-data", type=str, default=None, help="Path to truth dataset (overrides config)")
    parser.add_argument("--sim-data", type=str, default=None, help="Path to simulated dataset (overrides config)")
    parser.add_argument("--output", type=str, default="resampled_latent.hdf5", help="Path to output HDF5 file")
    parser.add_argument("--plots", type=str, default=None, help="Directory to save diagnostic plots (if not specified, no plots are saved)")
    parser.add_argument("--max-score", type=float, default=0.95, help="Maximum calibrated score for weight clipping (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible resampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    if args.plots:
        Path(args.plots).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Latent-feature Resampling")
    print("=" * 70)

    config = load_config_and_paths(args.config, args.truth_data, args.sim_data)

    print(f"Loading model from {args.checkpoint}...")
    model, truth_dataloader, sim_dataloader = load_model_and_data(config, args.checkpoint, device=args.device)

    print("Collecting predictions on latent datasets...")
    predictions = collect_predictions(model, truth_dataloader, sim_dataloader, device=args.device)

    print("Calibrating predictions using isotonic regression...")
    cal_sim, cal_truth, ir = calibrate_scores(predictions["yhat_sim"], predictions["yhat_truth"], predictions["y_sim"], predictions["y_truth"])

    print("Computing metrics...")
    metrics = compute_metrics(predictions["yhat_sim"], predictions["yhat_truth"], predictions["y_sim"], predictions["y_truth"], cal_sim, cal_truth)
    print(f"  Raw AUC: {metrics['auc_raw']:.4f}")
    print(f"  Calibrated AUC: {metrics['auc_cal']:.4f}")

    print(f"Computing resampling mask (max_score={args.max_score}, seed={args.seed})...")
    mask, density_ratio, normalized_density_ratio = compute_resampling_mask(cal_sim, max_score=args.max_score, seed=args.seed)
    print(f"  Original sim events: {len(mask)}")
    print(f"  Events to keep: {np.sum(mask)} ({100*np.sum(mask)/len(mask):.1f}%)")

    print(f"Saving resampled latent dataset to {args.output}...")
    save_resampled_latent(predictions["X_latent_sim"], predictions["e_sim"], mask, args.output)
    save_metadata(args.output, predictions, metrics, mask, seed=args.seed)

    if args.plots:
        print(f"Generating diagnostic plots in {args.plots}...")
        plot_predictions_histogram(predictions["yhat_sim"], predictions["yhat_truth"], cal_sim, cal_truth, args.plots)
        plot_roc_curves(metrics, args.plots)
        plot_calibration_curves(predictions["yhat_sim"], predictions["yhat_truth"], predictions["y_sim"], predictions["y_truth"], cal_sim, cal_truth, args.plots)

        # Covariance heatmaps (cov - I)
        # randomly select a subsets of dimensions and events for visualization
        random_dims = np.random.choice(predictions["X_latent_sim"].shape[1], size=N_DIM_COV, replace=False).tolist()

        truth_idx = np.random.choice(predictions["X_latent_truth"].shape[0], size=min(N_MAX_COV, predictions["X_latent_truth"].shape[0]), replace=False).tolist()
        sim_idx = np.random.choice(predictions["X_latent_sim"].shape[0], size=min(N_MAX_COV, predictions["X_latent_sim"].shape[0]), replace=False).tolist()
        print(predictions["X_latent_truth"].shape)
        X_truth = predictions["X_latent_truth"][truth_idx][:, random_dims]
        X_sim = predictions["X_latent_sim"][sim_idx][:, random_dims]
        X_resampled = predictions["X_latent_sim"][mask]
        sim_idx_resampled = np.random.choice(X_resampled.shape[0], size=min(N_MAX_COV, X_resampled.shape[0]), replace=False).tolist()
        X_resampled = X_resampled[sim_idx_resampled][:, random_dims]


        print("Printing covariance matrix statistics:")
        plot_latent_covariance_heatmaps(
            X_truth,
            X_sim,
            X_resampled,
            args.plots,
        )

        print(f"Plots saved to {args.plots}")

    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
