"""
Resample simulated showers to match truth distribution using a trained classifier.

This script loads a trained MLP classifier, evaluates it on truth and simulated datasets,
calibrates predictions using isotonic regression, and resamples simulated events based on
the density ratio to match the truth distribution. Diagnostic plots are generated and
the resampled showers are saved to HDF5.
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

from module.lightning import MLPClassifier
from data.datamodule import SimpleMLPDataModule
from data.HighLevelFeatures import HighLevelFeatures
from data.utils import _get_hlf_extractor
from plotting_utils import (
    plot_calibration_curves,
    plot_hlf_comparison,
    plot_predictions_histogram,
    plot_roc_curves,
)

def load_config_and_paths(config_file, truth_path=None, sim_path=None):
    """Load YAML config and optionally override data paths."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    if truth_path:
        config["data"]["init_args"]["truth_data_path"] = truth_path
    if sim_path:
        config["data"]["init_args"]["gen_data_path"] = sim_path
    
    return config


def reconstruct_raw_shower(X_proc, cond_proc):
    """Reconstruct raw shower from preprocessed components.
    
    Inverts the preprocessing done in SimpleMLPDataModule._preprocess_data:
    X_proc is already divided by cond and flattened, cond_proc is in GeV.
    Returns X in original shape (to be reshaped for HighLevelFeatures.CalculateFeatures).
    """
    # Convert cond from GeV back to original unit (MeV)
    cond_mev = cond_proc * 1000.0
    # Multiply X back by condition to recover original shower
    X_raw = X_proc * cond_mev.reshape(-1, 1) if X_proc.ndim == 2 else X_proc * cond_mev
    return X_raw


def build_hlf_name_map(hlf_extractor: HighLevelFeatures) -> dict:
    """Build a mapping from HLF index to LaTeX-style names.
    
    Order matches the extraction order:
    1. Total energy (Etot)
    2-N. Per-layer energies (E_i)
    N+1 onwards: Eta centers, Phi centers, Eta widths, Phi widths, Sparsity
    """
    name_map = {}
    idx = 0
    
    # Total energy
    name_map[idx] = r"$E_{\mathrm{tot}}$"
    idx += 1
    
    # Per-layer energies
    E_layers = hlf_extractor.GetElayers()
    for layer_id in sorted(E_layers.keys()):
        name_map[idx] = f"$E_{{{layer_id}}}$"
        idx += 1
    
    # Eta centers
    EC_etas = hlf_extractor.GetECEtas()
    for layer_id in sorted(EC_etas.keys()):
        name_map[idx] = f"$\\langle \\eta_{{{layer_id}}} \\rangle$"
        idx += 1
    
    # Phi centers
    EC_phis = hlf_extractor.GetECPhis()
    for layer_id in sorted(EC_phis.keys()):
        name_map[idx] = f"$\\langle \\phi_{{{layer_id}}} \\rangle$"
        idx += 1
    
    # Eta widths
    Width_etas = hlf_extractor.GetWidthEtas()
    for layer_id in sorted(Width_etas.keys()):
        name_map[idx] = f"$\\sigma_{{\\eta_{{{layer_id}}}}}$"
        idx += 1
    
    # Phi widths
    Width_phis = hlf_extractor.GetWidthPhis()
    for layer_id in sorted(Width_phis.keys()):
        name_map[idx] = f"$\\sigma_{{\\phi_{{{layer_id}}}}}$"
        idx += 1
    
    # Sparsity
    Sparsity = hlf_extractor.GetSparsity()
    for layer_id in sorted(Sparsity.keys()):
        name_map[idx] = f"$\\lambda_{{{layer_id}}}$"
        idx += 1
    
    return name_map


def compute_all_hlf(X_raw, cond_raw, xml_path, particle):
    """Compute all HLF (including sparsity) from raw shower data.
    
    Args:
        X_raw: Raw shower data (N_events, N_voxels)
        cond_raw: Incident energies (N_events,) in MeV
        xml_path: Path to XML geometry file
        particle: Particle name
    
    Returns:
        Tuple of (hlf_array, hlf_extractor) where hlf_array has shape (N_events, N_hlf)
    """
    hlf_extractor = _get_hlf_extractor(particle, xml_path)
    setattr(hlf_extractor, "Einc", cond_raw)
    hlf_extractor.CalculateFeatures(X_raw)
    
    hlf_features = [
        hlf_extractor.GetEtot(),
        *hlf_extractor.GetElayers().values(),
        *hlf_extractor.GetECEtas().values(),
        *hlf_extractor.GetECPhis().values(),
        *hlf_extractor.GetWidthEtas().values(),
        *hlf_extractor.GetWidthPhis().values(),
        *hlf_extractor.GetSparsity().values(),  # Include sparsity
    ]
    
    hlf_array = np.stack(hlf_features, axis=1, dtype=np.float32)
    return hlf_array, hlf_extractor


def load_model_and_data(config, checkpoint_path, device="cuda"):
    """Load model checkpoint and initialize dataloaders."""
    model = MLPClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    datamodule = SimpleMLPDataModule(**config["data"]["init_args"])
    datamodule.prepare_data()
    datamodule.setup("test")
    
    truth_dataloader = datamodule.test_dataloader()
    
    # Re-initialize with sim data for second pass
    config_sim = config.copy()
    config_sim["data"]["init_args"]["test_data_path"] = config["data"]["init_args"]["gen_data_path"]
    datamodule_sim = SimpleMLPDataModule(**config_sim["data"]["init_args"])
    datamodule_sim.prepare_data()
    datamodule_sim.setup("test")
    sim_dataloader = datamodule_sim.test_dataloader()
    
    return model, truth_dataloader, sim_dataloader


def collect_predictions(model, truth_dataloader, sim_dataloader, xml_path, particle, device="cuda"):
    """Forward pass on truth and sim data; collect predictions and compute HLF for plotting.
    
    Uses stored HLF for model inference, but also computes HLF from raw showers for plotting.
    """
    sim_preds = []
    truth_preds = []
    e_sim = []
    e_truth = []
    X_hlf_stored_truth = []  # Stored HLF for model (from batch)
    X_hlf_stored_sim = []  # Stored HLF for model (from batch)
    X_hlf_computed_truth = []  # Computed HLF for plotting only
    X_hlf_computed_sim = []  # Computed HLF for plotting only
    X_raw_sim = []
    
    with torch.no_grad():
        for truth_batch in tqdm(truth_dataloader, desc="Evaluating truth data"):
            X_proc, cond_proc, X_hlf, _ = truth_batch
            # Reconstruct raw shower for computing HLF
            X_proc_np = X_proc.numpy()
            cond_proc_np = cond_proc.numpy().flatten()
            X_raw_truth_batch = reconstruct_raw_shower(X_proc_np, cond_proc_np)
            
            # Compute HLF from raw shower (for plotting only)
            cond_mev = cond_proc_np * 1000.0  # Convert back to MeV
            hlf_computed_batch, _ = compute_all_hlf(X_raw_truth_batch, cond_mev, xml_path, particle)
            X_hlf_computed_truth.append(hlf_computed_batch)
            
            # Model forward pass uses STORED HLF from batch
            X, y = model.get_input_from_batch(truth_batch)
            y_hat = torch.sigmoid(model(X.to(device)))
            e_truth.append(cond_proc.cpu())
            truth_preds.append(y_hat.cpu())
            X_hlf_stored_truth.append(X_hlf.cpu())
        
        for sim_batch in tqdm(sim_dataloader, desc="Evaluating sim data"):
            X_proc, cond_proc, X_hlf, _ = sim_batch
            # Reconstruct raw shower for computing HLF
            X_proc_np = X_proc.numpy()
            cond_proc_np = cond_proc.numpy().flatten()
            X_raw_sim_batch = reconstruct_raw_shower(X_proc_np, cond_proc_np)
            X_raw_sim.append(X_raw_sim_batch)
            
            # Compute HLF from raw shower (for plotting only)
            cond_mev = cond_proc_np * 1000.0  # Convert back to MeV
            hlf_computed_batch, _ = compute_all_hlf(X_raw_sim_batch, cond_mev, xml_path, particle)
            X_hlf_computed_sim.append(hlf_computed_batch)
            
            # Model forward pass uses STORED HLF from batch
            X, y = model.get_input_from_batch(sim_batch)
            y_hat = torch.sigmoid(model(X.to(device)))
            sim_preds.append(y_hat.cpu())
            e_sim.append(cond_proc.cpu())
            X_hlf_stored_sim.append(X_hlf.cpu())
    
    yhat_sim = torch.cat(sim_preds, dim=0).numpy().flatten()
    yhat_truth = torch.cat(truth_preds, dim=0).numpy().flatten()
    e_sim = torch.cat(e_sim, dim=0).numpy().flatten()
    e_truth = torch.cat(e_truth, dim=0).numpy().flatten()
    X_hlf_computed_sim = np.concatenate(X_hlf_computed_sim, axis=0)
    X_hlf_computed_truth = np.concatenate(X_hlf_computed_truth, axis=0)
    X_raw_sim = np.concatenate(X_raw_sim, axis=0)
    
    y_sim = np.zeros_like(yhat_sim)
    y_truth = np.ones_like(yhat_truth)
    
    return {
        "yhat_sim": yhat_sim,
        "yhat_truth": yhat_truth,
        "y_sim": y_sim,
        "y_truth": y_truth,
        "e_sim": e_sim,
        "e_truth": e_truth,
        "X_hlf_sim": X_hlf_computed_sim,  # Computed HLF for plotting
        "X_hlf_truth": X_hlf_computed_truth,  # Computed HLF for plotting
        "X_raw_sim": X_raw_sim,
    }


def calibrate_scores(yhat_sim, yhat_truth, y_sim, y_truth):
    """Fit isotonic regression calibration on combined predictions."""
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
    """Compute AUC and other metrics."""
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
    """Compute resampling mask using density ratio weights."""
    if seed is not None:
        np.random.seed(seed)
    
    density_ratio = cal_sim / (1 - cal_sim + 1e-8)
    max_density_ratio = max_score / (1 - max_score + 1e-8)
    density_ratio = np.clip(density_ratio, 0, max_density_ratio)
    normalized_density_ratio = density_ratio / np.max(density_ratio)
    
    mask = np.random.binomial(1, normalized_density_ratio) == 1
    
    return mask, density_ratio, normalized_density_ratio


def save_resampled_showers(X_raw_sim, e_sim, mask, output_path):
    """Save resampled showers to HDF5 with energy scaling."""
    X_resampled = X_raw_sim[mask]
    e_resampled = e_sim[mask]
    
    # Scale shower by energy: multiply by cond (which is in GeV from preprocessing)
    e_resampled_mev = e_resampled * 1.0e3
    X_resampled_scaled = X_resampled * e_resampled_mev.reshape(-1, 1)
    
    with h5py.File(output_path, "w") as f:
        f.create_dataset("showers", data=X_resampled_scaled, compression="lzf")
        f.create_dataset("incident_energies", data=np.rint(e_resampled_mev).astype(np.float32), compression="lzf")
    
    print(f"Saved resampled showers to {output_path}")
    print(f"  Original sim events: {len(e_sim)}")
    print(f"  Resampled events: {len(e_resampled)}")
    print(f"  Retention rate: {100 * len(e_resampled) / len(e_sim):.1f}%")


def save_metadata(output_path, predictions_data, metrics, mask, seed=None):
    """Save metadata and metrics to JSON alongside HDF5."""
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
    parser = argparse.ArgumentParser(
        description="Resample simulated showers using a trained MLP classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config/mlp_caloinn300.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--truth-data",
        type=str,
        default=None,
        help="Path to truth dataset (overrides config)"
    )
    parser.add_argument(
        "--sim-data",
        type=str,
        default=None,
        help="Path to simulated dataset (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resampled_showers.hdf5",
        help="Path to output HDF5 file with resampled showers"
    )
    parser.add_argument(
        "--plots",
        type=str,
        default=None,
        help="Directory to save diagnostic plots (if not specified, no plots are saved)"
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="Maximum calibrated score for weight clipping (0-1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible resampling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    if args.plots:
        Path(args.plots).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Classifier-based Shower Resampling")
    print("=" * 70)
    
    # Load config and data
    print(f"Loading config from {args.config}...")
    config = load_config_and_paths(args.config, args.truth_data, args.sim_data)
    
    print(f"Loading model from {args.checkpoint}...")
    model, truth_dataloader, sim_dataloader = load_model_and_data(
        config, args.checkpoint, device=args.device
    )
    
    # Collect predictions and compute HLF
    print("Collecting predictions and computing HLFs...")
    particle = config["data"]["init_args"]["particle"]
    xml_path = config["data"]["init_args"]["xml_path"]
    predictions = collect_predictions(model, truth_dataloader, sim_dataloader, xml_path, particle, device=args.device)
    
    # Calibrate
    print("Calibrating predictions using isotonic regression...")
    cal_sim, cal_truth, ir = calibrate_scores(
        predictions["yhat_sim"], predictions["yhat_truth"],
        predictions["y_sim"], predictions["y_truth"]
    )
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(
        predictions["yhat_sim"], predictions["yhat_truth"],
        predictions["y_sim"], predictions["y_truth"],
        cal_sim, cal_truth
    )
    print(f"  Raw AUC: {metrics['auc_raw']:.4f}")
    print(f"  Calibrated AUC: {metrics['auc_cal']:.4f}")
    
    # Compute resampling mask
    print(f"Computing resampling mask (max_score={args.max_score}, seed={args.seed})...")
    mask, density_ratio, normalized_density_ratio = compute_resampling_mask(
        cal_sim, max_score=args.max_score, seed=args.seed
    )
    print(f"  Original sim events: {len(mask)}")
    print(f"  Events to keep: {np.sum(mask)} ({100*np.sum(mask)/len(mask):.1f}%)")
    
    # Save resampled showers
    print(f"Saving resampled showers to {args.output}...")
    save_resampled_showers(predictions["X_raw_sim"], predictions["e_sim"], mask, args.output)
    
    # Save metadata
    save_metadata(args.output, predictions, metrics, mask, seed=args.seed)
    
    # Generate plots if requested
    if args.plots:
        print(f"Generating diagnostic plots in {args.plots}...")
        plot_predictions_histogram(
            predictions["yhat_sim"], predictions["yhat_truth"],
            cal_sim, cal_truth,
            args.plots
        )
        plot_roc_curves(metrics, args.plots)
        plot_calibration_curves(
            predictions["yhat_sim"], predictions["yhat_truth"],
            predictions["y_sim"], predictions["y_truth"],
            cal_sim, cal_truth,
            args.plots
        )
        # Build HLF name map for plotting
        _, hlf_extractor = compute_all_hlf(
            predictions["X_raw_sim"][:1], 
            predictions["e_sim"][:1] * 1000.0,
            xml_path, particle
        )
        hlf_name_map = build_hlf_name_map(hlf_extractor)
        
        plot_hlf_comparison(
            predictions["X_hlf_sim"], predictions["X_hlf_truth"],
            mask,
            args.plots,
            hlf_name_map=hlf_name_map
        )
        print(f"Plots saved to {args.plots}")
    
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
