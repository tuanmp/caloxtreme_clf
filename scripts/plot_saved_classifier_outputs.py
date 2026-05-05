"""Plot HLF comparisons from saved classifier-resampling outputs.

This script loads truth, sim, and sim_resampled HDF5 files through
`SimpleMLPDataModule`, using each file as a `test_data_path` so the processed
`X_proc`, `cond_proc`, and `X_hlf` fields are created the same way as in the
main resampling workflow.
"""

import copy
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import _get_hlf_extractor
from data.datamodule import SimpleMLPDataModule
from plotting_utils import plot_hlf_comparison
from scripts.resample_classifier import (
    reconstruct_raw_shower,
    compute_all_hlf,
)


def load_config(config_file):
    """Load YAML config."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def build_hlf_name_map(hlf_extractor):
    """Build a mapping from HLF index to LaTeX-style names."""
    name_map = {}
    idx = 0

    name_map[idx] = r"$E_{\mathrm{tot}}$"
    idx += 1

    for layer_id in sorted(hlf_extractor.GetElayers().keys()):
        name_map[idx] = f"$E_{{{layer_id}}}$"
        idx += 1

    for layer_id in sorted(hlf_extractor.GetECEtas().keys()):
        name_map[idx] = rf"$\langle \eta_{{{layer_id}}} \rangle$"
        idx += 1

    for layer_id in sorted(hlf_extractor.GetECPhis().keys()):
        name_map[idx] = rf"$\langle \phi_{{{layer_id}}} \rangle$"
        idx += 1

    for layer_id in sorted(hlf_extractor.GetWidthEtas().keys()):
        name_map[idx] = rf"$\sigma_{{\eta_{{{layer_id}}}}}$"
        idx += 1

    for layer_id in sorted(hlf_extractor.GetWidthPhis().keys()):
        name_map[idx] = rf"$\sigma_{{\phi_{{{layer_id}}}}}$"
        idx += 1

    for layer_id in sorted(hlf_extractor.GetSparsity().keys()):
        name_map[idx] = rf"$\lambda_{{{layer_id}}}$"
        idx += 1

    return name_map


def load_file_via_datamodule(file_path: str, config: dict):
    """Load processed batches through SimpleMLPDataModule using a single file as test data."""
    init_args = copy.deepcopy(config["data"]["init_args"])
    init_args["gen_data_path"] = file_path
    init_args["truth_data_path"] = file_path
    init_args["test_data_path"] = file_path

    datamodule = SimpleMLPDataModule(**init_args)
    datamodule.prepare_data()
    datamodule.setup("test")

    x_hlf_batches = []
    x_proc_batches = []
    cond_proc_batches = []

    for batch in tqdm(datamodule.test_dataloader(), desc=f"Loading {Path(file_path).name}"):
        X_proc, cond_proc, X_hlf, _ = batch
        x_hlf_batches.append(X_hlf.cpu().numpy())
        x_proc_batches.append(X_proc.cpu().numpy())
        cond_proc_batches.append(cond_proc.cpu().numpy())

    X_hlf = np.concatenate(x_hlf_batches, axis=0) if x_hlf_batches else np.empty((0, 0), dtype=np.float32)
    X_proc = np.concatenate(x_proc_batches, axis=0) if x_proc_batches else np.empty((0, 0), dtype=np.float32)
    cond_proc = np.concatenate(cond_proc_batches, axis=0) if cond_proc_batches else np.empty((0,), dtype=np.float32)

    return {
        "X_hlf": X_hlf.astype(np.float32, copy=False),
        "X_proc": X_proc.astype(np.float32, copy=False),
        "cond_proc": cond_proc.astype(np.float32, copy=False),
    }


def plot_saved_classifier_outputs(config, truth_file, sim_file, sim_resampled_file, output_dir):
    """Make the HLF comparison plots from saved classifier-resampling outputs."""
    truth_data = load_file_via_datamodule(truth_file, config)
    sim_data = load_file_via_datamodule(sim_file, config)
    sim_resampled_data = load_file_via_datamodule(sim_resampled_file, config)

    data_init_args = config["data"]["init_args"]
    xml_path = data_init_args["xml_path"]
    particle = data_init_args["particle"]

    truth_X_raw = reconstruct_raw_shower(truth_data["X_proc"], truth_data["cond_proc"])
    sim_X_raw = reconstruct_raw_shower(sim_data["X_proc"], sim_data["cond_proc"])
    sim_resampled_X_raw = reconstruct_raw_shower(sim_resampled_data["X_proc"], sim_resampled_data["cond_proc"])

    cond_truth_mev = truth_data["cond_proc"] * 1000.0
    cond_sim_mev = sim_data["cond_proc"] * 1000.0
    cond_sim_resampled_mev = sim_resampled_data["cond_proc"] * 1000.0

    X_hlf_truth, hlf_extractor = compute_all_hlf(truth_X_raw, cond_truth_mev, xml_path, particle)
    X_hlf_sim, _ = compute_all_hlf(sim_X_raw, cond_sim_mev, xml_path, particle)
    X_hlf_sim_resampled, _ = compute_all_hlf(sim_resampled_X_raw, cond_sim_resampled_mev, xml_path, particle)

    hlf_name_map = build_hlf_name_map(hlf_extractor)
    plot_hlf_comparison(
        X_hlf_sim,
        X_hlf_truth,
        mask=None,
        output_dir=output_dir,
        hlf_name_map=hlf_name_map,
        X_hlf_sim_resampled=X_hlf_sim_resampled,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate HLF comparison plots from saved classifier-resampling HDF5 files."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config used for the data module")
    parser.add_argument("--truth", type=str, required=True, help="Path to the truth HDF5 file")
    parser.add_argument("--sim", type=str, required=True, help="Path to the simulated HDF5 file")
    parser.add_argument("--sim-resampled", type=str, required=True, help="Path to the resampled simulated HDF5 file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    rank_zero_info("Loaded config and starting plot generation.")
    plot_saved_classifier_outputs(config, args.truth, args.sim, args.sim_resampled, args.output_dir)


if __name__ == "__main__":
    main()
