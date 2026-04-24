import sys
sys.path.append("..")  # Add parent directory to sys.path to import data.utils
from data.utils import prepare_gen_latent
from tqdm import tqdm
import numpy as np

OUTPUT_CHUNK = 500_000

if __name__ == "__main__":
    import argparse
    import os
    import h5py

    parser = argparse.ArgumentParser(description="Prepare latent vectors for generator")
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Path to input HDF5 file containing conditions")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to output HDF5 file to save latent vectors")
    parser.add_argument("-f", "--latent-field", type=str, help="Dimensionality of the latent vectors")
    args = parser.parse_args()

    # Read conditions from input HDF5 file
    with h5py.File(args.input_path, 'r') as f:
        cond = f['incident_energies'][:].astype(np.float32)
        latent_dim = f[args.latent_field].shape[1]

    # Save latent vectors to output HDF5 file in chunks to avoid memory issues
    for start_idx in tqdm(range(0, len(cond), OUTPUT_CHUNK), desc="Preparing latent vectors"):
        end_idx = min(start_idx + OUTPUT_CHUNK, len(cond))
        chunk_cond = cond[start_idx:end_idx]
        gen_latent = prepare_gen_latent(chunk_cond, latent_dim)

        with h5py.File(args.output_path, 'a') as f_out:
            if args.latent_field not in f_out:
                maxshape = (None, latent_dim)
                f_out.create_dataset(args.latent_field, data=gen_latent, maxshape=maxshape, chunks=True)
                f_out.create_dataset("incident_energies", data=chunk_cond, maxshape=(None,1), chunks=True)
            else:
                f_out[args.latent_field].resize((f_out[args.latent_field].shape[0] + gen_latent.shape[0]), axis=0)
                f_out[args.latent_field][-gen_latent.shape[0]:] = gen_latent
                f_out["incident_energies"].resize((f_out["incident_energies"].shape[0] + chunk_cond.shape[0]), axis=0)
                f_out["incident_energies"][-chunk_cond.shape[0]:] = chunk_cond

    print(f"Saved generated latent vectors to {args.output_path}")