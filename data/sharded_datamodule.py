"""Large HDF5 raw-file-first data pipeline with optional caching.

This module implements an independent data pipeline that reads directly from raw HDF5 files
without writing derived split files by default. Key design:

1. **Raw Input**: Reads showers and incident_energies from raw HDF5 files.
2. **On-the-fly Preprocessing**: Computes X_proc (flattened showers), cond_proc (scaled energy),
   and X_hlf (high-level features) per batch without disk writes.
3. **Splitting**: Performs deterministic train/val/test splits with energy filtering.
4. **Optional Cache**: Supports memmap or zarr for fast subsequent runs (explicit opt-in).
5. **No Artifacts**: By default, leaves no derived files—preprocessing happens in workers.

Data flow:
  Raw HDF5 → split indices → worker reads batch → preprocess → return tensors → optionally cache
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from lightning.pytorch.core import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.utils import get_high_level_features, scale_shower

_H5_RDCC = {"rdcc_nbytes": 64 * 1024 * 1024, "rdcc_nslots": 4093}


def _read_rows(dataset, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return dataset[:0]

    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.size <= 1:
        return np.asarray(dataset[ordered])

    # Fast path: DataLoader often provides monotonic indices already.
    if np.all(ordered[1:] >= ordered[:-1]):
        return np.asarray(dataset[ordered])

    order = np.argsort(ordered)
    sorted_indices = ordered[order]
    data = np.asarray(dataset[sorted_indices])

    inverse_order = np.argsort(order)
    data = data[inverse_order]

    return data


class _RawHDF5Source:
    """Lazy reader for raw HDF5 files with efficient row selection.
    
    Opens an HDF5 file once per worker and caches the handle. Supports
    efficient reads of arbitrary row indices by sorting and re-ordering.
    """
    def __init__(self, file_path: str, fields: tuple[str, str] = ("showers", "incident_energies")):
        self.file_path = file_path
        self.fields = fields
        self._file = None

        with h5py.File(self.file_path, "r") as handle:
            self._length = int(handle[self.fields[0]].shape[0])

    def __len__(self) -> int:
        return self._length

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r", **_H5_RDCC)
        return self._file

    def read_rows(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        handle = self._get_file()
        X = _read_rows(handle[self.fields[0]], indices)
        cond = _read_rows(handle[self.fields[1]], indices)
        return X, cond

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        file_handle = getattr(self, "_file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass


class _RawProcessedDataset(Dataset):
    """Processes raw data on-the-fly to produce (X_proc, cond_proc, X_hlf, y).
    
    Given a source and a subset of indices, this dataset:
    1. Reads raw showers and energies from source.
    2. Applies voxel_energy_cutoff if specified.
    3. Computes high-level features (X_hlf) using XML binning.
    4. Normalizes showers by energy and converts energy to GeV.
    5. Flattens normalized showers to X_proc.
    6. Assigns labels (gen=0, truth/test=1).
    
    All computation happens per-batch when samples are requested, with no disk writes.
    """
    def __init__(
        self,
        source: _RawHDF5Source,
        indices: np.ndarray,
        label: float,
        xml_path: str,
        particle: str,
        voxel_energy_cutoff: Optional[float] = None,
        log_transform: bool = False,
    ):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label = float(label)
        self.xml_path = xml_path
        self.particle = particle
        self.voxel_energy_cutoff = voxel_energy_cutoff
        self.log_transform = log_transform

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _preprocess_batch(self, X: np.ndarray, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply all preprocessing steps: voxel cut, HLF, normalization, flattening.
        
        Args:
            X: raw showers (batch, features).
            cond: incident energies (batch,) or (batch, 1).
        
        Returns:
            X_proc (flattened normalized showers), cond_proc (energy in GeV), X_hlf (high-level features).
        """
        # Apply voxel energy threshold to suppress noise.
        if self.voxel_energy_cutoff is not None:
            X = np.where(X > self.voxel_energy_cutoff, X, 0.0)

        # Compute high-level features from raw showers and energy.
        X_hlf = get_high_level_features(X, cond, self.xml_path, self.particle)
        
        # Normalize showers by energy and convert energy from MeV to GeV.
        X, cond = scale_shower(X, cond)

        # Optionally apply log transform to handle wide dynamic range (not enabled by default).
        if self.log_transform:
            X = np.log1p(X)
        
        # Flatten showers for MLP input.
        X = X.reshape(X.shape[0], -1)
        return X.astype(np.float32, copy=False), cond.astype(np.float32, copy=False), X_hlf

    def get_numpy_batch(self, local_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source_indices = self.indices[np.asarray(local_indices, dtype=np.int64)]
        X, cond = self.source.read_rows(source_indices)
        X_proc, cond_proc, X_hlf = self._preprocess_batch(X.astype(np.float32), cond.astype(np.float32))
        y = np.full((X_proc.shape[0],), self.label, dtype=np.float32)
        return X_proc, cond_proc, X_hlf.astype(np.float32, copy=False), y

    def __getitem__(self, index: int):
        X_proc, cond_proc, X_hlf, y = self.get_numpy_batch(np.asarray([index]))
        return (
            torch.from_numpy(X_proc[0]),
            torch.from_numpy(cond_proc[0]),
            torch.from_numpy(X_hlf[0]),
            torch.from_numpy(y[0:1]).squeeze(0),
        )

    def __getitems__(self, indices):
        indices_arr = np.asarray(indices, dtype=np.int64)
        X_proc, cond_proc, X_hlf, y = self.get_numpy_batch(indices_arr)

        X_tensor = torch.from_numpy(X_proc)
        cond_tensor = torch.from_numpy(cond_proc)
        hlf_tensor = torch.from_numpy(X_hlf)
        y_tensor = torch.from_numpy(y)

        return [
            (X_tensor[i], cond_tensor[i], hlf_tensor[i], y_tensor[i])
            for i in range(len(indices_arr))
        ]


class _ConcatDataset(Dataset):
    """Concatenate multiple datasets (gen, truth) into one contiguous view.
    
    Maps global indices to (dataset_idx, local_idx) so that DataLoader sees
    all data as a single dataset and can shuffle across both sources.
    """
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self._lengths = [len(ds) for ds in self.datasets]
        total = 0
        self._cumulative_lengths = []
        for length in self._lengths:
            total += length
            self._cumulative_lengths.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _locate_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("index out of range")
        dataset_idx = bisect_right(self._cumulative_lengths, index)
        previous_total = 0 if dataset_idx == 0 else self._cumulative_lengths[dataset_idx - 1]
        local_index = index - previous_total
        return dataset_idx, local_index

    def __getitem__(self, index: int):
        dataset_idx, local_index = self._locate_index(int(index))
        return self.datasets[dataset_idx][local_index]

    def __getitems__(self, indices):
        requested_indices = np.asarray(indices, dtype=np.int64)
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_position, global_index in enumerate(requested_indices.tolist()):
            dataset_idx, local_index = self._locate_index(int(global_index))
            grouped.setdefault(dataset_idx, []).append((output_position, local_index))

        output = [None] * len(requested_indices)
        for dataset_idx, items in grouped.items():
            positions = np.asarray([pos for pos, _ in items], dtype=np.int64)
            local_indices = np.asarray([idx for _, idx in items], dtype=np.int64)
            rows = self.datasets[dataset_idx].__getitems__(local_indices)
            for row_idx, output_position in enumerate(positions):
                output[int(output_position)] = rows[row_idx]

        return output

    def get_numpy_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        requested_indices = np.asarray(indices, dtype=np.int64)
        if requested_indices.size == 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_position, global_index in enumerate(requested_indices.tolist()):
            dataset_idx, local_index = self._locate_index(int(global_index))
            grouped.setdefault(dataset_idx, []).append((output_position, local_index))

        X_out = cond_out = hlf_out = y_out = None
        for dataset_idx, items in grouped.items():
            positions = np.asarray([pos for pos, _ in items], dtype=np.int64)
            local_indices = np.asarray([idx for _, idx in items], dtype=np.int64)
            X_proc, cond_proc, X_hlf, y = self.datasets[dataset_idx].get_numpy_batch(local_indices)

            if X_out is None:
                X_out = np.empty((requested_indices.size, X_proc.shape[1]), dtype=np.float32)
                cond_out = np.empty((requested_indices.size, cond_proc.shape[1]), dtype=np.float32)
                hlf_out = np.empty((requested_indices.size, X_hlf.shape[1]), dtype=np.float32)
                y_out = np.empty((requested_indices.size,), dtype=np.float32)

            X_out[positions] = X_proc
            cond_out[positions] = cond_proc
            hlf_out[positions] = X_hlf
            y_out[positions] = y

        assert X_out is not None and cond_out is not None and hlf_out is not None and y_out is not None
        return X_out, cond_out, hlf_out, y_out


class _MemmapDataset(Dataset):
    """Loads cached data from memmap files.
    
    Reads metadata JSON that describes where each field (X_proc, cond_proc, X_hlf, y)
    is stored as a memmap file, then memory-maps them for fast random access.
    """
    def __init__(self, meta_path: Path):
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        self.fields = meta["fields"]
        self.arrays = {}
        for name, info in self.fields.items():
            file_path = meta_path.parent / info["file"]
            shape = tuple(info["shape"])
            dtype = np.dtype(info["dtype"])
            self.arrays[name] = np.memmap(file_path, mode="r", dtype=dtype, shape=shape)

        self._length = int(self.fields["y"]["shape"][0])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.arrays["X_proc"][index]),
            torch.from_numpy(self.arrays["cond_proc"][index]),
            torch.from_numpy(self.arrays["X_hlf"][index]),
            torch.from_numpy(self.arrays["y"][index]),
        )

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        X = torch.from_numpy(self.arrays["X_proc"][idx])
        cond = torch.from_numpy(self.arrays["cond_proc"][idx])
        hlf = torch.from_numpy(self.arrays["X_hlf"][idx])
        y = torch.from_numpy(self.arrays["y"][idx])
        return [(X[i], cond[i], hlf[i], y[i]) for i in range(len(idx))]


class _ZarrDataset(Dataset):
    """Loads cached data from zarr store.
    
    Similar to _MemmapDataset but uses zarr format for potential compression
    and better handling of large datasets. Requires zarr to be installed.
    """
    def __init__(self, store_path: Path):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("cache_mode='zarr' requires zarr to be installed") from exc

        self.group = zarr.open_group(store_path, mode="r")
        self._length = int(self.group["y"].shape[0])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(np.asarray(self.group["X_proc"][index])),
            torch.from_numpy(np.asarray(self.group["cond_proc"][index])),
            torch.from_numpy(np.asarray(self.group["X_hlf"][index])),
            torch.from_numpy(np.asarray(self.group["y"][index])),
        )

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        X = torch.from_numpy(np.asarray(self.group["X_proc"][idx]))
        cond = torch.from_numpy(np.asarray(self.group["cond_proc"][idx]))
        hlf = torch.from_numpy(np.asarray(self.group["X_hlf"][idx]))
        y = torch.from_numpy(np.asarray(self.group["y"][idx]))
        return [(X[i], cond[i], hlf[i], y[i]) for i in range(len(idx))]


class LargeHDF5MLPDataModule(LightningDataModule):
    """Lightning DataModule for large HDF5 files with on-the-fly preprocessing.

    Designed to handle large raw HDF5 datasets without creating derived split files.
    By default, preprocessing (HLF, normalization, flattening) happens on-the-fly
    in workers. Optional cache (memmap or zarr) can be enabled for faster subsequent runs.
    
    Config usage:
        data:
          class_path: data.raw_datamodule.LargeHDF5MLPDataModule
          init_args:
            gen_data_path: /path/to/gen.hdf5
            truth_data_path: /path/to/truth.hdf5
            test_data_path: /path/to/test.hdf5
            xml_path: /path/to/binning.xml
            particle: pion
            batch_size: 8192
            min_energy: 9500
            cache_mode: none  # or memmap, zarr
    """

    def __init__(
        self,
        gen_data_path: str,
        truth_data_path: str,
        test_data_path: str,
        xml_path: str,
        particle: str,
        val_fraction: float = 0.05,
        batch_size: int = 1024,
        num_workers: int = 2,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        min_energy: Optional[float] = None,
        voxel_energy_cutoff: Optional[float] = None,
        log_transform: bool = False,
        split_seed: int = 42,
        cache_mode: str = "none",
        cache_dir: Optional[str] = None,
        cache_chunk_size: int = 8192,
    ):
        super().__init__()
        self.gen_data_path = gen_data_path
        self.truth_data_path = truth_data_path
        self.test_data_path = test_data_path
        self.xml_path = xml_path
        self.particle = particle
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.min_energy = min_energy
        self.voxel_energy_cutoff = voxel_energy_cutoff
        self.log_transform = log_transform
        self.split_seed = split_seed
        self.cache_mode = cache_mode
        self.cache_dir = cache_dir
        self.cache_chunk_size = cache_chunk_size

        self._split_indices: dict[str, dict[str, np.ndarray]] = {}

    def _validate_cache_mode(self):
        allowed = {"none", "memmap", "zarr"}
        if self.cache_mode not in allowed:
            raise ValueError(f"cache_mode must be one of {sorted(allowed)}")

    def _filter_indices_by_energy(self, file_path: str, indices: np.ndarray) -> np.ndarray:
        """Filter indices to keep only samples with incident_energy >= min_energy."""
        if self.min_energy is None:
            return indices

        with h5py.File(file_path, "r") as handle:
            energies = np.asarray(handle["incident_energies"][indices], dtype=np.float32).reshape(-1)
        filtered = indices[energies >= self.min_energy]
        return filtered

    def _build_split_indices(self):
        """Compute and cache train/val/test split indices with energy filtering.
        
        Reads full dataset sizes, applies min_energy filter, then splits gen and truth
        into train/val with deterministic random seed. Test is kept whole.
        """
        if self._split_indices:
            return

        gen_source = _RawHDF5Source(self.gen_data_path)
        truth_source = _RawHDF5Source(self.truth_data_path)
        test_source = _RawHDF5Source(self.test_data_path)

        gen_indices = np.arange(len(gen_source))
        truth_indices = np.arange(len(truth_source))
        test_indices = np.arange(len(test_source))

        gen_indices = self._filter_indices_by_energy(self.gen_data_path, gen_indices)
        truth_indices = self._filter_indices_by_energy(self.truth_data_path, truth_indices)
        test_indices = self._filter_indices_by_energy(self.test_data_path, test_indices)

        gen_train, gen_val = train_test_split(
            gen_indices,
            test_size=self.val_fraction,
            random_state=self.split_seed,
            shuffle=True,
        )
        truth_train, truth_val = train_test_split(
            truth_indices,
            test_size=self.val_fraction,
            random_state=self.split_seed,
            shuffle=True,
        )

        self._split_indices = {
            "train": {"gen": gen_train, "truth": truth_train},
            "val": {"gen": gen_val, "truth": truth_val},
            "test": {"truth": test_indices},
        }

    def _cache_root(self) -> Path:
        if self.cache_dir is None:
            return Path("/tmp/caloxtreme_clf_cache")
        return Path(self.cache_dir)

    def _memmap_meta_path(self, split: str) -> Path:
        return self._cache_root() / f"{split}_memmap.json"

    def _zarr_store_path(self, split: str) -> Path:
        return self._cache_root() / f"{split}.zarr"

    def _cache_exists(self, split: str) -> bool:
        if self.cache_mode == "memmap":
            return self._memmap_meta_path(split).exists()
        if self.cache_mode == "zarr":
            return self._zarr_store_path(split).exists()
        return False

    def _write_memmap_cache(self, split: str, dataset: _ConcatDataset):
        """Write preprocessed data to memmap files for fast subsequent loads.
        
        Creates .dat files for each field (X_proc, cond_proc, X_hlf, y) and a
        metadata JSON file that describes their shape/dtype/path for _MemmapDataset.
        """
        meta_path = self._memmap_meta_path(split)
        if meta_path.exists():
            return

        cache_root = meta_path.parent
        cache_root.mkdir(parents=True, exist_ok=True)

        total_len = len(dataset)
        if total_len == 0:
            return

        first_count = min(self.cache_chunk_size, total_len)
        X0, cond0, hlf0, y0 = dataset.get_numpy_batch(np.arange(first_count))
        x_dim = int(X0.shape[1])
        hlf_dim = int(hlf0.shape[1])

        fields = {
            "X_proc": {"file": f"{split}_X_proc.dat", "shape": [total_len, x_dim], "dtype": "float32"},
            "cond_proc": {"file": f"{split}_cond_proc.dat", "shape": [total_len, 1], "dtype": "float32"},
            "X_hlf": {"file": f"{split}_X_hlf.dat", "shape": [total_len, hlf_dim], "dtype": "float32"},
            "y": {"file": f"{split}_y.dat", "shape": [total_len], "dtype": "float32"},
        }

        arrays = {}
        for name, info in fields.items():
            file_path = cache_root / info["file"]
            arrays[name] = np.memmap(file_path, mode="w+", dtype=info["dtype"], shape=tuple(info["shape"]))

        for start in tqdm(
            range(0, total_len, self.cache_chunk_size),
            desc=f"memmap cache '{split}'",
        ):
            stop = min(start + self.cache_chunk_size, total_len)
            batch_indices = np.arange(start, stop)
            X_proc, cond_proc, X_hlf, y = dataset.get_numpy_batch(batch_indices)
            arrays["X_proc"][start:stop] = X_proc
            arrays["cond_proc"][start:stop] = cond_proc
            arrays["X_hlf"][start:stop] = X_hlf
            arrays["y"][start:stop] = y

        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump({"format": "memmap", "fields": fields}, handle, indent=2)

    def _write_zarr_cache(self, split: str, dataset: _ConcatDataset):
        """Write preprocessed data to zarr store for compression and random access.
        
        Similar to _write_memmap_cache but uses zarr format which may offer
        better compression and scalability for very large datasets.
        """
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("cache_mode='zarr' requires zarr to be installed") from exc

        store_path = self._zarr_store_path(split)
        if store_path.exists():
            return

        store_path.parent.mkdir(parents=True, exist_ok=True)
        total_len = len(dataset)
        if total_len == 0:
            return

        first_count = min(self.cache_chunk_size, total_len)
        X0, cond0, hlf0, _ = dataset.get_numpy_batch(np.arange(first_count))
        x_dim = int(X0.shape[1])
        hlf_dim = int(hlf0.shape[1])

        group = zarr.open_group(store_path, mode="w")
        group.create_dataset("X_proc", shape=(total_len, x_dim), dtype="float32", chunks=(self.cache_chunk_size, x_dim))
        group.create_dataset("cond_proc", shape=(total_len, 1), dtype="float32", chunks=(self.cache_chunk_size, 1))
        group.create_dataset("X_hlf", shape=(total_len, hlf_dim), dtype="float32", chunks=(self.cache_chunk_size, hlf_dim))
        group.create_dataset("y", shape=(total_len,), dtype="float32", chunks=(self.cache_chunk_size,))

        for start in tqdm(
            range(0, total_len, self.cache_chunk_size),
            desc=f"zarr cache '{split}'",
        ):
            stop = min(start + self.cache_chunk_size, total_len)
            batch_indices = np.arange(start, stop)
            X_proc, cond_proc, X_hlf, y = dataset.get_numpy_batch(batch_indices)
            group["X_proc"][start:stop] = X_proc
            group["cond_proc"][start:stop] = cond_proc
            group["X_hlf"][start:stop] = X_hlf
            group["y"][start:stop] = y

    def _maybe_build_cache(self, split: str, dataset: _ConcatDataset):
        if self.cache_mode == "none":
            return

        if self._cache_exists(split):
            return

        rank_zero_info(f"Building {self.cache_mode} cache for split '{split}'...")
        if self.cache_mode == "memmap":
            self._write_memmap_cache(split, dataset)
        elif self.cache_mode == "zarr":
            self._write_zarr_cache(split, dataset)

    def _build_raw_split_dataset(self, split: str) -> _ConcatDataset:
        gen_source = _RawHDF5Source(self.gen_data_path)
        truth_source = _RawHDF5Source(self.truth_data_path)
        test_source = _RawHDF5Source(self.test_data_path)

        if split in ("train", "val"):
            gen_indices = self._split_indices[split]["gen"]
            truth_indices = self._split_indices[split]["truth"]
            gen_ds = _RawProcessedDataset(
                gen_source,
                gen_indices,
                label=0.0,
                xml_path=self.xml_path,
                particle=self.particle,
                voxel_energy_cutoff=self.voxel_energy_cutoff,
                log_transform=self.log_transform,
            )
            truth_ds = _RawProcessedDataset(
                truth_source,
                truth_indices,
                label=1.0,
                xml_path=self.xml_path,
                particle=self.particle,
                voxel_energy_cutoff=self.voxel_energy_cutoff,
                log_transform=self.log_transform,
            )
            return _ConcatDataset([gen_ds, truth_ds])

        test_indices = self._split_indices["test"]["truth"]
        test_ds = _RawProcessedDataset(
            test_source,
            test_indices,
            label=1.0,
            xml_path=self.xml_path,
            particle=self.particle,
            voxel_energy_cutoff=self.voxel_energy_cutoff,
            log_transform=self.log_transform,
        )
        return _ConcatDataset([test_ds])

    def _build_cached_split_dataset(self, split: str) -> Dataset:
        if self.cache_mode == "memmap":
            return _MemmapDataset(self._memmap_meta_path(split))
        if self.cache_mode == "zarr":
            return _ZarrDataset(self._zarr_store_path(split))
        raise ValueError(f"Unsupported cache_mode {self.cache_mode}")

    # def prepare_data(self):
    #     self._validate_cache_mode()
    #     if self.cache_mode == "none":
    #         return

    #     self._build_split_indices()
    #     for split in ("train", "val", "test"):
    #         dataset = self._build_raw_split_dataset(split)
    #         self._maybe_build_cache(split, dataset)

    def setup(self, stage: str = "fit"):
        self._validate_cache_mode()
        self._build_split_indices()

        if stage == "fit" or stage is None:
            train_raw = self._build_raw_split_dataset("train")
            val_raw = self._build_raw_split_dataset("val")
            self._maybe_build_cache("train", train_raw)
            self._maybe_build_cache("val", val_raw)

            if self.cache_mode == "none":
                self.train_dataset = train_raw
                self.val_dataset = val_raw
            else:
                self.train_dataset = self._build_cached_split_dataset("train")
                self.val_dataset = self._build_cached_split_dataset("val")

        if stage == "test" or stage is None:
            test_raw = self._build_raw_split_dataset("test")
            self._maybe_build_cache("test", test_raw)

            if self.cache_mode == "none":
                self.test_dataset = test_raw
            else:
                self.test_dataset = self._build_cached_split_dataset("test")

    def _dataloader_kwargs(self, shuffle: bool = False):
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = True

        return kwargs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._dataloader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._dataloader_kwargs())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._dataloader_kwargs())
