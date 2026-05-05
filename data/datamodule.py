import hashlib
import os
import shutil
from bisect import bisect_right
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from lightning.pytorch.core import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data.utils import get_high_level_features, lookup_hdf5, scale_shower
from tqdm import tqdm

_H5_RDCC = {"rdcc_nbytes": 64 * 1024 * 1024, "rdcc_nslots": 4093}
_H5_ROW_CHUNK = 1024


class HDF5Dataset(Dataset):
    def __init__(self, file_path: str, fields: tuple[str, ...]):
        self.file_path = file_path
        self.fields = fields
        self._file = None

        with h5py.File(self.file_path, "r") as handle:
            self._length = int(handle[self.fields[0]].shape[0])

    def __len__(self):
        return self._length

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r", **_H5_RDCC)
        return self._file

    @staticmethod
    def _read_rows(dataset, indices: np.ndarray):
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

    def __getitem__(self, index):
        handle = self._get_file()
        return tuple(handle[field][index] for field in self.fields)

    def __getitems__(self, indices):
        handle = self._get_file()
        batch_indices = np.asarray(indices, dtype=np.int64)

        field_batches = [self._read_rows(handle[field], batch_indices) for field in self.fields]
        return [tuple(row) for row in zip(*field_batches)]

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


class MultiHDF5Dataset(Dataset):
    def __init__(self, file_paths: list[str], fields: tuple[str, ...]):
        self.file_paths = file_paths
        self.fields = fields
        self._files = None
        self._lengths = []
        self._cumulative_lengths = []

        total = 0
        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as handle:
                length = int(handle[self.fields[0]].shape[0])
            self._lengths.append(length)
            total += length
            self._cumulative_lengths.append(total)

        self._length = total

    def __len__(self):
        return self._length

    def _get_files(self):
        if self._files is None:
            self._files = [h5py.File(file_path, "r", **_H5_RDCC) for file_path in self.file_paths]
        return self._files

    def _locate_index(self, index: int):
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("index out of range")

        file_idx = bisect_right(self._cumulative_lengths, index)
        previous_total = 0 if file_idx == 0 else self._cumulative_lengths[file_idx - 1]
        local_index = index - previous_total
        return file_idx, local_index

    def __getitem__(self, index):
        file_idx, local_index = self._locate_index(int(index))
        handle = self._get_files()[file_idx]
        return tuple(handle[field][local_index] for field in self.fields)

    def __getitems__(self, indices):
        requested_indices = np.asarray(indices, dtype=np.int64)
        files = self._get_files()

        grouped = {}
        for output_position, global_index in enumerate(requested_indices.tolist()):
            file_idx, local_index = self._locate_index(int(global_index))
            grouped.setdefault(file_idx, []).append((output_position, local_index))

        output: list[Optional[tuple]] = [None] * len(requested_indices)
        for file_idx, items in grouped.items():
            positions = np.asarray([pos for pos, _ in items], dtype=np.int64)
            local_indices = np.asarray([idx for _, idx in items], dtype=np.int64)
            handle = files[file_idx]

            field_batches = [HDF5Dataset._read_rows(handle[field], local_indices) for field in self.fields]
            grouped_rows = [tuple(row) for row in zip(*field_batches)]
            for row_idx, output_position in enumerate(positions):
                output[int(output_position)] = grouped_rows[row_idx]

        assert all(s is not None for s in output), "Some batch indices failed to resolve in MultiHDF5Dataset.__getitems__"
        return output

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = None
        return state

    def __del__(self):
        files = getattr(self, "_files", None)
        if files is None:
            return

        for file_handle in files:
            if file_handle is None:
                continue
            try:
                file_handle.close()
            except Exception:
                pass


class BaseDataModule(LightningDataModule):
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
        tmp_dir: str = "/tmp/caloxtreme_clf",
        split_chunk_size: int = 100000,
        use_lazy_hdf5: bool = False,
        min_energy: Optional[float] = None,
        voxel_energy_cutoff: Optional[float] = None,
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
        self.tmp_dir = tmp_dir
        self.split_chunk_size = split_chunk_size
        self.use_lazy_hdf5 = use_lazy_hdf5
        self.min_energy = min_energy
        self.voxel_energy_cutoff = voxel_energy_cutoff

        # Generate threshold-specific split filenames to prevent stale cache reuse
        energy_suffix = "" if min_energy is None else f"_e{int(min_energy)}"
        voxel_cutoff_suffix = "" if voxel_energy_cutoff is None else f"_voxelE{int(voxel_energy_cutoff)}"
        self.train_data_files = [
            self.gen_data_path.replace(".hdf5", f"{energy_suffix}{voxel_cutoff_suffix}_train.hdf5"),
            self.truth_data_path.replace(".hdf5", f"{energy_suffix}{voxel_cutoff_suffix}_train.hdf5"),
        ]

        self.val_data_files = [
            self.gen_data_path.replace(".hdf5", f"{energy_suffix}{voxel_cutoff_suffix}_val.hdf5"),
            self.truth_data_path.replace(".hdf5", f"{energy_suffix}{voxel_cutoff_suffix}_val.hdf5"),
        ]

        self.test_data_files = [
            self.test_data_path.replace(".hdf5", f"{energy_suffix}{voxel_cutoff_suffix}_test.hdf5"),
        ]

    def _dataset_fields(self):
        return ("X", "cond", "y")

    def _base_dataset_fields(self):
        """Fields written by _write_subset_file. Subclasses must not override this."""
        return ("X", "cond", "y")

    def _split_source_fields(self):
        return ("showers", "incident_energies")

    def _has_fields(self, file_path: str, required_fields: tuple[str, ...]):
        if not os.path.exists(file_path):
            return False

        with h5py.File(file_path, "r") as handle:
            return all(field in handle for field in required_fields)

    def _signature(self, file_path: str):
        stat_result = os.stat(file_path)
        signature = f"{os.path.abspath(file_path)}|{stat_result.st_mtime_ns}|{stat_result.st_size}"
        return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]

    def _stage_file(self, file_path: str):
        source = Path(file_path)
        stage_root = Path(self.tmp_dir)
        stage_root.mkdir(parents=True, exist_ok=True)
        staged_path = stage_root / self._signature(file_path) / source.name

        if staged_path.exists():
            return str(staged_path)

        staged_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = staged_path.with_name(f"{staged_path.name}.{os.getpid()}.tmp")
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, staged_path)
        return str(staged_path)

    def _filter_by_energy(self, indices: np.ndarray, source_path: str) -> np.ndarray:
        """Filter indices to keep only showers with incident_energy >= min_energy."""
        if self.min_energy is None:
            return indices

        rank_zero_info(f"Filtering showers with incident_energy >= {self.min_energy}...")
        with h5py.File(source_path, "r") as f:
            energies = np.asarray(f["incident_energies"][indices], dtype=np.float32).reshape(-1)
        filtered_indices = indices[energies >= self.min_energy]
        rank_zero_info(f"Filtered {len(indices)} → {len(filtered_indices)} showers (threshold: {self.min_energy})")
        return filtered_indices

    def _indexed_rows(self, dataset, indices: np.ndarray):
        if len(indices) == 0:
            return dataset[:0]

        ordered = np.asarray(indices, dtype=np.int64)
        order = np.argsort(ordered)
        sorted_indices = ordered[order]
        data = dataset[sorted_indices]

        if not np.array_equal(order, np.arange(len(order))):
            inverse_order = np.argsort(order)
            data = data[inverse_order]

        return data

    def _write_subset_file(self, source_path: str, target_path: str, indices: np.ndarray, label_value: float):
        if self._has_fields(target_path, self._base_dataset_fields()):
            return

        indices = np.sort(indices)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(source_path, "r") as source, h5py.File(target, "w") as destination:
            source_X = source[self._split_source_fields()[0]]
            source_cond = source[self._split_source_fields()[1]]
            num_entries = int(len(indices))
            x_chunks = (_H5_ROW_CHUNK,) + source_X.shape[1:]
            cond_chunks = (_H5_ROW_CHUNK,) + source_cond.shape[1:]

            X_dataset = destination.create_dataset(
                "X",
                shape=(num_entries,) + source_X.shape[1:],
                dtype=np.float32,
                chunks=x_chunks,
                compression="lzf",
            )
            cond_dataset = destination.create_dataset(
                "cond",
                shape=(num_entries,) + source_cond.shape[1:],
                dtype=np.float32,
                chunks=cond_chunks,
                compression="lzf",
            )
            y_dataset = destination.create_dataset(
                "y",
                shape=(num_entries,),
                dtype=np.float32,
                chunks=(_H5_ROW_CHUNK,),
            )

            for start in tqdm(range(0, num_entries, self.split_chunk_size), desc=f"Writing {target_path}"):
                stop = min(start + self.split_chunk_size, num_entries)
                chunk_indices = indices[start:stop]

                X_dataset[start:stop] = np.asarray(source_X[chunk_indices], dtype=np.float32)
                cond_dataset[start:stop] = np.asarray(source_cond[chunk_indices], dtype=np.float32)
                y_dataset[start:stop] = np.full(stop - start, label_value, dtype=np.float32)

    def _write_two_subset_files(
        self,
        source_path: str,
        target_a: str, indices_a: np.ndarray, label_a: float,
        target_b: str, indices_b: np.ndarray, label_b: float,
    ):
        a_done = self._has_fields(target_a, self._base_dataset_fields())
        b_done = self._has_fields(target_b, self._base_dataset_fields())

        if a_done and b_done:
            return
        if a_done:
            self._write_subset_file(source_path, target_b, indices_b, label_b)
            return
        if b_done:
            self._write_subset_file(source_path, target_a, indices_a, label_a)
            return

        tags_a = np.ones(len(indices_a), dtype=bool)
        tags_b = np.zeros(len(indices_b), dtype=bool)
        merged_indices = np.concatenate([indices_a, indices_b])
        merged_tags = np.concatenate([tags_a, tags_b])
        order = np.argsort(merged_indices, kind="stable")
        merged_indices = merged_indices[order]
        merged_tags = merged_tags[order]

        Path(target_a).parent.mkdir(parents=True, exist_ok=True)
        Path(target_b).parent.mkdir(parents=True, exist_ok=True)

        with (
            h5py.File(source_path, "r") as source,
            h5py.File(target_a, "w") as dest_a,
            h5py.File(target_b, "w") as dest_b,
        ):
            src_X = source[self._split_source_fields()[0]]
            src_cond = source[self._split_source_fields()[1]]
            n_a, n_b = int(len(indices_a)), int(len(indices_b))
            x_chunks = (_H5_ROW_CHUNK,) + src_X.shape[1:]
            cond_chunks = (_H5_ROW_CHUNK,) + src_cond.shape[1:]

            def _make_datasets(dest, n):
                Xd = dest.create_dataset("X", shape=(n,) + src_X.shape[1:], dtype=np.float32, chunks=x_chunks, compression="lzf")
                cd = dest.create_dataset("cond", shape=(n,) + src_cond.shape[1:], dtype=np.float32, chunks=cond_chunks, compression="lzf")
                yd = dest.create_dataset("y", shape=(n,), dtype=np.float32, chunks=(_H5_ROW_CHUNK,))
                return Xd, cd, yd

            Xa, ca, ya = _make_datasets(dest_a, n_a)
            Xb, cb, yb = _make_datasets(dest_b, n_b)

            pos_a = pos_b = 0
            total = len(merged_indices)
            for start in tqdm(range(0, total, self.split_chunk_size), desc=f"Writing {Path(target_a).name} + {Path(target_b).name}"):
                stop = min(start + self.split_chunk_size, total)
                cidx = merged_indices[start:stop]
                ctag = merged_tags[start:stop]

                chunk_X = np.asarray(src_X[cidx], dtype=np.float32)
                chunk_cond = np.asarray(src_cond[cidx], dtype=np.float32)

                mask_a = ctag
                mask_b = ~ctag
                na, nb = int(mask_a.sum()), int(mask_b.sum())

                if na:
                    Xa[pos_a:pos_a + na] = chunk_X[mask_a]
                    ca[pos_a:pos_a + na] = chunk_cond[mask_a]
                    ya[pos_a:pos_a + na] = label_a
                    pos_a += na
                if nb:
                    Xb[pos_b:pos_b + nb] = chunk_X[mask_b]
                    cb[pos_b:pos_b + nb] = chunk_cond[mask_b]
                    yb[pos_b:pos_b + nb] = label_b
                    pos_b += nb

    def _build_lazy_dataset(self, file_paths: list[str]):
        if len(file_paths) == 1:
            return HDF5Dataset(file_paths[0], self._dataset_fields())
        return MultiHDF5Dataset(file_paths, self._dataset_fields())

    def _build_in_memory_dataset(self, file_paths: list[str]):
        fields = self._dataset_fields()
        tensors_per_field = [[] for _ in fields]

        for file_path in file_paths:
            with h5py.File(file_path, "r") as handle:
                for field_idx, field_name in enumerate(fields):
                    field_array = np.asarray(handle[field_name], dtype=np.float32)
                    tensors_per_field[field_idx].append(torch.from_numpy(field_array))

        concatenated_tensors = []
        for tensor_group in tensors_per_field:
            if len(tensor_group) == 1:
                concatenated_tensors.append(tensor_group[0])
            else:
                concatenated_tensors.append(torch.cat(tensor_group, dim=0))

        return TensorDataset(*concatenated_tensors)

    def _build_dataset(self, file_paths: list[str]):
        if self.use_lazy_hdf5:
            return self._build_lazy_dataset(file_paths)
        return self._build_in_memory_dataset(file_paths)

    def _materialized_file_paths(self, file_paths: list[str]):
        if self.use_lazy_hdf5:
            return [self._stage_file(file_path) for file_path in file_paths]
        return file_paths

    def prepare_data(self):
        if all([os.path.exists(f) for f in self.train_data_files + self.val_data_files + self.test_data_files]):
            return

        rank_zero_info("Preparing data...")

        if not all([os.path.exists(f) for f in self.train_data_files + self.val_data_files]):
            num_gen = lookup_hdf5(self.gen_data_path, field=self._split_source_fields()[0])
            num_truth = lookup_hdf5(self.truth_data_path, field=self._split_source_fields()[0])

            gen_indices = np.arange(num_gen)
            truth_indices = np.arange(num_truth)

            # Apply energy filter before split to ensure filtered dataset is cached
            gen_indices = self._filter_by_energy(gen_indices, self.gen_data_path)
            truth_indices = self._filter_by_energy(truth_indices, self.truth_data_path)

            gen_train_indices, gen_val_indices = train_test_split(
                gen_indices,
                test_size=self.val_fraction,
                random_state=42,
                shuffle=True,
            )
            truth_train_indices, truth_val_indices = train_test_split(
                truth_indices,
                test_size=self.val_fraction,
                random_state=42,
                shuffle=True,
            )

            self._write_two_subset_files(
                self.gen_data_path,
                self.train_data_files[0], gen_train_indices, 0.0,
                self.val_data_files[0], gen_val_indices, 0.0,
            )
            rank_zero_info("Gen train/val data prepared and saved.")

            self._write_two_subset_files(
                self.truth_data_path,
                self.train_data_files[1], truth_train_indices, 1.0,
                self.val_data_files[1], truth_val_indices, 1.0,
            )
            rank_zero_info("Truth train/val data prepared and saved.")

        if not os.path.exists(self.test_data_files[0]):
            test_num_entries = lookup_hdf5(self.test_data_path, field=self._split_source_fields()[0])
            test_indices = np.arange(test_num_entries)
            test_indices: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._filter_by_energy(test_indices, self.test_data_path)
            self._write_subset_file(self.test_data_path, self.test_data_files[0], test_indices, 1.0)
            rank_zero_info("Test data prepared and saved to {}.".format(self.test_data_files[0]))


    def _preprocess_data(self, X, cond):
        return X, cond

    def setup(self, stage: str = "fit"):
        mode = "lazy HDF5Dataset" if self.use_lazy_hdf5 else "in-memory TensorDataset"
        rank_zero_info(f"Creating datasets ({mode})...")

        if stage == "fit" or stage is None:
            train_files = self._materialized_file_paths(self.train_data_files)
            val_files = self._materialized_file_paths(self.val_data_files)

            self.train_dataset = self._build_dataset(train_files)
            self.val_dataset = self._build_dataset(val_files)

        if stage == "test" or stage is None:
            test_files = self._materialized_file_paths(self.test_data_files)
            self.test_dataset = self._build_dataset(test_files)

        rank_zero_info("Datasets created.")

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


class SimpleMLPDataModule(BaseDataModule):
    def __init__(self, *args, use_synthetic_data: bool = False, feature_chunk_size: int = 8192, synthetic_hlf_dim: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_synthetic_data = use_synthetic_data
        self.feature_chunk_size = feature_chunk_size
        self.synthetic_hlf_dim = synthetic_hlf_dim

    def _dataset_fields(self):
        return ("X_proc", "cond_proc", "X_hlf", "y")

    def _preprocess_data(self, X, cond):
        if self.voxel_energy_cutoff is not None:
            X = np.where(X > self.voxel_energy_cutoff, X, 0.0)
        X_hlf = get_high_level_features(X, cond, self.xml_path, self.particle)
        X, cond = scale_shower(X, cond)
        X = X.reshape(X.shape[0], -1)
        return X, cond, X_hlf

    def _write_processed_fields(self, file_path: str):
        processed_fields = self._dataset_fields()[:-1]
        if self._has_fields(file_path, processed_fields):
            return

        with h5py.File(file_path, "a") as handle:
            for field_name in processed_fields:
                if field_name in handle:
                    del handle[field_name]

            raw_X = handle["X"]
            raw_cond = handle["cond"]
            num_entries = int(raw_X.shape[0])
            flattened_dim = int(np.prod(raw_X.shape[1:]))

            X_dataset = handle.create_dataset(
                "X_proc",
                shape=(num_entries, flattened_dim),
                dtype=np.float32,
                chunks=(_H5_ROW_CHUNK, flattened_dim),
                compression="lzf",
            )
            cond_dataset = handle.create_dataset(
                "cond_proc",
                shape=(num_entries, 1),
                dtype=np.float32,
                chunks=(_H5_ROW_CHUNK, 1),
            )
            hlf_example = None
            hlf_dataset = None

            for start in tqdm(range(0, num_entries, self.feature_chunk_size), desc=f"Computing and writing High-level features to {file_path}"):
                stop = min(start + self.feature_chunk_size, num_entries)

                X_chunk = np.asarray(raw_X[start:stop], dtype=np.float32)
                cond_chunk = np.asarray(raw_cond[start:stop], dtype=np.float32)

                # hlf_chunk = get_high_level_features(X_chunk.copy(), cond_chunk.copy(), self.xml_path, self.particle)
                # X_chunk, cond_chunk = scale_shower(X_chunk, cond_chunk)
                # X_chunk = X_chunk.reshape(X_chunk.shape[0], -1)

                X_chunk, cond_chunk, hlf_chunk = self._preprocess_data(X_chunk, cond_chunk)

                if hlf_dataset is None:
                    hlf_example = hlf_chunk
                    hlf_dataset = handle.create_dataset(
                        "X_hlf",
                        shape=(num_entries, hlf_example.shape[1]),
                        dtype=np.float32,
                        chunks=(_H5_ROW_CHUNK, hlf_example.shape[1]),
                    )

                X_dataset[start:stop] = X_chunk.astype(np.float32, copy=False)
                cond_dataset[start:stop] = cond_chunk.astype(np.float32, copy=False)
                hlf_dataset[start:stop] = hlf_chunk.astype(np.float32, copy=False)

    def prepare_data(self):
        if self.use_synthetic_data:
            rank_zero_info("Using synthetic data for testing.")
            return

        super().prepare_data()

        rank_zero_info("Preprocessing data and computing high-level features...")
        for file_path in self.train_data_files + self.val_data_files + self.test_data_files:
            if os.path.exists(file_path):
                self._write_processed_fields(file_path)

        rank_zero_info("Data preparation complete.")
        

    def setup(self, stage: str = "fit"):
        if self.use_synthetic_data:
            rank_zero_info("Creating synthetic datasets...")
            n_samples = 512
            input_dim = 50

            if stage == "fit" or stage is None:
                train_X = torch.randn(n_samples * 2, input_dim, dtype=torch.float32)
                train_cond = torch.randn(n_samples * 2, 1, dtype=torch.float32)
                train_hlf = torch.randn(n_samples * 2, self.synthetic_hlf_dim, dtype=torch.float32)
                train_y = torch.randint(0, 2, (n_samples * 2, 1), dtype=torch.float32)
                val_X = torch.randn(n_samples // 2, input_dim, dtype=torch.float32)
                val_cond = torch.randn(n_samples // 2, 1, dtype=torch.float32)
                val_hlf = torch.randn(n_samples // 2, self.synthetic_hlf_dim, dtype=torch.float32)
                val_y = torch.randint(0, 2, (n_samples // 2, 1), dtype=torch.float32)

                self.train_dataset = TensorDataset(train_X, train_cond, train_hlf, train_y)
                self.val_dataset = TensorDataset(val_X, val_cond, val_hlf, val_y)

            if stage == "test" or stage is None:
                test_X = torch.randn(n_samples // 4, input_dim, dtype=torch.float32)
                test_cond = torch.randn(n_samples // 4, 1, dtype=torch.float32)
                test_hlf = torch.randn(n_samples // 4, self.synthetic_hlf_dim, dtype=torch.float32)
                test_y = torch.randint(0, 2, (n_samples // 4, 1), dtype=torch.float32)
                self.test_dataset = TensorDataset(test_X, test_cond, test_hlf, test_y)
        else:
            return super().setup(stage)


class SimpleMLPLatentDataModule(SimpleMLPDataModule):

    def _split_source_fields(self):
        return ["latent_features", "incident_energies"]
    
    def _dataset_fields(self):
        return ("X_proc", "cond_proc", "y")

    def _preprocess_data(self, X, cond):
        # X, cond = scale_shower(X, cond)
        # X = X.reshape(X.shape[0], -1)
        cond = cond.reshape(-1, 1) 
        cond /= 1000. # convert to GeV

        return X, cond

    def _write_processed_fields(self, file_path: str):
        processed_fields = self._dataset_fields()[:-1]
        if self._has_fields(file_path, processed_fields):
            return

        with h5py.File(file_path, "a") as handle:
            for field_name in processed_fields:
                if field_name in handle:
                    del handle[field_name]

            raw_X = handle["X"]
            raw_cond = handle["cond"]
            num_entries = int(raw_X.shape[0])
            flattened_dim = int(np.prod(raw_X.shape[1:]))

            X_dataset = handle.create_dataset(
                "X_proc",
                shape=(num_entries, flattened_dim),
                dtype=np.float32,
                chunks=(_H5_ROW_CHUNK, flattened_dim),
                compression="lzf",
            )
            cond_dataset = handle.create_dataset(
                "cond_proc",
                shape=(num_entries, 1),
                dtype=np.float32,
                chunks=(_H5_ROW_CHUNK, 1),
            )

            for start in tqdm(range(0, num_entries, self.feature_chunk_size), desc=f"Saving filtered dataset to {file_path}"):
                stop = min(start + self.feature_chunk_size, num_entries)

                X_chunk = np.asarray(raw_X[start:stop], dtype=np.float32)
                cond_chunk = np.asarray(raw_cond[start:stop], dtype=np.float32)

                X_chunk, cond_chunk = self._preprocess_data(X_chunk, cond_chunk)

                X_dataset[start:stop] = X_chunk.astype(np.float32, copy=False)
                cond_dataset[start:stop] = cond_chunk.astype(np.float32, copy=False)

    