#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_from_class_path(class_path: str):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_datamodule(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    dm_cls = _import_from_class_path(data_cfg['class_path'])
    dm = dm_cls(**data_cfg.get('init_args', {}))
    return dm


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    return batch


def run_case(config_path: str, steps: int, warmup: int, transfer_to_gpu: bool) -> dict[str, float | str]:
    dm = _build_datamodule(config_path)
    # Reuse already materialized e9500 split/cache files when present.
    dm.setup(stage='fit')
    loader = dm.train_dataloader()

    it = iter(loader)
    device = torch.device('cuda' if transfer_to_gpu and torch.cuda.is_available() else 'cpu')

    total_steps = 0
    total_samples = 0

    # Warmup phase: exclude first batches for stable measurement.
    for _ in range(warmup):
        batch = next(it)
        if device.type == 'cuda':
            _ = _to_device(batch, device)
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        batch = next(it)
        if device.type == 'cuda':
            _ = _to_device(batch, device)
            torch.cuda.synchronize()

        if isinstance(batch, (list, tuple)) and len(batch) > 0 and hasattr(batch[0], 'shape'):
            total_samples += int(batch[0].shape[0])
        else:
            # Fallback in uncommon collate formats.
            total_samples += 0
        total_steps += 1
    dt = time.perf_counter() - t0

    return {
        'config': config_path,
        'steps': float(total_steps),
        'seconds': dt,
        'batches_per_sec': total_steps / dt if dt > 0 else 0.0,
        'samples_per_sec': total_samples / dt if dt > 0 else 0.0,
        'device': device.type,
    }


def main():
    parser = argparse.ArgumentParser(description='Measure DataLoader throughput from sandbox configs.')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--optimized-config', required=True)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--gpu-transfer', action='store_true')
    args = parser.parse_args()

    baseline = run_case(args.baseline_config, args.steps, args.warmup, args.gpu_transfer)
    optimized = run_case(args.optimized_config, args.steps, args.warmup, args.gpu_transfer)

    speedup = (optimized['batches_per_sec'] / baseline['batches_per_sec']) if baseline['batches_per_sec'] else 0.0

    print('=== Dataloader Throughput ===')
    print(f"baseline config:  {Path(str(baseline['config'])).as_posix()}")
    print(f"optimized config: {Path(str(optimized['config'])).as_posix()}")
    print(f"device transfer:  {baseline['device']}")
    print('')
    print(f"baseline  : {baseline['batches_per_sec']:.4f} batches/s | {baseline['samples_per_sec']:.1f} samples/s | {baseline['seconds']:.2f}s")
    print(f"optimized : {optimized['batches_per_sec']:.4f} batches/s | {optimized['samples_per_sec']:.1f} samples/s | {optimized['seconds']:.2f}s")
    print(f"speedup   : {speedup:.3f}x ({(speedup - 1.0) * 100.0:+.1f}%)")


if __name__ == '__main__':
    main()
