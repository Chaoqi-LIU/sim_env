# Adapted from https://github.com/Physical-Intelligence/openpi
# -----------------------------------------------------------
"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import numpy as np
import torch
import tqdm
import click
from torch.utils.data import DataLoader
from typing import Dict

from policy_research.dataset.droid_dataset import DroidRldsDataset


# class RunningStats:
#     """Compute running statistics of a batch of vectors."""

#     def __init__(self):
#         self._count = 0
#         self._mean = None
#         self._mean_of_squares = None
#         self._min = None
#         self._max = None
#         self._histograms = None
#         self._bin_edges = None
#         self._num_quantile_bins = 5000  # for computing quantiles on the fly

#     def update(self, batch: np.ndarray) -> None:
#         """
#         Update the running statistics with a batch of vectors.

#         Args:
#             vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
#         """
#         batch = batch.reshape(-1, batch.shape[-1])
#         num_elements, vector_length = batch.shape
#         if self._count == 0:
#             self._mean = np.mean(batch, axis=0)
#             self._mean_of_squares = np.mean(batch**2, axis=0)
#             self._min = np.min(batch, axis=0)
#             self._max = np.max(batch, axis=0)
#             self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
#             self._bin_edges = [
#                 np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
#                 for i in range(vector_length)
#             ]
#         else:
#             if vector_length != self._mean.size:
#                 raise ValueError("The length of new vectors does not match the initialized vector length.")
#             new_max = np.max(batch, axis=0)
#             new_min = np.min(batch, axis=0)
#             max_changed = np.any(new_max > self._max)
#             min_changed = np.any(new_min < self._min)
#             self._max = np.maximum(self._max, new_max)
#             self._min = np.minimum(self._min, new_min)

#             if max_changed or min_changed:
#                 self._adjust_histograms()

#         self._count += num_elements

#         batch_mean = np.mean(batch, axis=0)
#         batch_mean_of_squares = np.mean(batch**2, axis=0)

#         # Update running mean and mean of squares.
#         self._mean += (batch_mean - self._mean) * (num_elements / self._count)
#         self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

#         self._update_histograms(batch)

#     def get_statistics(self) -> Dict[str, np.ndarray]:
#         """
#         Compute and return the statistics of the vectors processed so far.

#         Returns:
#             dict: A dictionary containing the computed statistics.
#         """
#         if self._count < 2:
#             raise ValueError("Cannot compute statistics for less than 2 vectors.")

#         variance = self._mean_of_squares - self._mean**2
#         stddev = np.sqrt(np.maximum(0, variance))
#         q01, q99 = self._compute_quantiles([0.01, 0.99])
#         return dict(mean=self._mean, std=stddev, q01=q01, q99=q99)

#     def _adjust_histograms(self):
#         """Adjust histograms when min or max changes."""
#         for i in range(len(self._histograms)):
#             old_edges = self._bin_edges[i]
#             new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

#             # Redistribute the existing histogram counts to the new bins
#             new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

#             self._histograms[i] = new_hist
#             self._bin_edges[i] = new_edges

#     def _update_histograms(self, batch: np.ndarray) -> None:
#         """Update histograms with new vectors."""
#         for i in range(batch.shape[1]):
#             hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
#             self._histograms[i] += hist

#     def _compute_quantiles(self, quantiles):
#         """Compute quantiles based on histograms."""
#         results = []
#         for q in quantiles:
#             target_count = q * self._count
#             q_values = []
#             for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
#                 cumsum = np.cumsum(hist)
#                 idx = np.searchsorted(cumsum, target_count)
#                 q_values.append(edges[idx])
#             results.append(np.array(q_values))
#         return results

class RunningStats:
    """Fast running mean/std with Welford combine + min/max (torch)."""
    def __init__(self):
        self._count = 0
        self._mean = None
        self._M2   = None
        self._min  = None
        self._max  = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # flatten all leading dims; last dim = features
        x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float64, device='cpu')
        N = x.shape[0]
        if N == 0:
            return

        b_mean = x.mean(dim=0)
        b_min  = x.amin(dim=0)
        b_max  = x.amax(dim=0)
        diff   = x - b_mean
        b_M2   = (diff * diff).sum(dim=0)

        if self._count == 0:
            self._count = N
            self._mean  = b_mean
            self._M2    = b_M2
            self._min   = b_min
            self._max   = b_max
            return

        n1, n2 = self._count, N
        delta = b_mean - self._mean
        new_count = n1 + n2

        self._mean = self._mean + delta * (n2 / new_count)
        self._M2   = self._M2 + b_M2 + (delta * delta) * (n1 * n2 / new_count)
        self._count = new_count
        self._min   = torch.minimum(self._min, b_min)
        self._max   = torch.maximum(self._max, b_max)

    def get_statistics(self) -> Dict[str, np.ndarray]:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        var = torch.clamp(self._M2 / (self._count - 1), min=0.0)
        std = torch.sqrt(var)
        # convert to numpy
        return dict(
            mean=self._mean.to(dtype=torch.float32).numpy(),
            std =std.to(dtype=torch.float32).numpy(),
            min =self._min.to(dtype=torch.float32).numpy(),
            max =self._max.to(dtype=torch.float32).numpy(),
        )


@click.command()
@click.option('-d', '--data_dir', type=str, required=True)
@click.option('--filter_dict_path', type=str, required=True)
@click.option('-s', '--norm_stats_save_path', type=str, required=True)
@click.option('--droid_name', type=str, default="droid")
@click.option('--droid_version', type=str, default="1.0.1")
@click.option('--num_frames', type=int, default=10_000_000)
@click.option('--skip_image', is_flag=True, default=False)
def compute_droid_norm_stats(
    data_dir: str,
    droid_name: str,
    droid_version: str,
    filter_dict_path: str,
    norm_stats_save_path: str,
    num_frames: int,
    skip_image: bool,
):
    nframes_per_chunk = 1
    dataset = DroidRldsDataset(
        data_dir=data_dir,
        droid_name=droid_name,
        droid_version=droid_version,
        n_action_steps=nframes_per_chunk,
        n_obs_steps=nframes_per_chunk,
        filter_dict_path=filter_dict_path,
        norm_stats_path=None,
    )
    batch_size = 2048
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    num_frames = min(num_frames, len(dataset))
    num_batches = num_frames // batch_size // nframes_per_chunk
    sample_batch = next(iter(dataloader))
    keys = [
        "action",
        *[key for key in sample_batch["observation"].keys()],
    ]
    if skip_image:
        keys = [key for key in keys if "image" not in key]
    print(f"Computing stats for keys: {keys}")
    stats = {key: RunningStats() for key in keys}

    print(f"{num_batches=}")
    for batch_idx, batch in enumerate(tqdm.tqdm(
        dataloader, total=num_batches, desc="Computing stats"
    )):
        if batch_idx >= num_batches:
            break
        for key in keys:
            if key == 'action':
                stats[key].update(batch[key])
            else:
                stats[key].update(batch['observation'][key])
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # save dict
    np.save(norm_stats_save_path, norm_stats)
    print(f"Saved normalization stats to {norm_stats_save_path}")


if __name__ == "__main__":
    compute_droid_norm_stats()
