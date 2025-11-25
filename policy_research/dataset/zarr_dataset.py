import torch
import numpy as np
import copy

from policy_research.common.pytorch_util import dict_apply
from policy_research.common.replay_buffer import ReplayBuffer
from policy_research.common.seq_sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from policy_research.model.common.normalizer import LinearNormalizer
from policy_research.dataset.base_dataset import BaseDataset

from typing import Dict, Optional, List


def is_numeric_dtype(x):
    return x.dtype.kind in 'biuf'  # bool, int, uint, float


class ZarrDataset(BaseDataset):
    def __init__(
        self,
        zarr_path: str,
        obs_keys: List[str] = [],
        action_key: str = 'action',
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()
        assert n_obs_steps + n_action_steps > 0, "should have at least one frame"

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[action_key, *obs_keys],
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        pad_before = max(n_obs_steps - 1, 0)
        pad_after = max(n_action_steps - 1, 0)
        seq_len = pad_before + 1 + pad_after
        self.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=seq_len,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        # text observation keys
        sample0 = self.seq_sampler.sample_sequence(0)
        numeric_obs_keys = []
        text_obs_keys = []
        for k in obs_keys:
            if is_numeric_dtype(sample0[k]):
                numeric_obs_keys.append(k)
            else:
                text_obs_keys.append(k)

        self.train_mask = train_mask
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.seq_len = seq_len
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_keys = obs_keys
        self.numeric_obs_keys = numeric_obs_keys
        self.text_obs_keys = text_obs_keys
        self.action_key = action_key


    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer[self.action_key],
            **{k: self.replay_buffer[k] for k in self.numeric_obs_keys}
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    

    def __len__(self):
        return len(self.seq_sampler)
    

    def _sample_to_data(self, sample):
        To = self.n_obs_steps
        Ta = self.n_action_steps

        obs = {}
        for k in self.numeric_obs_keys:
            if sample[k].dtype.kind == 'f': # floatX -> float32
                obs[k] = sample[k][:To].astype(np.float32)
            else:                           # remain the same dtype
                obs[k] = sample[k][:To]
        for k in self.text_obs_keys:
            obs[k] = sample[k][0]           # every frame is the same

        start = max(To - 1, 0)
        end = start + Ta
        act = sample[self.action_key][start:end].astype(np.float32)
        assert np.allclose(act, sample[self.action_key][-Ta:]), "action mismatch"

        return {'obs': obs, 'action': act}
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.seq_sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        # torch_data = dict_apply(data, lambda x: torch.from_numpy(x))
        torch_obs = {}
        for k, v in data['obs'].items():
            if isinstance(v, np.ndarray) and is_numeric_dtype(v):
                torch_obs[k] = torch.from_numpy(v)
            elif isinstance(v, bytes):
                torch_obs[k] = v.decode('utf-8')
            else:
                torch_obs[k] = v
        torch_act = torch.from_numpy(data['action'])
        return {'obs': torch_obs, 'action': torch_act}
