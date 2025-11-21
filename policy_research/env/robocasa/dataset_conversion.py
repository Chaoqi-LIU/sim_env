import h5py
import json
import numpy as np
import tqdm
from robocasa.utils.dataset_registry import get_ds_path
from typing import Optional

from policy_research.common.replay_buffer import ReplayBuffer


def convert_robocasa_hdf5_to_zarr(
    task_name: str,
    ds_type: str = "human_im",
    sample_ndemo: Optional[int] = None,
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr()
    hdf5_path = get_ds_path(task=task_name, ds_type=ds_type)
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        num_demo = len(data)
        if sample_ndemo is None:
            sample_ndemo = num_demo
        sample_ndemo = min(sample_ndemo, num_demo)
        sample_demo_indices = np.random.choice(num_demo, sample_ndemo, replace=False)
        offset = 0
        # check demo idx starts from 0 or 1
        try:
            _ = data['demo_0']
        except KeyError:
            offset = 1
        # for demo_idx in tqdm.tqdm(range(num_demo), desc=f"Converting {task_name} ({ds_type})"):
        for demo_idx in tqdm.tqdm(sample_demo_indices, desc=f"Converting {task_name} ({ds_type})"):
            demo = data[f'demo_{demo_idx + offset}']
            meta = json.loads(demo.attrs["ep_meta"])
            task_prompt: str = meta["lang"]

            demo_len = len(demo['actions'])
            this_data_collected = {
                'action': demo['actions'][:].astype(np.float32),
                'robot0_agentview_left_rgb': demo['obs']['robot0_agentview_left_image'][:].astype(np.uint8),
                'robot0_agentview_right_rgb': demo['obs']['robot0_agentview_right_image'][:].astype(np.uint8),
                'robot0_eye_in_hand_rgb': demo['obs']['robot0_eye_in_hand_image'][:].astype(np.uint8),
                'robot0_joint_pos': demo['obs']['robot0_joint_pos'][:].astype(np.float32),
                'robot0_eef_pos': demo['obs']['robot0_eef_pos'][:].astype(np.float32),
                'robot0_eef_quat': demo['obs']['robot0_eef_quat'][:].astype(np.float32),
                'robot0_gripper_qpos': demo['obs']['robot0_gripper_qpos'][:].astype(np.float32),
                'prompt': np.array([task_prompt] * demo_len),
            }
            replay_buffer.add_episode(this_data_collected)
    
    # print
    print('-' * 50)
    print(f"Task: {task_name}\n{replay_buffer}")

    return replay_buffer
