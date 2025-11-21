import h5py
import numpy as np
import tqdm
import json
from robosuite.utils.transform_utils import axisangle2quat
from typing import Optional

from policy_research.env.libero.env import task_name_to_suite_and_ids
from policy_research.common.replay_buffer import ReplayBuffer


axisangle2quat_vectorized = np.vectorize(
    axisangle2quat, 
    signature='(3)->(4)',  # Input: (3,) axis-angle vector, Output: (4,) quaternion
    otypes=[np.float64]
)

def convert_libero_hdft_to_zarr(
    hdf5_path: str,
    sample_ndemo: Optional[int] = None,
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr()
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        num_demos = data.attrs['num_demos']
        if sample_ndemo is None:
            sample_ndemo = num_demos
        sample_ndemo = min(sample_ndemo, num_demos)
        sample_demo_indices = np.random.choice(num_demos, sample_ndemo, replace=False)

        task_name = data.attrs['bddl_file_name'].split('/')[-1][:-5]
        task_prompt = json.loads(data.attrs['problem_info'])['language_instruction']
        _, _, task_uid = task_name_to_suite_and_ids[task_name]

        for demo_idx in tqdm.tqdm(sample_demo_indices, desc=f"Converting Libero Dataset"):
            demo = data[f'demo_{demo_idx}']
            demo_len = len(demo['actions'])

            this_data_collected = {
                'action': demo['actions'][:].astype(np.float32),
                'agentview_rgb': np.flip(demo['obs']['agentview_rgb'][:].astype(np.uint8), axis=1),
                'robot0_eye_in_hand_rgb': np.flip(demo['obs']['eye_in_hand_rgb'][:].astype(np.uint8), axis=1),
                'robot0_joint_pos': demo['obs']['joint_states'][:].astype(np.float32),
                'robot0_eef_pos': demo['obs']['ee_pos'][:].astype(np.float32),
                'robot0_eef_quat': axisangle2quat_vectorized(demo['obs']['ee_ori'][:]).astype(np.float32),
                'robot0_gripper_qpos': demo['obs']['gripper_states'][:].astype(np.float32),
                'prompt': np.array([task_prompt] * demo_len),
                'task_uid': np.array([task_uid] * demo_len)[:, np.newaxis],
            }
            replay_buffer.add_episode(this_data_collected)

    # print
    print('-' * 50)
    print(f"Task: {task_name}\n{replay_buffer}")
    return replay_buffer
