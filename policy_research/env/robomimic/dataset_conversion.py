import h5py
import json
import numpy as np
import tqdm
from typing import Optional, List

from policy_research.common.replay_buffer import ReplayBuffer
from policy_research.env.robomimic.env import RobomimicEnv


def convert_robomimic_hdf5_to_zarr(
    hdf5_path: str,
    sample_ndemo: Optional[int] = None,
    desired_image_size: int = 128,
    desired_camera_names: List[str] = [
        'agentview',
        'robot0_eye_in_hand',
    ],
    desired_state_ports: List[str] = [
        'robot0_joint_pos',
        'robot0_eef_pos',
        'robot0_eef_quat',
        'robot0_gripper_qpos',
    ],
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr()
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        env_meta = json.loads(f["data"].attrs["env_args"])
        task_name = env_meta['env_name']

        env = RobomimicEnv(     # env for replay dataset
            task_name=task_name,
            image_size=desired_image_size,
            camera_names=desired_camera_names,
            state_ports=desired_state_ports,
        )
        env.reset()

        num_demo = len(data)
        if sample_ndemo is None:
            sample_ndemo = num_demo
        sample_ndemo = min(sample_ndemo, num_demo)
        sample_demo_indices = np.random.choice(num_demo, sample_ndemo, replace=False)
        for demo_idx in tqdm.tqdm(sample_demo_indices, desc=f"Converting {task_name}"):
            demo = data[f'demo_{demo_idx}']
            
            this_data_collected = {
                # 'states': demo['states'][:].astype(np.float32),
                'action': demo['actions'][:].astype(np.float32),
                **{f"{cam}_rgb": [] for cam in desired_camera_names},
                **{state: [] for state in desired_state_ports}
            }

            # replay
            for state in demo['states'][:]:
                env.env.sim.set_state_from_flattened(state)
                env.env.sim.forward()
                obs = env._extract_obs()
                for cam in desired_camera_names:
                    this_data_collected[f"{cam}_rgb"].append(obs[f"{cam}_rgb"])
                for state_port in desired_state_ports:
                    this_data_collected[state_port].append(obs[state_port])

            # to numpy arrays
            for cam in desired_camera_names:
                this_data_collected[f"{cam}_rgb"] = np.array(this_data_collected[f"{cam}_rgb"], dtype=np.uint8)
            for state in desired_state_ports:
                this_data_collected[state] = np.array(this_data_collected[state], dtype=np.float32)
        
            replay_buffer.add_episode(this_data_collected)

        env.close()

    # print
    print('-' * 50)
    print(f"Task: {task_name}\n{replay_buffer}")

    return replay_buffer
