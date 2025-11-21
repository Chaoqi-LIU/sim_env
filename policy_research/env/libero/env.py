import os
import string
import numpy as np
import gymnasium
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

from typing import List, Optional, Dict


task_name_to_suite_and_ids = {
    # task_name: (suite_name, task_idx_in_suite, global_task_uid)
}
global_task_id = 0
for suite_name, suite_task_names in libero_task_map.items():
    for local_id, task_name in enumerate(suite_task_names):
        task_name_to_suite_and_ids[task_name] = (suite_name, local_id, global_task_id)
        global_task_id += 1
num_libero_tasks = global_task_id


class LiberoEnv(gymnasium.Env):
    def __init__(self,
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: List = [
            'agentview',
            'robot0_eye_in_hand',
        ],
        state_ports: List = [
            'robot0_joint_pos',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        video_camera: str = 'agentview',
        video_resolution: int = 512,
        max_episode_steps: int = 550,
        enable_render: bool = True,
    ):
        super().__init__()

        libero_suite, task_suite_id, task_uid = task_name_to_suite_and_ids[task_name]
        task = benchmark.get_benchmark_dict()[libero_suite]().get_task(task_suite_id)
        env = ControlEnv(
            bddl_file_name=os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file
            ),
            camera_names=list(set(list(camera_names) + [video_camera])),
            camera_heights=image_size,
            camera_widths=image_size,
            has_renderer=False,
            use_camera_obs=enable_render,
            has_offscreen_renderer=enable_render,
        )
        # env.env.hard_reset = False  # TODO: check if it's safe to set to False
        env.seed(seed)

        self.env = env
        self.task_name = task.name
        self.task_prompt = task.language
        self.task_uid = task_uid
        self.state_ports = state_ports
        self.camera_names = camera_names
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False
        self.cur_step = 0

        # setup gym spaces
        obs_dict = env.env._get_observations()
        observation_space = gymnasium.spaces.Dict({})
        for port in state_ports:
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in camera_names:
            observation_space.spaces[f"{cam_name}_rgb"] = gymnasium.spaces.Box(
                low=0, high=255, 
                shape=(image_size, image_size, 3), dtype=np.uint8
            )
        observation_space.spaces['prompt'] = gymnasium.spaces.Text(
            min_length=0, max_length=512,
            charset=string.printable
        )
        observation_space.spaces['task_uid'] = gymnasium.spaces.Box(
            low=0, high=num_libero_tasks-1,
            shape=(1,), dtype=np.uint8
        )
        self.observation_space = observation_space
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(env.env.action_dim,), dtype=np.float32
        )
        self._let_objects_fall()

    def _let_objects_fall(self):
        # libero env needs a few steps to let objects fall to the table/ground
        dummy_action = [0.] * 6 + [-1.]
        for _ in range(10):
            self.env.step(dummy_action)

    def _extract_obs(self, 
        raw_obs: Optional[Dict[str, np.ndarray]]=None
    ) -> Dict[str, np.ndarray]:
        if raw_obs is None:
            raw_obs = self.env.env._get_observations()

        obs_dict = {}

        # robot state
        for port in self.state_ports:
            obs_dict[port] = raw_obs[port].astype(np.float32)

        # rgb
        for cam_name in self.camera_names:
            obs_dict[f"{cam_name}_rgb"] = np.flip(
                raw_obs[f"{cam_name}_image"], axis=0).astype(np.uint8)
            
        # prompt
        obs_dict['prompt'] = self.task_prompt

        # task id
        obs_dict['task_uid'] = np.array([self.task_uid,], dtype=np.uint8)

        return obs_dict
    
    def step(self, action: np.ndarray):
        obs, reward, terminated, info = self.env.step(action)
        self.cur_step += 1
        if self.env.check_success():
            reward = 1.0
        else:
            reward = 0.0
        self.done = self.done or terminated or (reward >= 1) \
            or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, info
    
    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        obs_dict = self._extract_obs(obs)
        self.done = False
        self.cur_step = 0
        self._let_objects_fall()
        return obs_dict, {'prompt': self.task_prompt}
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        frame = np.flip(self.env.sim.render(
            height=self.video_resolution, width=self.video_resolution, 
            camera_name=self.video_camera
        ), axis=0).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()
