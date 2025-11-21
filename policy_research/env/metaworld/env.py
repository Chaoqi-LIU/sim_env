import gym
import gym.spaces
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=
    ".*Box bound precision lowered by casting to float32")
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from typing import Optional, Tuple, Dict, Union, List



class MetaworldEnv(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"], 
        "video.frames_per_second": 10,
    }

    def __init__(self,
        task_name: str,
        device: str = 'cuda:0',
        image_size: int = 128,
        seed: Optional[int] = None,
        camera_names: List[str] = [
            'topview', 'corner', 'corner2',
            'corner3', 'behindGripper', 'gripperPOV'
        ],
        oracle: bool = False,
        video_camera: str = 'corner2',
        video_resolution: int = 512,
        max_episode_steps: int = 200,
    ):
        super().__init__()
        
        # env seeding and instantiation
        # from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        self.env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
            f"{task_name}-v2-goal-observable"](seed=seed)
        if seed is not None:
            self.env.seed(seed)
        self.env._freeze_rand_vec = not oracle

        # init env state
        self.env.reset()
        env_init_state = self.env.get_env_state()

        # adjust camera near-far
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5

        # NOTE: hack corner2 camera setup, https://arxiv.org/abs/2212.05698
        cam_id = self.env.sim.model.camera_name2id('corner2')
        assert cam_id == 2      # human knowledge
        self.env.sim.model.cam_pos0[cam_id] = [0.6, 0.295, 0.8]
        self.env.sim.model.cam_pos[cam_id]  = [0.6, 0.295, 0.8]

        # attr
        self.env_init_state = env_init_state
        self.camera_names = camera_names
        self.action_space = self.env.action_space
        self.episode_length = self._max_episode_steps = max_episode_steps
        self.oracle = oracle
        self.image_size = image_size
        self.video_resolution = video_resolution
        self.task_name = task_name
        self.video_camera = video_camera
        self.gpu_id = int(device.split(':')[-1])
        
        # setup gym observation space
        observation_space = gym.spaces.Dict()
        for name in camera_names:
            observation_space[f"{name}_rgb"] = gym.spaces.Box(
                low=0, high=255,
                shape=(image_size, image_size, 3),
                dtype=np.uint8
            )
        observation_space["agent_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=np.concatenate([
                self.env.get_endeff_pos(),
                self.env._get_site_pos('leftEndEffector'),
                self.env._get_site_pos('rightEndEffector')
            ]).shape,
            dtype=np.float32
        )
        if oracle:
            observation_space['full_state'] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.env.observation_space.shape,
                dtype=np.float32
            )
        self.observation_space = observation_space

    def _get_rgb(self, 
        cam_name: Optional[List[str]] = None,
        image_size: Optional[int] = None,
    ) -> Union[
        Dict[str, np.ndarray], 
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ]:
        if cam_name is None:
            cam_name = self.camera_names
        if image_size is None:
            image_size = self.image_size

        return {
            cam: self.env.sim.render(
                width=image_size, height=image_size,
                camera_name=cam, depth=False, 
                device_id=self.gpu_id
            ) for cam in cam_name
        }

    def _extract_obs(self) -> Dict[str, np.ndarray]:
        obs_dict = {}

        # robot state
        obs_dict['agent_pos'] = np.concatenate([
            self.env.get_endeff_pos(),
            self.env._get_site_pos('leftEndEffector'),
            self.env._get_site_pos('rightEndEffector')
        ])

        # rgb images
        rgbs = self._get_rgb()
        for cam_name in self.camera_names:
            obs_dict[f"{cam_name}_rgb"] = rgbs[cam_name]

        return obs_dict
    
    def step(self, action: np.ndarray):
        full_state, reward, done, info = self.env.step(action)
        self.cur_step += 1
        obs_dict = self._extract_obs()
        if self.oracle:
            obs_dict['full_state'] = full_state
        if info['success']:
            self.succ_step += 1
        done = done or (self.cur_step >= self.episode_length) or (self.succ_step >= 10)
        return obs_dict, reward, done, info
    
    def reset(self) -> Dict[str, np.ndarray]:
        self.env.reset()
        self.env.reset_model()
        full_state = self.env.reset()
        self.env.set_env_state(self.env_init_state)
        self.cur_step = 0
        self.succ_step = 0
        obs_dict = self._extract_obs()
        if self.oracle:
            obs_dict['full_state'] = full_state
        return obs_dict

    def seed(self, seed=None):
        self.env.seed(seed)
        self.env_init_state = self._env_init_state_of_seed(seed)

    def render(self, mode='rgb_array'):
        # NOTE: only for video recording wrapper
        assert mode == 'rgb_array'
        return self._get_rgb([self.video_camera], 
            image_size=self.video_resolution)[self.video_camera]
    
    def close(self):
        self.env.close()
    
    def _env_init_state_of_seed(self, seed: int):
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
            f"{self.task_name}-v2-goal-observable"](seed=seed)
        env.reset()
        init_state = env.get_env_state()
        env.close()
        return init_state
    