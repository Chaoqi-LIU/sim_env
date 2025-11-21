import wandb
import numpy as np
import torch
import tqdm
import math
import pathlib
import dill
import wandb.sdk.data_types.video as wandb_video

from policy_research.gym_util.multistep_wrapper import MultiStepWrapper
from policy_research.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from policy_research.gym_util.async_vector_env import AsyncVectorEnv
from policy_research.env.robomimic.env import RobomimicEnv
from policy_research.env.robomimic.factory import get_subtasks
from policy_research.env_runner.base_runner import BaseRunner
from policy_research.policy.base_policy import BasePolicy
from policy_research.common.pytorch_util import dict_apply

from typing import Optional, List


class RobomimicRunner(BaseRunner):
    # import multiprocessing as mp
    # mp.set_start_method('spawn', force=True)

    def __init__(self,
        output_dir,
        task_name: str,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 20,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        # other robomimic env args
        image_size: int = 128,
        camera_names: List[str] = [
            "agentview",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            # 'robot0_joint_pos', 
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        max_episode_steps: int = 800,
    ):
        super().__init__(output_dir)

        if n_parallel_envs is None:
            n_parallel_envs = n_test

        assert n_test % n_parallel_envs == 0, "n_test must be divisible by n_parallel_envs"
        assert n_test_vis <= n_test, "n_test_vis must be less than or equal to n_test"

        # get subtasks and distribute across envs
        subtask_names = get_subtasks(task_name)
        env_task_names = [
            subtask_names[i % len(subtask_names)] 
            for i in range(n_test)
        ]
        
        # setup env
        env_seeds = []
        env_fns = []
        env_init_fn_dills = []
        for i in range(n_test):
            this_task_name = env_task_names[i]
            this_seed = test_start_seed + i
            env_seeds.append(this_seed)
            enable_render = i < n_test_vis

            if i < n_parallel_envs:
                def env_fn(task_name=this_task_name, seed=this_seed):
                    return MultiStepWrapper(
                        VideoRecordingWrapper(
                            RobomimicEnv(
                                task_name=task_name,
                                image_size=image_size,
                                seed=seed,
                                camera_names=camera_names,
                                state_ports=state_ports,
                                max_episode_steps=max_episode_steps,
                            ),
                            video_recoder=VideoRecorder.create_h264(
                                fps=fps,
                                codec='h264',
                                input_pix_fmt='rgb24',
                                crf=crf,
                                thread_type='FRAME',
                                thread_count=1
                            ),
                            file_path=None,
                            steps_per_render=1
                        ),
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps,
                        reward_agg_method='max'
                    )
                
                env_fns.append(env_fn)
            
            def init_fn(env, task_name=this_task_name, seed=this_seed, enable_render=enable_render):
                if not env.env.env.task_name == task_name:
                    raise RuntimeError(
                        f"Env task name: {env.env.env.task_name} does not "
                        f"match expected task name: {task_name}."
                    )
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        f'media/{task_name}', wandb_video.util.generate_id() + '.mp4'
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.reset()
            env_init_fn_dills.append(dill.dumps(init_fn))

        assert len(env_fns) == n_parallel_envs
        assert len(env_init_fn_dills) == n_test

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicEnv(
                        task_name=env_task_names[0],
                        image_size=image_size,
                        # seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                        enable_render=False,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=1
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                reward_agg_method='max'
            )

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        
        # attr assignment
        self.task_name = task_name
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_task_names = env_task_names
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    @torch.inference_mode()
    def run(self, 
        policy: BasePolicy,
        # policy inference args
        **kwargs,
    ):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            self.env.call_each(
                'run_dill_function', 
                args_list=[(x,) for x in this_init_fns]
            )

            # start rollout
            obs = self.env.reset()
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in RoboMimic::{self.task_name} {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec
            )
            
            done = False
            while not done and pbar.n < pbar.total:
                # create obs dict
                obs_dict = dict_apply(
                    dict(obs),
                    lambda x: torch.from_numpy(x).to(device=device, dtype=dtype)
                )

                # run policy
                with torch.inference_mode():
                    action = policy.predict_action({
                        port: obs_dict[port] 
                        for port in policy.get_observation_ports()
                    }, **kwargs)['action'].detach().cpu().numpy()

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN of Inf action")
                
                # step env
                obs, reward, done, _ = self.env.step(action) # NOTE: reward=1 if success
                done = np.logical_or(
                    done[this_local_slice],
                    all_success[this_global_slice][this_local_slice]
                )
                done = np.all(done[this_local_slice])

                all_success[this_global_slice] = np.logical_or(
                    all_success[this_global_slice],
                    [r >= 1 for r in reward[this_local_slice]]
                )

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = self.env.render()[this_local_slice]

        # clear out video buffer
        _ = self.env.reset()

        # log
        log_data = dict()

        for task_name in set(self.env_task_names):
            task_success = [
                all_success[i] 
                for i in range(n_inits) 
                if self.env_task_names[i] == task_name
            ]
            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            task_name = self.env_task_names[i]

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                video = wandb.Video(video_path, format='mp4')
                log_data[f"{task_name}/video_{seed}"] = video
            
        # log aggregate metrics
        log_data['mean_success_rate'] = np.mean(all_success)

        return log_data
    
    
    def close(self):
        self.env.close()
