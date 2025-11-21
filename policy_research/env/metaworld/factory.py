import numpy as np
from policy_research.env.metaworld.env import MetaworldEnv
from typing import Optional, List


MT_TASKS = {
    'mt4': [
        # medium series
        'box-close',
        'coffee-pull',
        # very hard series
        'disassemble',
        'stick-pull',
    ],
    'mt10': [
        # medium series
        'hammer',
        'peg-insert-side',
        'push-wall',
        'box-close',
        'coffee-pull',
        'coffee-push',
        # very hard series
        'shelf-place',
        'disassemble',
        'stick-pull',
        'stick-push',
    ],
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]


def get_metaworld_env(
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
    max_episode_steps: int = 200,
) -> List[MetaworldEnv]:
    
    return [
        MetaworldEnv(
            task_name=task,
            device=device,
            seed=seed,
            camera_names=camera_names,
            oracle=oracle,
            video_camera=video_camera,
            max_episode_steps=max_episode_steps
        ) for task in get_subtasks(task_name)
    ]


def take_a_glance(task_names: List[str], 
    camera_name: Optional[str]) -> List[np.ndarray]:
    obs = []
    for task_name in task_names:
        env = get_metaworld_env(
            task_name=task_name,
            image_size=512,
            camera_names=[camera_name],
        )[0]
        if camera_name is None:
            obs.append(env.render())
        else:
            obs.append(env.reset()[camera_name + '_rgb'])
        env.close()
    if camera_name == 'topview' or camera_name == 'behindGripper':
        obs = [np.flipud(o) for o in obs]
    return obs
