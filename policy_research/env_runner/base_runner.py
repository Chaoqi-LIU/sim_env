from typing import Dict, List
import torch
from policy_research.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy, **kwargs) -> Dict:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()