from policy_research.model.common.module_attr_mixin import ModuleAttrMixin
from policy_research.model.common.normalizer import LinearNormalizer
from typing import List, Dict, Union


class BaseObservationEncoder(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    def forward(self, obs_dict: Union[Dict, List[Dict]]) -> Dict:
        raise NotImplementedError

    def modalities(self) -> List[str]:
        raise NotImplementedError
    
    def output_feature_dim(self) -> int:
        raise NotImplementedError
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError
