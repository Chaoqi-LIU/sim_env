import torch
import torch.nn as nn
from typing import List, Dict, Optional

from policy_research.perception.base_obs_encoder import BaseObservationEncoder
from policy_research.model.common.normalizer import LinearNormalizer, _normalize


def warning_msg(msg: str):
    return f"ProjectionStateEncoder - warning: {msg}"


class ProjectionStateEncoder(BaseObservationEncoder):
    """
    State: identity encoder
        If out_dim is None, output the state as is.
        If out_dim is specified, project the state to out_dim with a linear layer.
    """
    def __init__(
        self,
        shape_meta: Dict,
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        # parse shape_meta
        state_ports = list()
        state_dim = 0
        for port_name, attr in shape_meta['obs'].items():
            port_type = attr.get('type', 'unsupported')
            if port_type == 'state':
                state_ports.append(port_name)
                port_shape = attr['shape']
                assert len(port_shape) == 1
                state_dim += port_shape[0]
        assert state_ports, "No state port found in shape_meta."

        # projection head
        if out_dim is None:
            self.state_proj = nn.Identity()
            out_dim = state_dim
        else:
            self.state_proj = nn.Linear(state_dim, out_dim)

        self.normalizer = LinearNormalizer()
        self.out_dim = out_dim
        self.shape_meta = shape_meta
        self.state_ports = state_ports

    def modalities(self) -> List[str]:
        return ['state',]

    def output_feature_dim(self) -> int:
        return self.out_dim
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs_dict(self, obs_dict: Dict) -> Dict:
        # normalization
        nobs = dict()
        for port, value in obs_dict.items():
            if port in self.state_ports:
                params = self.normalizer.params_dict.get(port, None)
                if params is None:
                    nobs[port] = value
                    print(warning_msg(f"no normalizer params for port {port}, skipping normalization."))
                else:
                    nobs[port] = _normalize(value, params, forward=True)
        return nobs
                
    def forward(self, obs_dict: Dict) -> torch.Tensor:
        """
        obs_dict: {
            <state_port>: [B, To, Ds] float32 tensor
        }
        Returns:
            [B, To, out_dim]
        """
        # normalization
        nobs = self._normalize_obs_dict(obs_dict)

        # load states
        states = [nobs[port] for port in self.state_ports]

        # encode states
        states = torch.cat(states, dim=-1)      # [B, To, ΣDs]
        state_feat = self.state_proj(states)    # [B, To, out_dim]
        return state_feat
