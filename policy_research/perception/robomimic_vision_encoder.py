from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_nets as rmon
from typing import Dict, Tuple, Union

from policy_research.common.pytorch_util import replace_submodules
from policy_research.perception.base_obs_encoder import BaseObservationEncoder
from policy_research.perception.crop_randomizer import CropRandomizer
from policy_research.model.common.normalizer import LinearNormalizer, _normalize


class RobomimicRgbEncoder(BaseObservationEncoder):
    """
    Assumes rgb input: B,H,W,C
    """
    def __init__(self,
        shape_meta: dict,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        use_group_norm: bool=True,
        eval_fixed_crop: bool = False,
        share_rgb_model: bool=False,
    ):
        super().__init__()

        rgb_ports = list()
        port_shape = dict()
        for key, attr in shape_meta['obs'].items():
            type = attr['type']
            shape = attr['shape']
            if type == 'rgb':
                rgb_ports.append(key)
                port_shape[key] = (shape[2], shape[0], shape[1])  # H,W,C -> C,H,W

        # init global state
        ObsUtils.initialize_obs_modality_mapping_from_dict({"rgb": rgb_ports})

        def crop_randomizer(shape, crop_shape):
            if crop_shape is None:
                return None
            return rmbn.CropRandomizer(
                input_shape=shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
                num_crops=1,
                pos_enc=False,
            )
            
        def visual_net(shape, crop_shape):
            if crop_shape is not None:
                shape = (shape[0], crop_shape[0], crop_shape[1])
            net = rmbn.VisualCore(
                input_shape=shape,
                feature_dimension=64,
                backbone_class='ResNet18Conv',
                backbone_kwargs={
                    'input_channels': shape[0],
                    'input_coord_conv': False,
                },
                pool_class='SpatialSoftmax',
                pool_kwargs={
                    'num_kp': 32,
                    'temperature': 1.0,
                    'noise': 0.0,
                },
                flatten=True,
            )
            return net

        obs_encoder = rmon.ObservationEncoder()
        if share_rgb_model:
            this_shape = port_shape[rgb_ports[0]]
            net = visual_net(this_shape, crop_shape)
            obs_encoder.register_obs_key(
                name=rgb_ports[0],
                shape=this_shape,
                net=net,
                randomizer=crop_randomizer(this_shape, crop_shape),
            )
            for port in rgb_ports[1:]:
                assert port_shape[port] == this_shape
                obs_encoder.register_obs_key(
                    name=port,
                    shape=this_shape,
                    randomizer=crop_randomizer(this_shape, crop_shape),
                    share_net_from=rgb_ports[0],
                )
        else:
            for port in rgb_ports:
                shape = port_shape[port]
                net = visual_net(shape, crop_shape)
                obs_encoder.register_obs_key(
                    name=port,
                    shape=shape,
                    net=net,
                    randomizer=crop_randomizer(shape, crop_shape),
                )

        if use_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features,
                )
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_encoder.make()
        self.encoder = obs_encoder
        self.rgb_keys = list(obs_encoder.obs_shapes.keys())
        self.normalizer = LinearNormalizer()

    def forward(self, obs_dict) -> Dict[str,torch.Tensor]:
        # normalize
        nobs = self._normalize_obs_dict(obs_dict)   # [B, To, H, W, C]
        
        sample = next(iter(nobs.values()))
        B, To, H, W, C = sample.shape
        
        for key in self.rgb_keys:
            # rgb H,W,C -> C,H,W
            nobs[key] = nobs[key].reshape(B*To, H, W, C).permute(0,3,1,2)
        image_feats = self.encoder(nobs)
        image_feats = image_feats.reshape(B, To, -1)
        return image_feats

    @torch.no_grad()
    def output_feature_dim(self) -> int:
        D = self.encoder.output_shape()[0]
        N = len(self.rgb_keys)
        dims = dict(zip(self.rgb_keys, [D // N] * N))
        return sum(dims.values())

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs_dict(self, obs_dict: Dict) -> Dict:
        nobs = dict()
        for port, value in obs_dict.items():
            if port in self.rgb_keys:
                params = self.normalizer.params_dict.get(port, None)
                if params is None:
                    nobs[port] = value
                    print(f"no normalizer params for port {port}, skipping normalization.")
                else:
                    nobs[port] = _normalize(value, params, forward=True)
        return nobs
