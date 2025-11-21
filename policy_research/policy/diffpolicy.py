import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Dict, Optional, Tuple, List, Union

from policy_research.policy.base_policy import BasePolicy
from policy_research.perception.base_obs_encoder import BaseObservationEncoder
from policy_research.model.common.normalizer import LinearNormalizer
from policy_research.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from policy_research.model.diffusion.conditional_unet1d import ConditionalUnet1D


class DiffusionTransformerPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        noise_scheduler: Union[DDIMScheduler, DDPMScheduler],
        obs_encoder: BaseObservationEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 768,
        n_layers: int = 8,
        n_heads: int = 12,
        dropout: float = 0.1,
        # inference params
        num_inference_steps: Optional[int] = None,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            obs_type = attr['type']
            if obs_type in modalities:
                obs_ports.append(key)

        # create diffusion transformer
        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities
    
    def get_observation_ports(self):
        return self.obs_ports
    
    def get_policy_name(self):
        base_name = 'dp_trans_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )
    
    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)
        
    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            lr=policy_lr,
            weight_decay=weight_decay
        )
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "lr": obs_enc_lr,
            "weight_decay": weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, betas=betas
        )
        return optimizer
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]

        # diffusion sampling
        scheduler = self.noise_scheduler
        model = self.model
        trajectory = torch.randn(
            size=(len(features), self.horizon, self.action_dim),
            dtype=features.dtype,
            device=features.device,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory = scheduler.step(
                model(trajectory, t, features),
                t, trajectory, 
            ).prev_sample

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(trajectory[...,:self.action_dim])

        # receding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    def forward(self, batch):
        # normalize action
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]
        assert features.shape[:2] == (batch_size, self.n_obs_steps)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # predict noise residual
        pred = self.model(noisy_trajectory, timesteps, features)
        loss = F.mse_loss(pred, noise)
        return loss



class DiffusionUnetPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        noise_scheduler: Union[DDIMScheduler, DDPMScheduler],
        obs_encoder: BaseObservationEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # inference params
        num_inference_steps: Optional[int] = None,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            obs_type = attr['type']
            if obs_type in modalities:
                obs_ports.append(key)

        # create diffusion unet
        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self) -> BaseObservationEncoder:
        return self.obs_encoder
    
    def get_observation_modalities(self) -> List[str]:
        return self.modalities
    
    def get_observation_ports(self) -> List[str]:
        return self.obs_ports

    def get_policy_name(self) -> str:
        base_name = 'dp_unet_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]
    
    def create_dummy_observation(self, 
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )
    
    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = [
            {
                "params": self.model.parameters(),
                "lr": policy_lr,
                "weight_decay": weight_decay
            },
            {
                "params": self.obs_encoder.parameters(),
                "lr": obs_enc_lr,
                "weight_decay": weight_decay
            }
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, betas=betas
        )
        return optimizer
    
    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]
        features = features.reshape(features.shape[0], -1)  # [B, To*d]

        # diffusion sampling
        scheduler = self.noise_scheduler
        model = self.model
        trajectory = torch.randn(
            size=(len(features), self.horizon, self.action_dim),
            dtype=features.dtype,
            device=features.device,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory = scheduler.step(
                model(trajectory, t, global_cond=features),
                t, trajectory, 
            ).prev_sample

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(trajectory[...,:self.action_dim])

        # receding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    def forward(self, batch):
        # normalize action
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        # encode observation
        features = self.obs_encoder(batch['obs'])  # [B, To, d]
        assert features.shape[:2] == (batch_size, self.n_obs_steps)
        features = features.reshape(batch_size, -1)  # [B, To*d]

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # predict noise residual
        pred = self.model(noisy_trajectory, timesteps, global_cond=features)
        loss = F.mse_loss(pred, noise)
        return loss
