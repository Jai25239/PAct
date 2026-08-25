"""
Flow Matching Trainer Implementation

This file implements trainers for diffusion models using the flow matching objective.
Flow matching is a generative modeling technique that defines a continuous path between
a noise distribution and the data distribution, and trains models to learn the velocity field 
of this path. Unlike traditional diffusion models that learn to denoise, flow matching directly
learns the vector field that transforms noise to data samples.

The file contains several trainer classes:
1. FlowMatchingTrainer: Base trainer for flow matching
2. FlowMatchingCFGTrainer: Adds classifier-free guidance support
3. TextConditionedFlowMatchingCFGTrainer: Supports text conditioning with CFG
4. ImageConditionedFlowMatchingCFGTrainer: Supports image conditioning with CFG
"""
'''LQM: no significant changes'''
import os
from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import torch.distributed as dist
from torchvision import utils
from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce, save_image_with_notes
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin
import imageio
from ...utils.data_utils import cycle, BalancedResumableSampler
class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        **kwargs
    ):
        # Initialize the parent class with all args and kwargs
        super().__init__(*args, **kwargs)
        # Store time schedule configuration for sampling timesteps
        self.t_schedule = t_schedule
        # Store minimum sigma value to prevent numerical instability
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        # Generate random noise if none is provided
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        # Reshape t to broadcast correctly across spatial dimensions
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        # Interpolate between clean data and noise based on timestep
        # Apply minimum sigma to ensure stability at t=1
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        
        Args:
            x_t: The noisy tensor at timestep t.
            t: The timestep values [0-1].
            noise: The noise component to remove.
            
        Returns:
            x_0: The recovered clean data.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        # Reshape t for proper broadcasting
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        # Invert the diffusion process to recover x_0
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        
        Args:
            x_0: Clean data input.
            noise: Noise input.
            t: Timestep values.
            
        Returns:
            v: Velocity vector field at time t.
        """
        # The velocity is the time derivative of the path from data to noise
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        
        Args:
            cond: Conditioning inputs.
            kwargs: Additional arguments.
            
        Returns:
            Processed conditioning data.
        """
        # print(f"debugging get_cond shape {cond.shape}")
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        
        Args:
            cond: Conditioning inputs.
            kwargs: Additional arguments for inference.
            
        Returns:
            Dictionary with conditioning data and additional arguments.
        """
        # print("debugging get_inference_cond")
        # print(cond)
        # print(kwargs)
        
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        
        Args:
            kwargs: Additional arguments for sampler.
            
        Returns:
            A flow-based sampler for generating samples.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        
        Args:
            kwargs: Arguments containing conditioning data.
            
        Returns:
            Dictionary with visualization data (empty by default).
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps based on the configured time schedule.
        
        Args:
            batch_size: Number of timesteps to sample.
            
        Returns:
            Tensor of timestep values in range [0, 1].
        """
        if self.t_schedule['name'] == 'uniform':
            # Uniform sampling between 0 and 1
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            # Logit-normal distribution for timestep sampling
            # Gives more samples near 0 and 1 than uniform
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            A dict with the key "loss" containing a tensor of shape [N],
            may also contain other keys for different terms.
        """
        # Generate random noise for diffusion
        noise = torch.randn_like(x_0)
        # Sample random timesteps for training
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        # Diffuse the clean data to timestep t
        x_t = self.diffuse(x_0, t, noise=noise)
        # print("cond shape ", cond.shape)
        # Process conditioning information
        cond, ordered_mask_dino = self.get_cond(cond, **kwargs)
        kwargs['ordered_mask_dino'] = ordered_mask_dino
        # print(f"FlowMatchingTrainer cond: {cond.shape}")
        
        # Get model's prediction of the velocity field
        # Multiply t by 1000 to match model's expected timestep scale
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        
        # Compute the target velocity vector field
        target = self.get_v(x_0, noise, t)
        
        # Calculate loss terms
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]  # Total loss is just MSE for basic flow matching

        # Calculate per-time-bin losses for analysis
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        # Divide time range [0,1] into 10 bins and log loss per bin
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Generate samples for visualization during training.
        
        Args:
            num_samples: Total number of samples to generate.
            batch_size: Batch size for generation.
            verbose: Whether to print progress information.
            
        Returns:
            Dictionary with generated samples and ground truth data.
        """
        # Create a dataloader to get ground truth samples
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # Get the sampler for inference
        sampler = self.get_sampler()
        sample_gt = []  # Ground truth samples
        sample = []     # Generated samples
        cond_vis = []   # Visualization of conditioning
        
        # Generate samples in batches
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            # print("data is instance")

            #   Move data to GPU and slice to current batch size
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}

            # print("data is instance2")
            # Generate random noise for sampling start point
            noise = torch.randn_like(data['x_0'])
            # Store ground truth data
            sample_gt.append(data['x_0'])
            # Prepare conditioning visualization
            cond_vis.append(self.vis_cond(**data))

            # print("data is instance3")

            # Remove ground truth from conditioning data
            del data['x_0']

            # print("data['cond'] shape ", data['cond'].shape) # torch.Size([4, 3, 3, 518, 518])
            # print("data is instance4.5")
            # for k, v in data.items():
            #     print(f"data {k} value shape {len(v)}")
            #     for i in v:
            #         print(f"{i.shape}")
            # data cond value shape 4
            # torch.Size([3, 3, 518, 518])
            # torch.Size([3, 3, 518, 518])
            # torch.Size([3, 3, 518, 518])
            # torch.Size([3, 3, 518, 518])
            # print(**data)
            # Get conditioning for inference
            args = self.get_inference_cond(**data)
            # args['cond']
            # print("args['cond'] shape ", args['cond'].shape) # torch.Size([4, 4122, 1024])
            # print("data is instance4")
            # Run the sampler to generate samples
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50,           # Number of sampling steps
                cfg_strength=3.0,   # Classifier-free guidance strength
                verbose=verbose,    # Whether to display progress
            ) ### LQM_check_this
            # print("data is instance5")
            # Store generated samples
            sample.append(res.samples)

        # Concatenate batches of samples
        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)

        # print("sample shape ", sample.shape)
        
        # Prepare results dictionary
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        # print("cond vis ", cond_vis)
        # raise NotImplementedError("Debugging sample_dict")
        # Add conditioning visualizations
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        # print("run snapshot done")
        # Debug function to print shapes of all values in nested dictionaries

        # Print all shapes in the sample_dict
        # print("Sample dictionary structure and shapes:")
        # sample_gt: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # sample: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # image: 
        # tensor with shape torch.Size([64, 3, 3, 518, 518])
        # image

        # Sample dictionary structure and shapes:
        # sample_gt: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # sample: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # image: 
        # image
        # tensor with shape torch.Size([64, 3, 518, 518])
        # print("Sample dictionary structure and shapes:")
        # for key, value in sample_dict.items():
        #     print(f"{key}: ")
        #     for k, v in value.items():
        #         # print(f"  {k.shape}: {v}")
        #         if k == 'type':
        #             print(v)
        #         elif k == 'value':
        #             print(f"tensor with shape {v.shape}")

        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass

from torch.nn.parallel import DistributedDataParallel as DDP
from modules.pact.trainers.utils import make_master_params
import modules.pact.utils.grad_clip_utils as grad_clip_utils
class PartBasedImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        # Initialize the parent class with all args and kwargs
        self.is_predict_arti_info = kwargs.get('is_predict_arti_info', False)
        self.is_train_Artihead_only = kwargs.get('is_train_Artihead_only', False)
        self.arti_info_diffusion = kwargs.get('arti_info_diffusion', False)
        self.arti_out_mode = kwargs.get('arti_out_mode', 'regression_last_step') ### 'regression_last_step', 'regression_mean_steps', 'diffusion'
        self.masked_articulation_loss = kwargs.get('masked_articulation_loss', 'False') ### 'regression_last_step', 'regression_mean_steps', 'diffusion'
        print("Masked articulation loss:", self.masked_articulation_loss)
    
        super().__init__(*args, **kwargs)
        

    
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        
        This method performs the following tasks:
        1. Sets up DDP for distributed training if world_size > 1
        2. Collects model parameters for optimization
        3. Sets up mixed precision training based on fp16_mode
        4. Initializes EMA parameters if needed
        5. Sets up the optimizer and learning rate scheduler
        6. Configures elastic memory management if enabled
        7. Sets up gradient clipping if configured
        """

        ##### LQM  LoRA##### 
        ##### LQM  LoRA##### 
        from peft import LoraConfig, TaskType, get_peft_model
        lora_setting = kwargs.get('lora_setting', None)
        if lora_setting and lora_setting["use_lora"]:
            self.finetune_from(self.finetune_ckpt) ### load base model
            self.finetune_ckpt = None ### avoid loading twice , key 会变，避免报错            
            print("Using Lora")
            peft_config = LoraConfig(  **lora_setting["args"],)
            name = lora_setting["name"]
            peft_config.save_pretrained(os.path.join(self.output_dir, "lora_config"))
            model = get_peft_model(self.models[name], peft_config, adapter_name=name)
            model.print_trainable_parameters()
            self.models[name] = model ### hard_coded
        ##### LQM  LoRA##### 
        ##### LQM  LoRA##### 
        
        ##### LQM  Arti-Model ##### 
        ##### LQM  Arti-Model ##### 
        if self.is_predict_arti_info:
            if self.arti_out_mode == 'diffusion_head_feature_cache':
                pass
                self.models["denoiser"] = self.models["Articulation"]
                del self.models["Articulation"]
                self.finetune_ckpt = None ### avoid loading twice , key 会变，避免报错   
            else:
                if lora_setting is None or (lora_setting and not lora_setting["use_lora"]):
                    self.finetune_from(self.finetune_ckpt,map_to_master_param=False)
                    self.finetune_ckpt = None ### avoid loading twice , key 会变，避免报错   
                if self.is_train_Artihead_only:
                    print("Training articulation head only.")
                    for param in self.models["denoiser"].parameters():
                        param.requires_grad = False
                # if self.arti_info_diffusion:
                    # self.models["denoiser"] = self.init_articulation_model(self.models["denoiser"], self.models["Articulation"])
                self.models["Articulation"].set_base_model(self.models["denoiser"])
                
                self.models["denoiser"] = self.models["Articulation"]
                del self.models["Articulation"]
        ##### LQM  Arti-Model #####         
        ##### LQM  Arti-Model #####      
        
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()], [])
        if self.fp16_mode == 'amp':
            # For AMP mode, we use PyTorch's automatic mixed precision
            self.master_params = self.model_params
            # self.scaler = torch.GradScaler() if self.fp16_mode == 'amp' else None
            self.scaler = torch.cuda.amp.GradScaler() if self.fp16_mode == 'amp' else None
        elif self.fp16_mode == 'inflat_all':
            # Manual FP16 mode with master params in FP32
            self.master_params = make_master_params(self.model_params)
            self.fp16_scale_growth = self.fp16_scale_growth
            self.log_scale = 20.0
        elif self.fp16_mode is None:
            # Standard FP32 training
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'FP16 mode {self.fp16_mode} is not implemented.')

        # Build EMA params - only the master process maintains EMA parameters
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]



        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            if self.optimizer_config.get('use_8bit_adam', False) and self.optimizer_config['name'] == 'AdamW':
                if self.is_master:
                    print("Using 8-bit AdamW optimizer from bitsandbytes")
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                    )
                optimizer_cls = bnb.optim.AdamW8bit
                self.optimizer = optimizer_cls(self.master_params, **self.optimizer_config['args'])
            else:
                self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            raise NotImplementedError("Elastic memory controller is not implemented yet.")
            # self.elastic_controller = ElasticMemoryController(

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])
        if lora_setting and lora_setting["use_lora"]:
            # self.check_ddp()
            pass
            # print("Using Lora, skip initializing master params")
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        sample_steps: int = 50,
        shuffle: bool = True,
        cfg_scale: float = 3.0,
    ) -> Dict:
        """
        Generate samples for visualization during training.
        
        Args:
            num_samples: Total number of samples to generate.
            batch_size: Batch size for generation.
            verbose: Whether to print progress information.
            
        Returns:
            Dictionary with generated samples and ground truth data.
        """
        # Create a dataloader to get ground truth samples
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # Get the sampler for inference
        sampler = self.get_sampler()
        sample_gt = []  # Ground truth samples
        sample = []     # Generated samples
        cond_vis = []   # Visualization of conditioning
        cond_vis_mask = []  # Visualization of conditioning mask
        num_part_list = []
        sample_gt_arti = []
        sample_arti = []
        data_iterator =iter(dataloader)
        # Generate samples in batches
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(data_iterator)
            # print("data is instance")

            #   Move data to GPU and slice to current batch size
            data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()} ### LQM: changed
            num_part_list.append(data['num_parts'])
            # print("data is instance2")
            # Generate random noise for sampling start point
            noise = torch.randn_like(data['x_0'])
            # Store ground truth data
            sample_gt.append(data['x_0'])
            # Prepare conditioning visualization
            cond_vis.append(self.vis_cond(**data))
            cond_vis_mask.append(self.vis_cond_mask(data['img_mask_vis']))
            # print("data is instance3")

            # Remove ground truth from conditioning data
            del data['x_0']


            args = self.get_inference_cond(**data)
            args["return_articulation"]= True if hasattr(self, 'is_predict_arti_info') and self.is_predict_arti_info else False ### LQM: code Ominipart
            # args['cond']
            # print("args['cond'] shape ", args['cond'].shape) # torch.Size([4, 4122, 1024])
            # print("data is instance4")
            # Run the sampler to generate samples
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=sample_steps,           # Number of sampling steps
                cfg_strength=cfg_scale,   # Classifier-free guidance strength
                verbose=verbose,    # Whether to display progress
            ) ### LQM_check_this
            # print("data is instance5")
            # Store generated samples
            sample.append(res.samples)
            if hasattr(self, 'is_predict_arti_info') and self.is_predict_arti_info:
                sample_gt_arti.append(data['x_0_arti'])
                if self.arti_out_mode == 'regression_last_step':
                    sample_arti.append(res.arti)
                elif self.arti_out_mode == 'regression_mean_steps':
                    arti_mean_num = self.arti_mean_num if hasattr(self,"arti_mean_num") else 10
                    feats = [pre for pre in res.pred_arti_t[-arti_mean_num:]]
                    feats = torch.mean(torch.stack(feats), dim=0)
                    art_mean = sp.SparseTensor(coords= res.arti.coords,feats=feats)
                    assert art_mean.shape == res.arti.shape
                    sample_arti.append(art_mean)
                elif self.arti_out_mode == 'diffusion':
                    sample_arti.append(res.arti) ## check this
                else:
                    raise ValueError(f"Unknown arti_out_mode: {self.arti_out_mode}")
            
            

        # Concatenate batches of samples
        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)

        # print("sample shape ", sample.shape)
        
        # Prepare results dictionary
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        
        if hasattr(self, 'is_predict_arti_info') and self.is_predict_arti_info:
            sample_gt_arti = sp.sparse_cat(sample_gt_arti)
            sample_arti = sp.sparse_cat(sample_arti)
            # sample_arti = sp.sparse_cat(sample_arti)
            
            # for idx, (art_gt, arti) in enumerate(zip(sample_gt_arti, sample_arti)):
            
            sample_dict['sample_gt_arti'] = {'value': sample_gt_arti, 'type': 'articulation'}
            sample_dict['sample_arti'] = {'value': sample_arti, 'type':"articulation"}
            
        
        # print("cond vis ", cond_vis)
        # raise NotImplementedError("Debugging sample_dict")
        # Add conditioning visualizations
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        

        sample_dict.update(dict_reduce(cond_vis_mask, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        sample_dict['num_parts'] = {'value': torch.cat(num_part_list, dim=0), 'type': 'layout'}

        return sample_dict
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=10, batch_size=4, verbose=False, 
                 sample_steps=50, shuffle=True,cfg_scale: float = 3.0):
        """
        Sample images from the model and save to disk.
        
        This function coordinates the generation of samples across all processes
        and gathers them on the master process for saving.
        
        Args:
            suffix: Suffix for the output directory name
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            verbose: Whether to print verbose information
            
        Note: This function should be called by all processes in distributed training.
        """
        # if self.world_size > 1:
        #     dist.barrier()
        # Print status message from the master process only
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        # Set default suffix to current step if none provided (used for organizing output files)
        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Calculate how many samples each process should generate in distributed setting
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        # # Generate samples using the model's snapshot implementation

        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose,
                                    sample_steps = sample_steps,
                                    shuffle= shuffle,
                                    cfg_scale=cfg_scale
                                    )

        def save_video_grid(video_list: List[np.ndarray], output_dir: str, filename: str, fps: int = 20) -> None:
            if not video_list:
                return
            os.makedirs(output_dir, exist_ok=True)
            num_videos = len(video_list)
            grid_size = int(np.ceil(np.sqrt(num_videos)))
            frame_counts = [len(video) for video in video_list]
            max_frames = max(frame_counts)
            first_frame = video_list[0][0]
            frame_shape = first_frame.shape
            height, width = frame_shape[:2]
            channels = 1 if len(frame_shape) == 2 else frame_shape[2]
            dtype = first_frame.dtype
            grid_frames = []
            blank_frame = np.zeros(frame_shape, dtype=dtype)
            for frame_idx in range(max_frames):
                if channels == 1:
                    grid_frame = np.zeros((grid_size * height, grid_size * width), dtype=dtype)
                else:
                    grid_frame = np.zeros((grid_size * height, grid_size * width, channels), dtype=dtype)
                for video_idx, video in enumerate(video_list):
                    row = video_idx // grid_size
                    col = video_idx % grid_size
                    top = row * height
                    left = col * width
                    if frame_idx < video.shape[0]:
                        frame = video[frame_idx]
                    else:
                        frame = blank_frame
                    if channels == 1:
                        grid_frame[top:top + height, left:left + width] = frame
                    else:
                        grid_frame[top:top + height, left:left + width, :] = frame
                grid_frames.append(grid_frame)
            grid_path = os.path.join(output_dir, filename)
            imageio.mimsave(grid_path, grid_frames, fps=fps)

        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                # Convert raw samples to visualizable format using the dataset's visualization method
                if self.model_version=="Arti_stage2_flow":
                    vis, videos = self.visualize_sample({'x_0': samples[key]['value'], 'part_layouts': samples['layout']['value']})
                    
                else:
                    videos = None
                    vis,videos = self.visualize_sample({'x_0': samples[key]['value'], 'num_parts': samples['num_parts']['value']})
                if isinstance(vis, dict):
                    # If visualization returns a dictionary, create multiple entries with different visualizations
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    # Remove the original entry since it's been replaced with specific visualizations
                    del samples[key]
                else:
                    # Otherwise, update the existing entry by replacing with visualization
                    samples[key] = {'value': vis, 'type': 'image'}
                if videos is not None and self.is_master:
                    is_save_video_grid = True
                    grid_size_max = 4
                    output_dir = os.path.join(self.output_dir, 'samples', suffix, f'videos_{key}')
                    print(f"Saving videos to {output_dir}...")
                    if is_save_video_grid:
                        for idx in range(0, len(videos), grid_size_max * grid_size_max):
                            sub_videos = videos[idx:idx + grid_size_max * grid_size_max]
                            if not sub_videos:
                                continue
                            save_video_grid(
                                [np.stack(video) if isinstance(video, (list, tuple)) else video for video in sub_videos],
                                output_dir,
                                f'{key}_grid_sub{idx // (grid_size_max * grid_size_max)}.mp4',
                                fps=20,
                            )
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        for i, video in enumerate(videos):
                            gaussian_video_path = f"{output_dir}/object_{i}_gs.mp4"
                            imageio.mimsave(gaussian_video_path, video, fps=20)
        # Remove the layout entry after processing
        if 'layout' in samples:
            del samples['layout']
        # Gather samples from all processes in distributed training setup
        if self.world_size > 1:
            dist.barrier()
            for key in samples.keys():
                # if isinstance(samples[key]['value'], list):
                #     continue
                if key in ['sample', 'sample_gt', 'image']:
                    # Ensure tensor is contiguous in memory for efficient gathering operation
                    samples[key]['value'] = samples[key]['value'].contiguous()
                    # print(samples[key]['value'].shape)
                    if self.is_master:
                        # Create buffers on master process to receive data from all processes
                        all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                    else:
                        # Non-master processes don't need to allocate receive buffers
                        all_images = []
                    # Gather data from all processes to the master (rank 0)
                    dist.gather(samples[key]['value'], all_images, dst=0)
                    if self.is_master:
                        # Concatenate all gathered samples and limit to requested number
                        samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]
            dist.barrier()
        # for key in samples.keys():
        #     print(key)
        # exit(0)
            
        # if self.world_size > 1:
        #     for key in samples.keys():
        #         if not isinstance(samples[key]['value'], list):
        #             continue

        #         print(samples[key]['value'])
        #         # if self.is_master:
        #         #     gathered = [None for _ in range(self.world_size)]
        #         # else:
        #         #     gathered = None
        #         gathered = [None for _ in range(self.world_size)]
        #         # Gather list objects to rank 0
        #         print(1111111111111111111111)
        #         dist.all_gather_object(gathered, samples[key]['value'])
        #         print(222222222222222222222)
        #         if self.is_master:
        #             print(333333333333333333333)
        #             samples[key]['value'] = sum(gathered, [])[:num_samples]


        # Save images to disk (only on master process)
        if self.is_master:
            # Create output directory for current snapshot
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys(): # Error: The size of tensor a (3) must match the size of tensor b (518) at non-singleton dimension 2
                # print(key)
                # print(samples[key])
                if samples[key]['type'] == 'image':
                    # print(f"Saving {key} images...")
                    # print(f"shape is {samples[key]['value'].shape}") # shape is torch.Size([64, 3, 3, 518, 518])
                    # Reshape [64, 3, 3, 518, 518] -> [64, 9, 518, 518]
                    if samples[key]['value'].ndim == 5:
                        value = samples[key]['value']
                        value = value.permute(1, 0, 2, 3, 4)
                        # print(f"Reshaped tensor from {value.shape} to {samples[key]['value'].shape}")
                        for indexx in range(value.shape[0]):
                            # Save image samples using torchvision utilities
                            utils.save_image(
                                value[indexx],
                                os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}_{indexx}.jpg'),
                                nrow=int(np.sqrt(num_samples)),  # Arrange images in a square grid
                                normalize=True,
                                value_range=self.dataset.value_range,  # Use dataset's specified value range for normalization
                            )
                    else:
                        image = samples[key]['value']
                        # Maximum number of images to save in one file
                        max_images_per_file = 100
                        
                        # Get the number of images in this batch
                        num_images = image.size(0)
                        
                        if num_images <= max_images_per_file:
                            # If fewer than max_images_per_file, save them all in one file
                            utils.save_image(
                                image,
                                os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                                nrow=int(np.sqrt(num_images)),
                                normalize=True,
                                value_range=self.dataset.value_range,
                            )
                        else:
                            # If more than max_images_per_file, split into multiple files
                            num_batches = (num_images + max_images_per_file - 1) // max_images_per_file
                            
                            for i in range(num_batches):
                                start_idx = i * max_images_per_file
                                end_idx = min((i + 1) * max_images_per_file, num_images)
                                
                                # Extract the current batch of images
                                batch_images = image[start_idx:end_idx]
                                batch_size_actual = batch_images.size(0)
                                
                                utils.save_image(
                                    batch_images,
                                    os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}_{i+1}.jpg'),
                                    nrow=int(np.sqrt(batch_size_actual)),
                                    normalize=True,
                                    value_range=self.dataset.value_range,
                                )
                    # print(f"key is ****{key}")
                elif samples[key]['type'] == 'number':
                    # Process and save numerical samples as images with annotations
                    min_val = samples[key]['value'].min()
                    max_val = samples[key]['value'].max()
                    # Normalize values to [0, 1] range for visualization
                    images = (samples[key]['value'] - min_val) / (max_val - min_val)
                    # Create a grid of images
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,  # Already normalized above
                    )
                    # Save the image with min/max annotations for reference
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min_val}, max: {max_val}',
                    )

        # Print completion message (master process only)
        if self.is_master:
            print(' Done.')
        
        # print(1111111111111111111111)
        # if self.world_size > 1:
        #     dist.barrier() 
        #     print(222222222222222222222)    
        
    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.
        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            A dict with the key "loss" containing a tensor of shape [N],
            may also contain other keys for different terms.
        """
        arti_dim = 24 ### LQM: TODO: hard-coded
        # Generate random noise for diffusion
        noise = torch.randn_like(x_0)
        # Sample random timesteps for training
        def partbased_sample_t():
            num_parts = kwargs["num_parts"]
            t = self.sample_t(kwargs["num_parts"].shape[0]).to(x_0.device).float()
            t_expanded = torch.cat([t[i].repeat(num_parts[i]) for i in range(len(num_parts))], dim=0)
            return t_expanded,t
            # t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        t,t_unexpand = partbased_sample_t()
        # Diffuse the clean data to timestep t
        x_t = self.diffuse(x_0, t, noise=noise)
        if self.is_predict_arti_info:
            x_0_arti = kwargs['x_0_arti']
            if hasattr(self,"arti_info_diffusion") and self.arti_info_diffusion:
                noise_arti = torch.randn_like(x_0_arti)
                x_t_arti = self.diffuse(x_0_arti, t, noise=noise_arti)
                kwargs['x_t_arti'] = x_t_arti[:, :arti_dim]
        
        # print("cond shape ", cond.shape)
        # Process conditioning information
        cond, ordered_mask_dino = self.get_cond(cond, **kwargs)
        kwargs['ordered_mask_dino'] = ordered_mask_dino
        # print(f"FlowMatchingTrainer cond: {cond.shape}")
        
        # Get model's prediction of the velocity field
        # Multiply t by 1000 to match model's expected timestep scale
        if self.is_predict_arti_info:
            pred, pred_arti = self.training_models['denoiser'](x_t, t * 1000, cond,return_articulation=True, **kwargs)
        else:
            pred = self.training_models['denoiser'](x_t, t * 1000, cond,return_intermediates=False, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        
        # Compute the target velocity vector field
        target = self.get_v(x_0, noise, t)
        if self.is_predict_arti_info:
            if hasattr(self,"arti_info_diffusion") and self.arti_info_diffusion:
                target_arti = self.get_v(x_0_arti, noise_arti, t)
            else:
                target_arti = x_0_arti
        
        # Calculate loss terms
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        # Calculate per-time-bin losses for analysis

        layouts = []
        start_idx = 0
        for num_part in kwargs["num_parts"]:
            layouts.append(slice(start_idx, start_idx+num_part.item()))
            start_idx += num_part.item()
        del start_idx
        
        mse_per_instance = np.array([
            F.mse_loss(pred[layouts[i]], target[layouts[i]]).item()
            for i in range(len(layouts))
        ])
        if self.is_predict_arti_info:
            arti_mask = target_arti[:,arti_dim:]
            if self.masked_articulation_loss:
                assert arti_mask.shape[1] == pred_arti.shape[1]
                terms["mse_arti"] = F.mse_loss(pred_arti*arti_mask, target_arti[:,:arti_dim]*arti_mask) ### LQM: TODO: hard-coded
                
                # mse_per_instance_arti = np.array([
                #     F.mse_loss(pred_arti*arti_mask,\
                #                 target_arti[:,:24]*arti_mask).item()
                    
                # ])
                mse_per_instance_arti = np.array([
                    F.mse_loss(pred_arti[layouts[i]]*arti_mask[layouts[i]],\
                                target_arti[layouts[i]][:,:arti_dim]*arti_mask[layouts[i]]).item()
                    for i in range(len(layouts))
                ])
            else:
                terms["mse_arti"] = F.mse_loss(pred_arti, target_arti[:,:arti_dim]) ### LQM: TODO: hard-coded
                
                mse_per_instance_arti = np.array([
                    F.mse_loss(pred_arti[layouts[i]],\
                                target_arti[layouts[i]][:,:arti_dim]).item()
                    for i in range(len(layouts))
                ])
            
            terms["loss"] = terms["mse"] + terms["mse_arti"]
        else:
            terms["loss"] = terms["mse"]  # Total loss is just MSE for basic flow matching
            

        # Divide time range [0,1] into 10 bins and log loss per bin
        time_bin = np.digitize( t_unexpand.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        # time_bin = np.digitize( t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
                if self.is_predict_arti_info:
                    terms[f"bin_{i}_arti"] = {"mse": mse_per_instance_arti[time_bin == i].mean()}
        
        
        if self.is_predict_arti_info:
            attr_mapping = {
                            "J_type":slice(0, 6),
                            "axis_d":slice(6, 9),
                            "axis_o":slice(9, 12),
                            "j_range":slice(12, 18),
                            "label":slice(18, 24),
                            }
            for attr_name, slices in attr_mapping.items():
                mse_attr_per_instance = np.array([
                    F.mse_loss(pred_arti[:,slices],\
                                target_arti[:,slices]).detach().item()
                ])

                terms[f"arti_{attr_name}"] = {"mse": mse_attr_per_instance.mean()}

        
        
        return terms, {}
        
        
    def test(self):
        # self.arti_out_mode = "regression_last_step"
        sample_steps= 20
        # sample_steps= 50
        # cfg_scale=5.0
        cfg_scale=3.0
        # sample_steps= 25
        # self.dataset.is_postprocessing_arti_info = False
        is_save_samples = True
        if hasattr(self,"load_from_step"):
            
            suffix = f"Step{self.load_from_step}_SAPL{sample_steps}_CFG{cfg_scale}"
        else:
            self.load_from_step = 0
            suffix = f"Step{self.load_from_step}_SAPL{sample_steps}_CFG{cfg_scale}"
        # suffix += "postProcess_axis_o_project"
        suffix += "_all"
        # suffix = "00_all"
        num_samples =  min( 100,len(self.dataset)    )
        # num_samples = len(self.dataset)
        ## self.snapshot(suffix=f"{self.load_from_step}_test", num_samples=32, batch_size=16, verbose=False)
        ## self.snapshot(suffix=f"{self.load_from_step}_{self.arti_mean_num}MeanArti_test_trick_axis_dir", num_samples=16, batch_size=8, verbose=False)
        ## self.snapshot(suffix=suffix, num_samples=16, batch_size=8, verbose=False)
        ## self.snapshot(suffix=f"{self.load_from_step}_{self.arti_mean_num}MeanArti_test_trick_axis_dir_NoAxis_o_recal_postProces_axis_o_project", num_samples=16, batch_size=8, verbose=False)
        self.snapshot(suffix=suffix, num_samples=num_samples, batch_size=4, verbose=True,
                      sample_steps=sample_steps,shuffle=False,
                      cfg_scale=cfg_scale)

        ### traning view overfitting test
        ### traning view overfitting test
        ### traning view overfitting test
        print("### Training view overfitting test ###")
        temp_dataset = self.dataset 
        num_samples = min(32,len(self.train_dataset))
        suffix = f"AAA_trainView" +  f"{self.load_from_step}_SAPL{sample_steps}_CFG{cfg_scale}" 
        
        self.dataset = self.train_dataset

        self.snapshot(suffix=suffix, num_samples=num_samples, batch_size=4, 
                      verbose=True,sample_steps=sample_steps,shuffle=False,
                      cfg_scale=cfg_scale)
        
        
        return
    
