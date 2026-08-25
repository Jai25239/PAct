"""
Sparse Flow Matching Trainer Implementation

This file implements trainers for sparse generative models using flow matching objectives.
Flow matching is a technique for generative modeling that defines probability flows between
noise and data distributions. This implementation specifically handles sparse data structures,
which are common in 3D point clouds, graphs, and other non-dense representations.

The file contains multiple trainer classes:
- SparseFlowMatchingTrainer: Base trainer for sparse flow matching models
- SparseFlowMatchingCFGTrainer: Adds classifier-free guidance for improved generation
- TextConditionedSparseFlowMatchingCFGTrainer: Enables text conditioning for sparse generation
- ImageConditionedSparseFlowMatchingCFGTrainer: Enables image conditioning for sparse generation

These trainers handle the training loop, loss calculation, data loading, and sampling for
sparse flow matching models.
"""

from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from modules.pact.pipelines import samplers

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin
import imageio
from torchvision import utils

class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
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
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader for training.
        
        This method initializes the data sampler and dataloader with proper configurations
        for efficient batch processing. It uses BalancedResumableSampler to ensure training
        can be resumed from checkpoints, and sets up an infinite iterator over the data.
        """
        print("original dataset size:", len(self.dataset))
        num_dataset = 128
        # Wrap your dataset in the DuplicatedDataset if it's too small
        # print(f"Dataset size: {len(self.dataset)}")
        if len(self.dataset) < num_dataset and not self.dataset.is_test:  # Adjust this threshold as needed
            from ...utils.data_utils import DuplicatedDataset

            self.dataset = DuplicatedDataset(self.dataset, repeat=num_dataset)
            print(f"Dataset duplicated to {len(self.dataset)} samples")
        ### LQM: code Ominipart
        # print("data_sampler:")
        # Create a sampler that can be resumed from checkpoints
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        
        configured_num_workers = kwargs.get('num_workers')
        if configured_num_workers is None:
            num_workers = int(np.ceil(os.cpu_count() / torch.cuda.device_count() / 8))
        else:
            num_workers = int(configured_num_workers)
            if num_workers < 0:
                raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        print(f"SparseFlowMatchingTrainer: num_workers={num_workers}")

        # Create the dataloader with optimized settings
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,  # Speed up CPU to GPU transfer
            drop_last=True,  # Ensure all batches are the same size
            persistent_workers=num_workers > 0,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        
        # Create an infinite data iterator to simplify training loop
        self.data_iterator = cycle(self.dataloader)
        
    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        This method implements the flow matching training objective for sparse data.
        It diffuses the input data according to the flow matching schedule,
        predicts the velocity field, and computes the loss between the prediction and target.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            A tuple containing:
            - terms: Dictionary with the key "loss" and other loss components
            - An empty dictionary (for compatibility with other trainers)
        """
        # Generate random noise with the same sparsity pattern as input
        noise = x_0.replace(torch.randn_like(x_0.feats))
        
        # Sample random timesteps for each item in the batch
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        
        # Apply diffusion process to get x_t
        x_t = self.diffuse(x_0, t, noise=noise)
        
        # Process conditional inputs if any
        cond, ordered_mask_dino = self.get_cond(cond, **kwargs)
        kwargs['ordered_mask_dino'] = ordered_mask_dino  ### LQM: code Ominipart
        # print(f"loss shape cond: {cond.shape}") # loss shape cond: torch.Size([2, 1374, 1024])
        
        # Run model to predict velocity field
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        # print(pred.shape, noise.shape, x_0.shape)
        assert pred.shape == noise.shape == x_0.shape
        
        # Calculate target vector field based on flow matching objective
        target = self.get_v(x_0, noise, t)
        
        # Compute loss terms
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # Log detailed loss statistics binned by timestep
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        # Divide time range [0,1] into 10 bins and compute per-bin statistics
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
        sample_steps: int = 50,
        shuffle: bool = True,
        dataset = None,
    ) -> Dict:
        """
        Generate samples for visualization and evaluation.
        
        This method creates samples from the model for visualization purposes. It processes
        batches of data from the dataset, generates conditional samples, and organizes
        them for logging and visualization.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size to use for generation
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing generated samples, ground truth, and conditioning information
        """
        dataset = dataset if dataset is not None else self.dataset
        # Create a temporary dataloader for sampling
        dataloader = DataLoader(
            copy.deepcopy(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # No parallelization for simplicity
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
        )

        # Get the sampler for generation
        sampler = self.get_sampler()
        
        # Initialize lists to store results
        sample_gt = []  # Ground truth samples
        sample = []     # Generated samples
        cond_vis = []   # Visualization of conditioning
        cond_vis_mask = []   # Visualization of conditioning
        sample_gt_arti = []
        sample_arti = []
        part_layouts_list = []
        data_iterator = iter(dataloader)
        # Generate samples in batches
        for i in range(0, num_samples, batch_size):
            # Get actual batch size (might be smaller for the last batch)
            batch = min(batch_size, num_samples - i)
            
            # Get a batch of data
            data = next(data_iterator) ## LQM: check it: FIXME:
            data = {k: v[:batch].cuda() if not isinstance(v, list) else v[:batch] for k, v in data.items()}
            ### LQM: code Ominipart
            for layout in data['part_layouts']:
                part_layouts_list.append(layout)
             ### LQM: code Ominipart
            # Create initial noise with same sparsity pattern as input
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            
            # Store ground truth and conditioning visualization
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            cond_vis_mask.append(self.vis_cond_mask(data['img_mask_vis']))
            
            # Remove ground truth from data dictionary
            del data['x_0']
            
            # Prepare conditioning for inference
            args = self.get_inference_cond(**data)
            args["return_articulation"]= True if hasattr(self, 'is_predict_arti_info') and self.is_predict_arti_info else False ### LQM: code Ominipart
            # Generate samples using the sampler
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=sample_steps,  # Number of sampling steps
                cfg_strength=3.0,  # Classifier-free guidance strength
                verbose=verbose,
            )
            sample.append(res.samples)
            if hasattr(self, 'is_predict_arti_info') and self.is_predict_arti_info:
                sample_gt_arti.append(data['x_0_arti'])
                if self.arti_out_mode == 'regression_last_step':
                    sample_arti.append(res.arti)
                elif self.arti_out_mode == 'regression_mean_steps':
                    arti_mean_num = self.arti_mean_num if hasattr(self,"arti_mean_num") else 10
                    feats = [pre.feats for pre in res.pred_arti_t[-arti_mean_num:]]
                    feats = torch.mean(torch.stack(feats), dim=0)
                    art_mean = sp.SparseTensor(coords= res.arti.coords,feats=feats)
                    assert art_mean.feats.shape == res.arti.feats.shape
                    sample_arti.append(art_mean)
                elif self.arti_out_mode == 'diffusion':
                    sample_arti.append(res.arti) ## check this
                else:
                    raise ValueError(f"Unknown arti_out_mode: {self.arti_out_mode}")

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},  # Ground truth
            'sample': {'value': sample, 'type': 'sample'},        # Generated samples
        }
        
        if hasattr(self, 'is_predict_arti_info') and self.is_predict_arti_info:
            sample_gt_arti = sp.sparse_cat(sample_gt_arti)
            sample_arti = sp.sparse_cat(sample_arti)
            # sample_arti = sp.sparse_cat(sample_arti)
            
            # for idx, (art_gt, arti) in enumerate(zip(sample_gt_arti, sample_arti)):
            
            sample_dict['sample_gt_arti'] = {'value': sample_gt_arti, 'type': 'articulation'}
            sample_dict['sample_arti'] = {'value': sample_arti, 'type':"articulation"}
            
            

        # Add conditioning visualization to dictionary
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        sample_dict.update(dict_reduce(cond_vis_mask, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        sample_dict['layout'] = {'value': part_layouts_list, 'type': 'layout'}
        ### LQM: code Ominipart
        return sample_dict


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
    This class extends SparseFlowMatchingTrainer with classifier-free guidance capabilities,
    which helps improve sample quality by learning both conditional and unconditional models.
    
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


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    This class adds text conditioning capabilities to the sparse flow matching trainer,
    allowing the generation of sparse data conditioned on text prompts.
    
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


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    This class adds image conditioning capabilities to the sparse flow matching trainer,
    allowing the generation of sparse data conditioned on input images.
    
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

from ..basic import * ### lazy module ### LQM: TODO: check if this import is necessary
class ImageConditionedSparseFlowMatchingCFGTrainer_Articulation(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arti_info_diffusion = kwargs.get('arti_info_diffusion', False)
        self.is_predict_arti_info = True
        self.arti_out_mode = kwargs.get('arti_out_mode', 'regression_last_step') ### 'regression_last_step', 'regression_mean_steps', 'diffusion'
        self.masked_articulation_loss = kwargs.get('masked_articulation_loss', 'False') ### 'regression_last_step', 'regression_mean_steps', 'diffusion'
        print("Masked articulation loss:", self.masked_articulation_loss)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        This method implements the flow matching training objective for sparse data.
        It diffuses the input data according to the flow matching schedule,
        predicts the velocity field, and computes the loss between the prediction and target.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            A tuple containing:
            - terms: Dictionary with the key "loss" and other loss components
            - An empty dictionary (for compatibility with other trainers)
        """
        # Generate random noise with the same sparsity pattern as input
        noise = x_0.replace(torch.randn_like(x_0.feats))
        
        # Sample random timesteps for each item in the batch
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        
        # Apply diffusion process to get x_t
        x_t = self.diffuse(x_0, t, noise=noise)
        
        x_0_arti = kwargs['x_0_arti']
        if hasattr(self,"arti_info_diffusion") and self.arti_info_diffusion:
            noise_arti = x_0_arti.replace(torch.randn_like(x_0_arti.feats))
            x_t_arti = self.diffuse(x_0_arti, t, noise=noise_arti)
            kwargs['x_t_arti'] = x_t_arti

        # Process conditional inputs if any
        cond, ordered_mask_dino = self.get_cond(cond, **kwargs)
        kwargs['ordered_mask_dino'] = ordered_mask_dino  ### LQM: code Ominipart
        # print(f"loss shape cond: {cond.shape}") # loss shape cond: torch.Size([2, 1374, 1024])
        
        # Run model to predict velocity field
        if hasattr(self,"arti_info_diffusion") and self.arti_info_diffusion:
            pred, pred_arti = self.training_models['denoiser'](x_t, t * 1000, cond, return_articulation=True, **kwargs)
        else:
            pred,pred_arti = self.training_models['denoiser'](x_t, t * 1000, cond,return_articulation=True, **kwargs)
        # print(pred.shape, noise.shape, x_0.shape)
        assert pred.shape == noise.shape == x_0.shape
        
        # Calculate target vector field based on flow matching objective
        target = self.get_v(x_0, noise, t)
        if hasattr(self,"arti_info_diffusion") and self.arti_info_diffusion:
            target_arti = self.get_v(x_0_arti, noise_arti, t)
        else:
            target_arti = x_0_arti
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]
        arti_mask = target_arti.feats[:,24:]
        if self.masked_articulation_loss:
            assert arti_mask.shape[1] == pred_arti.feats.shape[1]
            terms["mse_arti"] = F.mse_loss(pred_arti.feats*arti_mask, target_arti.feats[:,:24]*arti_mask) ### LQM: TODO: hard-coded
        else:
            terms["mse_arti"] = F.mse_loss(pred_arti.feats, target_arti.feats[:,:24]) ### LQM: TODO: hard-coded
        
        terms["loss"] = terms["loss"] + terms["mse_arti"]

        # Log detailed loss statistics binned by timestep
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        mse_per_instance_arti = np.array([
            F.mse_loss(pred_arti.feats[x_0_arti.layout[i]],\
                        target_arti.feats[x_0_arti.layout[i]][:,:24]).item()
            for i in range(x_0_arti.shape[0])
        ])
        # Divide time range [0,1] into 10 bins and compute per-bin statistics
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
                terms[f"bin_{i}_arti"] = {"mse": mse_per_instance_arti[time_bin == i].mean()}
        if True:
            attr_mapping = {
                            "J_type":slice(0, 6),
                            "axis_d":slice(6, 9),
                            "axis_o":slice(9, 12),
                            "j_range":slice(12, 18),
                            "label":slice(18, 24),
                            }
            for attr_name, slices in attr_mapping.items():
                mse_attr_per_instance = np.array([
                    F.mse_loss(pred_arti.feats[:,slices],\
                                target_arti.feats[:,slices]).detach().item()
                ])

                terms[f"arti_{attr_name}"] = {"mse": mse_attr_per_instance.mean()}

            # terms["mse_arti"] = F.mse_loss(pred_arti.feats, target_arti.feats[:,:24]) ### LQM: TODO: hard-coded
           

        return terms, {}
    def init_articulation_model(self, base_model, model_args):
        # import models
        # from modules.pact import models
        # model = getattr(models, model_args["name"])(base_model, **model_args["args"]).cuda()
        # # pass
        # return model
        pass 
    
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
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

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
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSamplerArticulation:
        """
        Get the specialized sampler for classifier-free guidance flow matching.
        
        Args:
            **kwargs: Additional arguments to pass to the sampler.
            
        Returns:
            An instance of FlowEulerCfgSampler configured with the model's sigma_min.
        """
        return samplers.FlowEulerCfgSamplerArticulation(self.sigma_min)
    
    def run_inference(self):
        pass 
    
        raise NotImplementedError
    def manipulate_with_articulation(self):
        pass 
        raise NotImplementedError
    
    @torch.no_grad()
    def save_articulated_sample(self, sample, save_path):
        """
        Convert a sample to an image for visualization.
        
        Args:
            sample: Data sample to visualize
            
        Returns:
            torch.Tensor or dict: Processed sample ready for visualization
        """
        if hasattr(self.dataset, 'save_articulated_sample'):
            return self.dataset.save_articulated_sample(sample,save_path)
        else:
            return sample
    def validate(self):
        # self.arti_out_mode = "regression_last_step"
        self.arti_out_mode = "regression_mean_steps"
        art_mode = "Mean" if self.arti_out_mode == "regression_mean_steps" else "last"
        self.arti_mean_num = 20
        sample_steps= 25
        self.validate_dataset.is_postprocessing_arti_info = True
        # self.dataset.is_postprocessing_arti_info = False
        is_save_samples = False
        is_save_gt = False
        
        suffix = f"_SAPL{sample_steps}"
        if self.arti_out_mode == "regression_mean_steps":
            suffix += f"_{self.arti_mean_num}{art_mode}Arti_test_trick_axis_dir_NoAxis_o_recal_"
        elif self.arti_out_mode == "regression_last_step":
            suffix += f"_{art_mode}Arti_test_trick_axis_dir_NoAxis_o_recal_"
        if self.validate_dataset.is_postprocessing_arti_info:
            suffix += "postProcess_axis_o_project"
        suffix += "_all"
        # suffix = "00_all"
        # num_samples = 4
        num_samples =   min(len(self.validate_dataset), 100)
        # num_samples =   min(len(self.validate_dataset), 4)
        ## self.snapshot(suffix=f"{self.load_from_step}_test", num_samples=32, batch_size=16, verbose=False)
        ## self.snapshot(suffix=f"{self.load_from_step}_{self.arti_mean_num}MeanArti_test_trick_axis_dir", num_samples=16, batch_size=8, verbose=False)
        ## self.snapshot(suffix=suffix, num_samples=16, batch_size=8, verbose=False)
        ## self.snapshot(suffix=f"{self.load_from_step}_{self.arti_mean_num}MeanArti_test_trick_axis_dir_NoAxis_o_recal_postProces_axis_o_project", num_samples=16, batch_size=8, verbose=False)
        with torch.inference_mode():
            samples = self.snapshot(dataset=self.validate_dataset, suffix=suffix, num_samples=num_samples, batch_size=8, 
                      verbose=False,sample_steps=sample_steps,
                      shuffle=False,save_gt=is_save_gt,save_samples=is_save_samples)
        # calculate_articulation_metrics(samples,
        #                               save_dir=os.path.join(self.output_dir, 'samples', suffix),
        # metric_dict = calculate_articulation_metrics(samples,
        #                               save_dir=os.path.join(self.output_dir, 'samples', suffix),
        #                               use_postprocessed_arti_info=self.dataset.is_postprocessing_arti_info)
        
        gt_arti  = samples["sample_gt_arti"]["value"].feats[:, :24]
        pred_arti = samples["sample_arti"]["value"].feats[:, :24]
        mask = samples["sample_gt_arti"]["value"].feats[:, 24:]
        assert gt_arti.shape == pred_arti.shape == mask.shape
        mse_arti = F.mse_loss(pred_arti*mask, gt_arti*mask).item()
        metric_dict = {"val/arti_mse": mse_arti,
                       "val/num_samples": samples["sample_gt_arti"]["value"].shape[0],}
        
        attr_mapping = {
                "J_type":slice(0, 6),
                "axis_d":slice(6, 9),
                "axis_o":slice(9, 12),
                "j_range":slice(12, 18),
                "label":slice(18, 24),
                }
        for attr_name, slices in attr_mapping.items():
            mse_attr = np.array([
                F.mse_loss(pred_arti[:,slices]*mask[:,slices],\
                            gt_arti[:,slices]*mask[:,slices]).detach().item()
            ])

            metric_dict[f"val/arti_{attr_name}"] = mse_attr.mean()
            # metric_dict[f"val/arti_{attr_name}"] = {"mse": mse_attr.mean()}
    
        
        # metric_dict["Overall Articulation MSE"] = mse_arti
        
        
        return metric_dict 
        
        
        
        # samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose,sample_steps=sample_steps,shuffle=shuffle)
    
    @torch.no_grad()
    def snapshot(self,dataset=None, suffix=None, num_samples=10, batch_size=4, verbose=False, save_samples=False,
                 save_gt= True,
                 sample_steps=50,shuffle=True) -> Dict:

        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        # Set default suffix to current step if none provided (used for organizing output files)
        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Calculate how many samples each process should generate in distributed setting
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        # # Generate samples using the model's snapshot implementation

        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose,sample_steps=sample_steps,shuffle=shuffle)

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
            if not save_gt and 'gt' in key:
                continue
            
            if samples[key]['type'] == 'sample':
                # Convert raw samples to visualizable format using the dataset's visualization method
                # vis = self.visualize_sample(samples[key]['value'])
                data_dict = {   'x_0': samples[key]['value'], 
                                'part_layouts': samples['layout']['value'],
                                "x_0_arti": samples.get(f'{key}_arti', {'value': None})['value'] \
                                    if samples.get(f'{key}_arti', None) is not None else None,
                                                     }
                if save_samples:
                    # raise NotImplementedError("Saving raw samples is not implemented yet.")

                    self.save_articulated_sample(data_dict, save_path=os.path.join(self.output_dir, 'samples',
                                                                                   suffix, f'geo_{key}',
                                                                                   f'articulated_{key}.pkl'))
                
                vis, videos,videos_art = self.visualize_sample(data_dict)
                if isinstance(vis, dict):
                    # If visualization returns a dictionary, create multiple entries with different visualizations
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    # Remove the original entry since it's been replaced with specific visualizations
                    del samples[key]
                else:
                    # Otherwise, update the existing entry by replacing with visualization
                    samples[key] = {'value': vis, 'type': 'image'}
                is_save_video_grid = True
                grid_size_max = 4  # Define a threshold for maximum grid size
                if videos is not None and self.is_master:
                    if is_save_video_grid:
                        for idx in range(0,len(videos),grid_size_max*grid_size_max):
                            sub_videos = videos[idx:idx+grid_size_max*grid_size_max]
                            if len(sub_videos) == 0:
                                continue
                            save_video_grid(
                                # videos,
                                [np.stack(video) for video in sub_videos],
                                os.path.join(self.output_dir, 'samples', suffix, f'videos_{key}'),
                                f'{key}_grid_sub{idx//(grid_size_max*grid_size_max)}.mp4',
                            fps=20,
                            )
                    else:
                        output_dir = os.path.join(self.output_dir, 'samples', suffix, f'videos_{key}')
                        if os.path.exists(output_dir) is False:
                                os.makedirs(output_dir)
                        for i, video in enumerate(videos):

                            gaussian_video_path = f"{output_dir}/object_{i}_gs.mp4"
                            imageio.mimsave(gaussian_video_path, video, fps=20)
                            
                        
                if videos_art is not None and self.is_master:
                    if is_save_video_grid:
                        for idx in range(0,len(videos_art),grid_size_max*grid_size_max):
                            sub_videos = videos_art[idx:idx+grid_size_max*grid_size_max]
                            if len(sub_videos) == 0:
                                continue
                            save_video_grid(
                                [np.stack(video) for video in sub_videos],
                                os.path.join(self.output_dir, 'samples', suffix, f'videos_art_{key}'),
                                f'{key}_art_grid_sub{idx//(grid_size_max*grid_size_max)}.mp4',
                                fps=20,
                            )
                    else:
                        # for idx,  (art_key, video_list) in enumerate(video_art_dict.items()):
                        output_dir = os.path.join(self.output_dir, 'samples', suffix, f'videos_art_{key}')
                        if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                        for j, video in enumerate(videos_art):

                            gaussian_video_path = f"{output_dir}/object_{j}_gs.mp4"
                            imageio.mimsave(gaussian_video_path, video,)
                            
                            
        # Remove the layout entry after processing
        if 'layout' in samples:
            del samples['layout']
        # Gather samples from all processes in distributed training setup
        if self.world_size > 1:
            dist.barrier()
            for key in samples.keys():
                # if isinstance(samples[key]['value'], list):
                #     continue
                if key in ['sample', 'sample_gt', 'image',"mask"]:
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
        return samples   
        
    def test(self):
        # self.arti_out_mode = "regression_last_step"
        self.arti_out_mode = "regression_mean_steps"
        art_mode = "Mean" if self.arti_out_mode == "regression_mean_steps" else "last"
        self.arti_mean_num = 20
        sample_steps= 25
        self.dataset.is_postprocessing_arti_info = True
        # self.dataset.is_postprocessing_arti_info = False
        is_save_samples = True
        is_save_gt = False
        
        suffix = f"{self.load_from_step}_SAPL{sample_steps}"
        if self.arti_out_mode == "regression_mean_steps":
            suffix += f"_{self.arti_mean_num}{art_mode}Arti_test_trick_axis_dir_NoAxis_o_recal_"
        elif self.arti_out_mode == "regression_last_step":
            suffix += f"_{art_mode}Arti_test_trick_axis_dir_NoAxis_o_recal_"
        if self.dataset.is_postprocessing_arti_info:
            suffix += "postProcess_axis_o_project"
        suffix += "_all"
        # suffix = "00_all"
        # num_samples = 4
        num_samples = len(self.dataset)
        ## self.snapshot(suffix=f"{self.load_from_step}_test", num_samples=32, batch_size=16, verbose=False)
        ## self.snapshot(suffix=f"{self.load_from_step}_{self.arti_mean_num}MeanArti_test_trick_axis_dir", num_samples=16, batch_size=8, verbose=False)
        ## self.snapshot(suffix=suffix, num_samples=16, batch_size=8, verbose=False)
        ## self.snapshot(suffix=f"{self.load_from_step}_{self.arti_mean_num}MeanArti_test_trick_axis_dir_NoAxis_o_recal_postProces_axis_o_project", num_samples=16, batch_size=8, verbose=False)
        self.snapshot(suffix=suffix, num_samples=num_samples, batch_size=8, 
                      verbose=True,sample_steps=sample_steps,shuffle=False,save_gt=is_save_gt,save_samples=is_save_samples)
        # self.snapshot(suffix=suffix, num_samples=num_samples, batch_size=8, verbose=False,sample_steps=sample_steps,shuffle=False,save_samples=is_save_samples)

        ### traning view overfitting test
        ### traning view overfitting test
        ### traning view overfitting test
        print("### Training view overfitting test ###")
        temp_dataset = self.dataset 
        num_samples = 32
        suffix = f"AAA_trainView" +  f"{self.load_from_step}_SAPL{sample_steps}_" + self.arti_out_mode
        
        self.dataset = self.train_dataset
        self.dataset.is_postprocessing_arti_info=False ## avoided 
        self.snapshot(suffix=suffix, num_samples=num_samples, batch_size=8, 
                      verbose=False,sample_steps=sample_steps,shuffle=False,save_gt=is_save_gt,save_samples=is_save_samples)
        
        
        return
    

