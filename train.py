#!/usr/bin/env python3
"""Train the released PAct models with the TRELLIS trainer hierarchy."""

import argparse
import copy
import glob
import json
import os
import random
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from easydict import EasyDict as edict

from modules.pact import datasets, models, trainers
from modules.pact.utils.dist_utils import setup_dist


def export_inference_artifact(trainer, cfg):
    """Export TRELLIS checkpoint weights in the existing PAct loader format."""
    from modules.pact.training.checkpoint import save_model_artifact

    model = trainer.models["denoiser"]
    if trainer.ema_rate:
        ema_state = trainer._master_params_to_state_dicts(trainer.ema_params[0])[
            "denoiser"
        ]
        model.load_state_dict(ema_state, strict=False)
    lora = cfg.trainer.args.get("lora_setting")
    if cfg.stage == "structured_latent" and lora and lora.get("use_lora"):
        model.set_base_model(model.base_model.merge_and_unload())
    destination = Path(cfg.output_dir) / "ckpts" / "inference" / "model"
    denoiser_args = dict(cfg.models.denoiser.args)
    if cfg.stage == "structured_latent":
        denoiser_args["use_checkpoint"] = False
    articulation_args = (
        dict(cfg.models.Articulation.args)
        if cfg.stage == "structured_latent"
        else None
    )
    if articulation_args is not None:
        articulation_args["use_checkpoint"] = False
    save_model_artifact(
        model,
        destination,
        cfg.stage,
        {"name": cfg.models.denoiser.name, "args": denoiser_args},
        (
            {
                "name": cfg.models.Articulation.name,
                "args": articulation_args,
            }
            if cfg.stage == "structured_latent"
            else None
        ),
    )
    print(f"Exported inference artifact: {destination}")


def find_ckpt(cfg):
    """Resolve TRELLIS' split-checkpoint step from --load-dir/--ckpt."""
    cfg.load_ckpt = None
    if not cfg.load_dir:
        return cfg
    if cfg.ckpt == "latest":
        files = glob.glob(os.path.join(cfg.load_dir, "ckpts", "denoiser_step*.pt"))
        if files:
            cfg.load_ckpt = max(
                int(Path(path).stem.rsplit("step", 1)[1]) for path in files
            )
    elif cfg.ckpt != "none":
        cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(seed, rank):
    value = int(seed) + rank
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    random.seed(value)


def get_model_summary(model):
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return (
        f"{model.__class__.__name__}\n"
        f"Number of parameters: {params}\n"
        f"Number of trainable parameters: {trainable}\n"
    )


def build_dataset(cfg, is_test=False):
    if cfg.smoke_test:
        from modules.pact.training.data import (
            SyntheticSparseStructureDataset,
            SyntheticStructuredLatentDataset,
        )

        dataset_cls = (
            SyntheticSparseStructureDataset
            if cfg.stage == "sparse_structure"
            else SyntheticStructuredLatentDataset
        )
        dataset = dataset_cls(cfg.models.denoiser.args)
        dataset.is_test = is_test
        return dataset
    return getattr(datasets, cfg.dataset.name)(
        cfg.data_dir, is_test=is_test, **cfg.dataset.args
    )


def run_worker(local_rank, cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("The formal TRELLIS/PAct trainer requires a CUDA GPU.")

    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)
    torch.cuda.set_device(local_rank)
    setup_rng(cfg.seed, rank)

    dataset = build_dataset(cfg, is_test=False)
    model_dict = {
        name: getattr(models, spec.name)(**spec.args).cuda()
        for name, spec in cfg.models.items()
    }

    if rank == 0:
        for name, model in model_dict.items():
            summary = get_model_summary(model)
            print(f"\nBackbone: {name}\n{summary}")
            Path(cfg.output_dir, f"{name}_model_summary.txt").write_text(
                summary, encoding="utf-8"
            )

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        **copy.deepcopy(dict(cfg.trainer.args)),
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir or None,
        step=cfg.load_ckpt,
    )

    if cfg.validate_only:
        if not hasattr(trainer, "validate"):
            raise ValueError(f"{cfg.trainer.name} does not implement validation")
        trainer.validate_dataset = build_dataset(cfg, is_test=True)
        metrics = trainer.validate()
        if rank == 0:
            print(json.dumps(metrics, indent=2))
        return

    if cfg.export_only:
        if cfg.load_ckpt is None:
            raise ValueError("--export-only requires a checkpoint selected by --ckpt")
        if trainer.is_master:
            export_inference_artifact(trainer, cfg)
            trainer.writer.close()
        return

    if cfg.tryrun:
        return
    if cfg.profile:
        trainer.profile()
        return

    if cfg.smoke_test:
        trainer.run_step(trainer.load_data())
        trainer.step += 1
        if trainer.is_master:
            trainer.save()
            export_inference_artifact(trainer, cfg)
            trainer.writer.close()
        return

    trainer.run()
    if trainer.is_master and trainer.step % trainer.i_save:
        trainer.save()
    if trainer.is_master:
        export_inference_artifact(trainer, cfg)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TRELLIS-style JSON config")
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", required=True
    )
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="./data")
    parser.add_argument(
        "--split-info",
        help="Optional JSON containing train/test instance lists",
    )
    parser.add_argument("--load-dir", "--load_dir", dest="load_dir", default="")
    parser.add_argument(
        "--ckpt", default="latest", help="latest, none, or a numeric checkpoint step"
    )
    parser.add_argument(
        "--pretrained-denoiser",
        help="Override trainer.args.finetune_ckpt.denoiser with a local .pt/.ckpt file",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tryrun", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--auto-retry", type=int, default=0)
    parser.add_argument("--num-nodes", "--num_nodes", dest="num_nodes", type=int, default=1)
    parser.add_argument("--node-rank", "--node_rank", dest="node_rank", type=int, default=0)
    parser.add_argument("--num-gpus", "--num_gpus", dest="num_gpus", type=int, default=-1)
    parser.add_argument("--master-addr", "--master_addr", dest="master_addr", default="localhost")
    parser.add_argument("--master-port", "--master_port", dest="master_port", default="12345")
    return parser


def load_config(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = edict(copy.deepcopy(config))
    cfg.update(vars(args))
    cfg.load_dir = cfg.load_dir or cfg.output_dir
    cfg.num_gpus = torch.cuda.device_count() if cfg.num_gpus == -1 else cfg.num_gpus
    if cfg.num_gpus < 1:
        raise RuntimeError("No CUDA GPU selected; pass --num-gpus with a positive value.")
    if args.max_steps is not None:
        cfg.trainer.args.max_steps = args.max_steps
    if args.num_workers is not None:
        cfg.trainer.args.num_workers = args.num_workers
    if args.split_info:
        cfg.dataset.args.split_info_json = str(
            Path(args.split_info).expanduser().resolve()
        )
    if args.pretrained_denoiser:
        cfg.trainer.args.finetune_ckpt = {
            "denoiser": str(Path(args.pretrained_denoiser).expanduser().resolve())
        }
    lora = cfg.trainer.args.get("lora_setting")
    if lora and lora.get("use_lora") and not cfg.trainer.args.get("finetune_ckpt"):
        raise ValueError(
            "LoRA training requires --pretrained-denoiser (a local TRELLIS/PAct "
            "state-dict checkpoint)."
        )
    return config, find_ckpt(cfg)


def main():
    args = build_parser().parse_args()
    os.environ.setdefault("SPCONV_ALGO", "native")
    config, cfg = load_config(args)

    if cfg.node_rank == 0:
        output = Path(cfg.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        cfg.output_dir = str(output)
        Path(output, "command.txt").write_text(
            shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
        )
        Path(output, "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

    attempts = max(1, cfg.auto_retry + 1)
    for attempt in range(attempts):
        try:
            if cfg.num_gpus > 1:
                mp.spawn(run_worker, args=(cfg,), nprocs=cfg.num_gpus, join=True)
            else:
                run_worker(0, cfg)
            return 0
        except Exception:
            if attempt + 1 == attempts:
                raise
            print(f"Training failed; retrying ({attempt + 1}/{cfg.auto_retry})")


if __name__ == "__main__":
    raise SystemExit(main())
