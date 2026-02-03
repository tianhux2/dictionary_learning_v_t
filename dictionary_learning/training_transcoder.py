"""
Training dictionaries
"""

import json
import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional
from contextlib import nullcontext

import torch as t
from tqdm import tqdm

import wandb

from .dictionary import AutoEncoder
from .evaluation import evaluate
from .trainers.standard import StandardTrainer


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except Empty:
            continue
    wandb.finish()


def log_transcoder_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
    verbose: bool=False,
):
    with t.no_grad():
        z = act.clone()
            
        for i, trainer in enumerate(trainers):
            log = {}
            
            act_i = z.clone()
            if activations_split_by_head:
                act_i = act_i[..., i, :]

            if not transcoder:
                # Autoencoder 模式
                act_out, act_hat, f, losslog = trainer.loss(act_i, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act_out, dim=0).sum()
                residual_variance = t.var(act_out - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  
                # Transcoder 模式
                # 注意：Modified JumpReluTrainer.loss 返回的第一个元素是 x_in (用于 logging)
                x, x_hat, f, losslog = trainer.loss(act_i, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # Transcoder 也可以计算 variance explained，但这通常基于 Output
                # 如果你想加，需要确保 trainer.loss 返回了 target (x_out)
                
            if verbose:
                # 修复 Transcoder 模式下 frac_variance_explained 未定义的报错
                if transcoder:
                    print(f"Step {step}: L0 = {l0}")
                else:
                    print(f"Step {step}: L0 = {l0}, frac_variance_explained = {frac_variance_explained}")

            # log parameters from training
            log.update({f"{k}": v.cpu().item() if isinstance(v, t.Tensor) else v for k, v in losslog.items()})
            log[f"l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def trainTranscoder(
    data,
    trainer_configs: list[dict],
    steps: int,
    use_wandb:bool=False,
    wandb_entity:str="",
    wandb_project:str="",
    save_steps:Optional[list[int]]=None,
    save_dir:Optional[str]=None,
    log_steps:Optional[int]=None,
    activations_split_by_head:bool=False,
    transcoder:bool=True,
    run_cfg:dict={},
    verbose:bool=False,
    device:str="cuda",
    autocast_dtype: t.dtype = t.float32,
):
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)

    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    wandb_processes = []
    log_queues = []

    if use_wandb:
        # Note: If encountering wandb and CUDA related errors, try setting start method to spawn in the if __name__ == "__main__" block
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
        # Everything should work fine with the default fork method but it may not be as robust
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v 
                          for k, v in wandb_config.items()}
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    try:
    # make save dirs, export config
        if save_dir is not None:
            save_dirs = [
                os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
            ]
            for trainer, dir in zip(trainers, save_dirs):
                os.makedirs(dir, exist_ok=True)
                # save config
                config = {"trainer": trainer.config}
                try:
                    config["buffer"] = data.config
                except:
                    pass
                with open(os.path.join(dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
        else:
            save_dirs = [None for _ in trainer_configs]

        for step, act in enumerate(tqdm(data, total=steps)):
            act = act.to(dtype=autocast_dtype)

            if step >= steps:
                break

            # logging
            if (use_wandb or verbose) and step % log_steps == 0:
                log_transcoder_stats(
                    trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues, verbose=verbose
                )

            # saving
            if save_steps is not None and step in save_steps:
                for dir, trainer in zip(save_dirs, trainers):
                    if dir is not None:

                        if not os.path.exists(os.path.join(dir, "checkpoints")):
                            os.mkdir(os.path.join(dir, "checkpoints"))

                        checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}

                        t.save(
                            checkpoint,
                            os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                        )

            # training
            for trainer in trainers:
                with autocast_context:
                    trainer.update(step, act)

        # save final SAEs
        for save_dir, trainer in zip(save_dirs, trainers):
            if save_dir is not None:
                final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(final, os.path.join(save_dir, "ae.pt"))
    except Exception as e:
        print(e)

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()
