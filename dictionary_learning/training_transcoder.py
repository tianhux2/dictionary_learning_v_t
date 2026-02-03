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


def get_transcoder_norm_stats(data, steps: int):
    """
    计算 Transcoder 输入和输出的统计量。
    分别计算 input 和 output 的 Mean 和 Scale (使得去均值后的 RMS 为 1)。
    """
    print(f"Calculating normalization stats over {steps} steps...")
    
    n_samples = 0
    sum_in = 0
    sum_out = 0
    
    # 临时存储这部分数据以便进行两遍扫描（如果内存允许）
    # 如果数据量太大，可以只用一部分数据估算，或者确保 data generator 可以 reset
    # 这里我们采用在线算法(Welford)的简化版：两遍扫描。
    # 为了简单起见，这里假设我们可以消耗掉这 steps 步的数据用于统计，
    # 或者调用者通过 infinite iterator 提供数据。
    
    batch_buffer = []

    # Pass 1: 计算均值 (Mean)
    for step, act in enumerate(tqdm(data, total=steps, desc="Computing Means")):
        if step >= steps:
            break
        
        # 缓存数据用于第二遍 Pass
        # 注意：如果显存不足，需要把 tensor 移回 CPU
        batch_buffer.append(act.detach().cpu())

        if act.ndim == 3 and act.shape[1] == 2:
            x_in, x_out = act.unbind(dim=1)
        else:
            # 兼容非 Transcoder 的情况，假设只有 Input
            x_in = act
            x_out = act # 这里仅仅为了逻辑跑通，实际非 Transcoder 逻辑不同

        # 累加用于计算 Mean
        sum_in += x_in.sum(dim=0).float() # 使用 float 累加防止溢出
        sum_out += x_out.sum(dim=0).float()
        n_samples += x_in.shape[0]

    mu_in = sum_in / n_samples
    mu_out = sum_out / n_samples

    # Pass 2: 计算缩放因子 (Scale)
    # Scale = sqrt( mean( ||x - mu||^2 ) ) / sqrt(d_model)? 
    # 或者让每个 vector 的 expected L2 norm 为 1? 
    # 原代码使用的是 "unit mean squared norm"，即 mean(sum(x^2)) 归一化。
    # 这里我们针对 centered data 做同样的归一化。
    
    total_sq_norm_in = 0
    total_sq_norm_out = 0
    
    for act in tqdm(batch_buffer, desc="Computing Scales"):
        act = act.to(mu_in.device) # 移回设备计算
        if act.ndim == 3 and act.shape[1] == 2:
            x_in, x_out = act.unbind(dim=1)
        else:
            x_in = act
            x_out = act

        # Center data
        x_in_centered = x_in - mu_in
        x_out_centered = x_out - mu_out
        
        # 计算 mean squared norm (dim=1 是 feature 维度)
        # mean over batch
        total_sq_norm_in += (x_in_centered ** 2).sum(dim=1).mean().item()
        total_sq_norm_out += (x_out_centered ** 2).sum(dim=1).mean().item()

    # 平均
    avg_sq_norm_in = total_sq_norm_in / len(batch_buffer)
    avg_sq_norm_out = total_sq_norm_out / len(batch_buffer)

    scale_in = (avg_sq_norm_in) ** 0.5
    scale_out = (avg_sq_norm_out) ** 0.5

    print(f"Stats: Mu_in norm: {mu_in.norm():.2f}, Scale_in: {scale_in:.4f}")
    print(f"Stats: Mu_out norm: {mu_out.norm():.2f}, Scale_out: {scale_out:.4f}")

    return mu_in, scale_in, mu_out, scale_out


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
                x, x_hat, f, losslog = trainer.loss(act_i, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                
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
    normalize_activations:bool=False,  # 新增参数
    verbose:bool=False,
    device:str="cuda",
    autocast_dtype: t.dtype = t.float32,
):
    """
    Train SAEs / Transcoders using the given trainers
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)

    # 1. 初始化 Trainers
    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    # 2. 初始化 WandB
    wandb_processes = []
    log_queues = []
    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            wandb_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v 
                           for k, v in wandb_config.items()}
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # 3. 初始化 Normalization Stats
    norm_stats = None
    if normalize_activations:
        # 使用前200步数据计算统计量
        # 注意：这会消耗 data iterator 的前200步数据
        stats_steps = 200
        mu_in, scale_in, mu_out, scale_out = get_transcoder_norm_stats(data, steps=stats_steps)
        
        norm_stats = {
            "mu_in": mu_in.to(device),
            "scale_in": scale_in,
            "mu_out": mu_out.to(device),
            "scale_out": scale_out
        }

    # 4. 创建保存目录并导出配置
    try:
        if save_dir is not None:
            save_dirs = [
                os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
            ]
            for trainer, dir in zip(trainers, save_dirs):
                os.makedirs(dir, exist_ok=True)
                config = {"trainer": trainer.config}
                try:
                    config["buffer"] = data.config
                except:
                    pass
                with open(os.path.join(dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
        else:
            save_dirs = [None for _ in trainer_configs]

        # 5. 训练主循环
        for step, act in enumerate(tqdm(data, total=steps)):
            act = act.to(device=device, dtype=autocast_dtype)

            if step >= steps:
                break

            # --- Apply Normalization ---
            if normalize_activations and norm_stats is not None:
                if act.ndim == 3 and act.shape[1] == 2:
                    # 分离 Input 和 Output
                    x_in, x_out = act.unbind(dim=1)
                    
                    # 归一化: (x - mu) / scale
                    x_in = (x_in - norm_stats["mu_in"]) / norm_stats["scale_in"]
                    x_out = (x_out - norm_stats["mu_out"]) / norm_stats["scale_out"]
                    
                    # 重新堆叠
                    act = t.stack([x_in, x_out], dim=1)
                else:
                    # Fallback for standard SAE (assume input=output)
                    # 实际上如果你只跑 standard SAE，应该适配不同的逻辑，但这里为了鲁棒性处理
                    act = (act - norm_stats["mu_in"]) / norm_stats["scale_in"]

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

                        # --- Fold Weights Before Saving (Folding) ---
                        if normalize_activations:
                            trainer.ae.fold_activation_normalization(
                                norm_stats["mu_in"], norm_stats["scale_in"], 
                                norm_stats["mu_out"], norm_stats["scale_out"]
                            )

                        checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                        t.save(
                            checkpoint,
                            os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                        )

                        # --- Unfold Weights To Resume Training (Unfolding) ---
                        if normalize_activations:
                            trainer.ae.unfold_activation_normalization(
                                norm_stats["mu_in"], norm_stats["scale_in"], 
                                norm_stats["mu_out"], norm_stats["scale_out"]
                            )

            # training
            for trainer in trainers:
                with autocast_context:
                    trainer.update(step, act)

        # save final SAEs
        for save_dir, trainer in zip(save_dirs, trainers):
            if save_dir is not None:
                # --- Fold Final Weights ---
                if normalize_activations:
                    trainer.ae.fold_activation_normalization(
                        norm_stats["mu_in"], norm_stats["scale_in"], 
                        norm_stats["mu_out"], norm_stats["scale_out"]
                    )
                
                final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(final, os.path.join(save_dir, "ae.pt"))
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error encountered: {e}")

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()