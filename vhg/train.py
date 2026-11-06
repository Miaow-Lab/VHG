from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VerlTrainingConfig:
    train_files: str
    val_files: str
    model_path: str
    project_name: str
    experiment_name: str
    output_dir: str
    reward_function_path: str = "vhg/rewards.py"
    reward_function_name: str = "compute_score_batched"
    reward_manager: str = "batch"
    train_batch_size: int = 128
    max_prompt_length: int = 768
    max_response_length: int = 3072
    learning_rate: str = "2e-6"
    ppo_mini_batch_size: int = 64
    ppo_micro_batch_size_per_gpu: int = 8
    ref_log_prob_micro_batch_size_per_gpu: int = 8
    ppo_max_token_len_per_gpu: int = 69632
    actor_kl_loss_coef: float = 0.001
    algorithm_kl_coef: float = 0.001
    rollout_n: int = 8
    rollout_gpu_memory_utilization: float = 0.95
    rollout_max_num_batched_tokens: int | None = None
    rollout_temperature: float | None = None
    rollout_top_p: float | None = None
    val_n: int = 10
    val_temperature: float = 1.0
    val_top_p: float = 0.7
    n_gpus_per_node: int = 4
    total_epochs: int = 100
    save_freq: int = 15
    test_freq: int = 15
    resume_mode: str = "auto"
    val_before_train: bool = True
    actor_param_offload: bool = False
    actor_optimizer_offload: bool = False
    ref_param_offload: bool = False
    enable_gradient_checkpointing: bool = True
    logger: str = "['wandb','console']"
    validation_data_dir: str | None = None
    rollout_data_dir: str | None = None


@dataclass(frozen=True)
class VerlSFTConfig:
    data_root: str
    model_path: str
    output_dir: str
    experiment_name: str
    n_gpus_per_node: int = 8
    train_batch_size: int = 32
    micro_batch_size_per_gpu: int = 4
    max_length: int = 4096
    learning_rate: str = "1e-5"
    total_epochs: int = 5
    save_freq: int = 50
    test_freq: int = 1
    resume_mode: str = "disable"
    logger: str = "['wandb','console']"


PLAIN_CHAT_TEMPLATE = (
    "{%for message in messages%}{{message.content}}"
    "{% if not loop.last %}\\n{% endif %}{%endfor%}"
)

SFT_PLAIN_CHAT_TEMPLATE = "{%for message in messages%}{{message.content}}{%endfor%}"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def build_verl_ppo_command(config: VerlTrainingConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={config.train_files}",
        f"data.val_files={config.val_files}",
        f"data.train_batch_size={config.train_batch_size}",
        f"data.max_prompt_length={config.max_prompt_length}",
        f"data.max_response_length={config.max_response_length}",
    ]
    command.append(
        f'+data.apply_chat_template_kwargs.chat_template="{PLAIN_CHAT_TEMPLATE}"'
    )
    command.extend(
        [
        f"actor_rollout_ref.model.path={config.model_path}",
        f"actor_rollout_ref.actor.optim.lr={config.learning_rate}",
        "actor_rollout_ref.model.use_remove_padding=true",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size}",
        f"+actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={config.ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=true",
        f"actor_rollout_ref.actor.kl_loss_coef={config.actor_kl_loss_coef}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_bool(config.enable_gradient_checkpointing)}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_bool(config.actor_param_offload)}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={_bool(config.actor_optimizer_offload)}",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.rollout_gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={config.rollout_n}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={config.val_temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={config.val_top_p}",
        f"actor_rollout_ref.rollout.val_kwargs.n={config.val_n}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={_bool(config.ref_param_offload)}",
        f"+actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={config.ref_log_prob_micro_batch_size_per_gpu}",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16",
        f"algorithm.kl_ctrl.kl_coef={config.algorithm_kl_coef}",
        f"reward_model.reward_manager={config.reward_manager}",
        f"custom_reward_function.path={config.reward_function_path}",
        f"custom_reward_function.name={config.reward_function_name}",
        "trainer.critic_warmup=0.1",
        f"trainer.logger={config.logger}",
        f"trainer.project_name={config.project_name}",
        f"trainer.experiment_name={config.experiment_name}",
        f"trainer.val_before_train={_bool(config.val_before_train)}",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        "trainer.nnodes=1",
        f"trainer.save_freq={config.save_freq}",
        f"trainer.test_freq={config.test_freq}",
        f"trainer.default_local_dir={config.output_dir}",
        f"trainer.resume_mode={config.resume_mode}",
        f"trainer.total_epochs={config.total_epochs}",
        ]
    )
    if config.rollout_max_num_batched_tokens is not None:
        command.append(
            f"actor_rollout_ref.rollout.max_num_batched_tokens={config.rollout_max_num_batched_tokens}"
        )
    if config.rollout_temperature is not None:
        command.append(
            f"actor_rollout_ref.rollout.temperature={config.rollout_temperature}"
        )
    if config.rollout_top_p is not None:
        command.append(f"actor_rollout_ref.rollout.top_p={config.rollout_top_p}")
    if config.validation_data_dir is not None:
        command.append(f"trainer.validation_data_dir={config.validation_data_dir}")
    if config.rollout_data_dir is not None:
        command.append(f"trainer.rollout_data_dir={config.rollout_data_dir}")
    return command


def build_verl_sft_command(config: VerlSFTConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={config.n_gpus_per_node}",
        "-m",
        "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={config.data_root}/train.parquet",
        f"data.val_files={config.data_root}/test.parquet",
        f"data.train_batch_size={config.train_batch_size}",
        "data.prompt_key=prompt",
        "data.response_key=response",
        f"data.max_length={config.max_length}",
        "data.truncation=right",
        f'+data.apply_chat_template_kwargs.chat_template="{SFT_PLAIN_CHAT_TEMPLATE}"',
        "use_remove_padding=true",
        f"optim.lr={config.learning_rate}",
        f"data.micro_batch_size_per_gpu={config.micro_batch_size_per_gpu}",
        f"model.partial_pretrain={config.model_path}",
        "model.fsdp_config.model_dtype=bf16",
        f"trainer.logger={config.logger}",
        "trainer.project_name=vhg_cold_sft",
        f"trainer.experiment_name={config.experiment_name}",
        "trainer.seed=42",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        f"trainer.save_freq={config.save_freq}",
        f"trainer.test_freq={config.test_freq}",
        f"trainer.default_local_dir={config.output_dir}",
        f"trainer.resume_mode={config.resume_mode}",
        f"trainer.total_epochs={config.total_epochs}",
    ]


def build_merge_checkpoint_command(
    local_dir: str | Path,
    target_dir: str | Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(local_dir),
        "--target_dir",
        str(target_dir),
    ]


def build_solver_rl_config(
    *,
    data_root: str,
    model_path: str,
    project_name: str,
    experiment_name: str,
    output_dir: str,
    n_gpus_per_node: int,
    reward_function_path: str = "vhg/rewards.py",
    reward_function_name: str = "compute_score_batched",
    domain: str = "integration",
) -> VerlTrainingConfig:
    if domain not in {"integration", "math"}:
        raise ValueError(f"Unknown solver domain: {domain}")
    return VerlTrainingConfig(
        train_files=f"{data_root}/solver/train.parquet",
        val_files=f"{data_root}/solver/test.parquet",
        model_path=model_path,
        project_name=project_name,
        experiment_name=experiment_name,
        output_dir=output_dir,
        reward_function_path=reward_function_path,
        reward_function_name=reward_function_name,
        reward_manager="batch",
        max_prompt_length=4096 if domain == "math" else 512,
        max_response_length=8192,
        rollout_gpu_memory_utilization=0.92,
        rollout_max_num_batched_tokens=57344,
        ppo_max_token_len_per_gpu=69632,
        actor_param_offload=False,
        actor_optimizer_offload=False,
        ref_param_offload=False,
        save_freq=50,
        test_freq=50,
        n_gpus_per_node=n_gpus_per_node,
        validation_data_dir=f"{output_dir}/validation_data",
        rollout_data_dir=f"{output_dir}/rollout_data",
    )


def build_setter_rl_config(
    *,
    data_root: str,
    model_path: str,
    project_name: str,
    experiment_name: str,
    output_dir: str,
    n_gpus_per_node: int,
    reward_function_path: str = "vhg/rewards.py",
    reward_function_name: str = "compute_score_batched",
    domain: str = "integration",
) -> VerlTrainingConfig:
    if domain not in {"integration", "math"}:
        raise ValueError(f"Unknown setter domain: {domain}")
    return VerlTrainingConfig(
        train_files=f"{data_root}/setter/train.parquet",
        val_files=f"{data_root}/setter/test.parquet",
        model_path=model_path,
        project_name=project_name,
        experiment_name=experiment_name,
        output_dir=output_dir,
        reward_function_path=reward_function_path,
        reward_function_name=reward_function_name,
        reward_manager="batch",
        max_prompt_length=4096 if domain == "math" else 256,
        max_response_length=8192,
        rollout_gpu_memory_utilization=0.9 if domain == "math" else 0.95,
        rollout_max_num_batched_tokens=24576 if domain == "math" else 34816,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        val_temperature=1.0,
        val_top_p=0.7,
        ppo_max_token_len_per_gpu=34816,
        total_epochs=200,
        save_freq=25,
        test_freq=25,
        n_gpus_per_node=n_gpus_per_node,
        validation_data_dir=f"{output_dir}/validation_data",
        rollout_data_dir=f"{output_dir}/rollout_data",
    )


def run_command(command: Sequence[str], *, cwd: str | Path | None = None) -> int:
    env = dict(os.environ)
    python_bin = str(Path(sys.executable).resolve().parent)
    path_parts = env.get("PATH", "").split(os.pathsep)
    if python_bin not in path_parts:
        env["PATH"] = os.pathsep.join([python_bin, *path_parts])
    return subprocess.call(list(command), cwd=str(cwd) if cwd else None, env=env)
