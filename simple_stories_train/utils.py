import os
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from torch import nn


def print0(*args: Any, **kwargs: Any) -> None:
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def is_checkpoint_step(step: int) -> bool:
    # step & (step - 1) == 0 iff step is a power of 2. Therefore, the following
    # expression will be true iff step is a power of two between 0 and 1000
    # or step is a multiple of 1000.
    return (0 < step < 1000 and (step & (step - 1)) == 0) or step % 1000 == 0


def save_model_and_config(
    save_dir: Path,
    model: nn.Module,
    config_dict: dict[str, Any],
    step: int,
    config_filename: str = "final_config.yaml",
) -> None:
    """Save the model to disk and wandb. Also save the config file if it doesn't exist.

    Args:
        save_dir: The directory to save the model and config to.
        model: The model to save.
        step: The current step (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / config_filename, "w") as f:
        yaml.dump(config_dict, f)
    model_file_name = f"model_step_{step}.pt"
    model_file = save_dir / model_file_name
    torch.save(model.state_dict(), model_file)
    print0(f"Saved model to {model_file}")
    if config_dict.get("wandb_project"):
        wandb.save(str(model_file), policy="now", base_path=save_dir)
        print0(f"Saved model to wandb: {str(model_file_name)}")


def init_wandb(config: Any, project: str) -> None:
    wandb.init(
        project=project,
        config=config,
    )


def log_metrics(step: int, metrics: dict[str, Any]) -> None:
    wandb.log(metrics, step=step)


def log_generations(step: int, generations: list[list[str]]) -> None:
    wandb.log(
        {
            "generation_tables": wandb.Table(
                data=generations,
                columns=["step", "generated text"],
            )
        },
        step=step,
    )
