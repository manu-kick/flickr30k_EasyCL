# utils.py
import os
import tempfile
import torch
import wandb


def save_checkpoint(
    save_dir: str,
    filename: str,
    text_encoder,
    vision_encoder,
    shared_head,
    optimizer,
    contra_temp,
    iteration: int,
    best_val_loss: float,
    cf=None,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    ckpt = {
        "iteration": iteration,
        "best_val_loss": float(best_val_loss),
        "text_encoder_state": text_encoder.state_dict(),
        "vision_encoder_state": vision_encoder.state_dict(),
        "shared_head_state": shared_head.state_dict() if shared_head is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "contra_temp": float(contra_temp.item()) if torch.is_tensor(contra_temp) else contra_temp,
        "config": vars(cf) if cf is not None else None,
    }

    torch.save(ckpt, path)
    return path


def log_model_to_wandb(
    text_encoder,
    vision_encoder,
    shared_head,
    optimizer,
    contra_temp,
    iteration: int,
    best_val_loss: float,
    cf=None,
    artifact_name: str = "best-model",
    artifact_alias: str = "latest",
    extra_metadata=None,
):
    """
    Carica su W&B il checkpoint come Artifact versionato.
    - artifact_name: nome fisso dell'artifact (W&B gestisce le versioni)
    - artifact_alias: di solito "latest" o "best"
    """

    if wandb.run is None:
        raise RuntimeError("wandb.run is None: hai chiamato wandb.init()?")

    metadata = {
        "iteration": int(iteration),
        "best_val_loss": float(best_val_loss),
        "contra_temp_learnable": bool(getattr(cf, "contra_temp_learnable", False)) if cf is not None else None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    ckpt = {
        "iteration": int(iteration),
        "best_val_loss": float(best_val_loss),
        "text_encoder_state": text_encoder.state_dict(),
        "vision_encoder_state": vision_encoder.state_dict(),
        "shared_head_state": shared_head.state_dict() if shared_head is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "contra_temp": float(contra_temp.item()) if torch.is_tensor(contra_temp) else contra_temp,
        "config": vars(cf) if cf is not None else None,
    }

    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        torch.save(ckpt, path)
        artifact.add_file(path, name="model.pt")

        wandb.log_artifact(artifact, aliases=[artifact_alias])
