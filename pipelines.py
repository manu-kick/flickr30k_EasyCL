# pipelines.py
from analysis.viz import visualize_3d
from loss import compute_loss_anchor
from loss import *
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm

from analysis.gap_mean_differences import gap_mean_differences
from analysis.gap_embedding_dim_pairs import gap_embedding_dim_pairs
from analysis.fisher_cumulative_expl_var import fisher_and_cumulative_explained_variance
from analysis.intrinsic_dimensions import intrinsic_dimension_mle
from analysis.modality_gap import compute_gap

from utils import log_model_to_wandb, save_checkpoint


def get_loss(loss_type, text_embedding, vision_embedding, bs, contra_temp):
    if loss_type == 'anchor':
        loss = compute_loss_anchor(text_embedding, vision_embedding, bs, contra_temp)
    elif loss_type == 'centroids':
        print("loss not implemented")
        return 0
    elif loss_type == 'volume':
        print("loss not implemented")
        return 0
    elif loss_type == 'area':
        print("loss not implemented")
        return 0
    elif loss_type == 'anchor_align_unif':
        print("loss not implemented")
        return 0
    else:
        print("loss not implemented")
        return 0
    return loss


def eval(cf, test_loader, text_encoder, vision_encoder, shared_head, device, iteration, contra_temp):
    """
    Eval usa:
    - contra_temp (tensor/param) se cf.contra_temp_learnable
    - cf.contra_temp_init se non learnable
    """
    text_encoder.eval()
    vision_encoder.eval()
    if shared_head is not None:
        shared_head.eval()

    text_embeddings = []
    vision_embeddings = []

    with torch.no_grad():
        for batch in test_loader:
            images, captions, fns, cap_idxs = batch

            images = images.to(device)

            text_emb = text_encoder(captions)
            vision_emb = vision_encoder(images)

            if shared_head is not None:
                text_emb = shared_head(text_emb)
                vision_emb = shared_head(vision_emb)

            text_emb = F.normalize(text_emb, dim=-1)
            vision_emb = F.normalize(vision_emb, dim=-1)

            text_embeddings.append(text_emb.detach().cpu().numpy())
            vision_embeddings.append(vision_emb.detach().cpu().numpy())

    # numpy arrays
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    vision_embeddings = np.concatenate(vision_embeddings, axis=0)

    # analyses
    visualize_3d(cf, text_embeddings, vision_embeddings, iteration)
    gap_mean_differences(cf, text_embeddings, vision_embeddings, iteration)
    gap_embedding_dim_pairs(cf, text_embeddings, vision_embeddings, iteration)
    fisher_and_cumulative_explained_variance(cf, text_embeddings, vision_embeddings, iteration)
    intrinsic_dimension_mle(cf, text_embeddings, vision_embeddings, iteration)

    gaps = {}
    for i in ['L2M', 'RMG', 'L2I']:
        gaps[i] = compute_gap(cf, i, text_embeddings, vision_embeddings, iteration)

    # val loss su torch (device)
    text_t = torch.from_numpy(text_embeddings).to(device=device, dtype=torch.float32)
    vis_t = torch.from_numpy(vision_embeddings).to(device=device, dtype=torch.float32)

    # temperatura corretta
    temp_for_eval = contra_temp if cf.contra_temp_learnable else cf.contra_temp_init

    val_loss = get_loss(cf.loss_type, text_t, vis_t, text_t.size(0), temp_for_eval)

    print(
        f"Iteration {iteration} ==> Validation Loss = {val_loss.item():.4f} "
        f"| L2M Gap: {gaps['L2M']['text_vision']:.4f} "
        f"| RMG Gap: {gaps['RMG']['text_vision']:.4f} "
        f"| L2I Gap: {gaps['L2I']['text_vision']:.4f}"
    )

    if cf.wandb:
        wandb.log({
            "validation_loss": val_loss.item(),
            "L2M_gap": gaps['L2M']['text_vision'],
            "RMG_gap": gaps['RMG']['text_vision'],
            "L2I_gap": gaps['L2I']['text_vision'],
            "iteration": iteration,
        })

    return val_loss.item(), gaps


def train_model_with_visualization(
    cf,
    text_encoder,
    vision_encoder,
    shared_head,
    train_loader,
    test_loader,
    optimizer,
    device,
    num_iterations,
    contra_temp,
    save_local: bool = False,
    save_dir: str = "checkpoints",
    save_name: str = "best.pt",
    wandb_artifact_name: str = "best-model",
):
    """
    - salva il best su W&B Artifacts quando migliora val_loss
    - (opzionale) salva anche localmente se save_local=True
    """

    if not cf.contra_temp_learnable:
        contra_temp.requires_grad = False

    text_encoder.train()
    vision_encoder.train()
    if shared_head is not None:
        shared_head.train()

    iteration = 0
    best_val_loss = float("inf")

    # eval iniziale
    with torch.no_grad():
        val_loss, gaps = eval(
            cf, test_loader, text_encoder, vision_encoder, shared_head,
            device, iteration, contra_temp
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        # W&B artifact (versionato)
        if cf.wandb:
            log_model_to_wandb(
                text_encoder=text_encoder,
                vision_encoder=vision_encoder,
                shared_head=shared_head,
                optimizer=optimizer,
                contra_temp=contra_temp,
                iteration=iteration,
                best_val_loss=best_val_loss,
                cf=cf,
                artifact_name=wandb_artifact_name,
                artifact_alias="latest",
                extra_metadata={
                    "L2M_gap": float(gaps["L2M"]["text_vision"]),
                    "RMG_gap": float(gaps["RMG"]["text_vision"]),
                    "L2I_gap": float(gaps["L2I"]["text_vision"]),
                }
            )
            wandb.log({"best_val_loss": best_val_loss})

        # opzionale: local checkpoint
        if save_local:
            path = save_checkpoint(
                save_dir=save_dir,
                filename=save_name,
                text_encoder=text_encoder,
                vision_encoder=vision_encoder,
                shared_head=shared_head,
                optimizer=optimizer,
                contra_temp=contra_temp,
                iteration=iteration,
                best_val_loss=best_val_loss,
                cf=cf,
            )
            print(f"[CKPT] Saved local best @ {path} (val_loss={best_val_loss:.6f})")

    # torna in train
    text_encoder.train()
    vision_encoder.train()
    if shared_head is not None:
        shared_head.train()

    running_loss = 0.0
    tq = tqdm(range(num_iterations), total=num_iterations, desc="Training")
    train_iterator = iter(train_loader)

    for _ in tq:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        images, captions, fns, cap_idxs = batch
        images = images.to(device)

        # forward
        text_emb = text_encoder(captions)
        vision_emb = vision_encoder(images)

        if shared_head is not None:
            text_emb = shared_head(text_emb)
            vision_emb = shared_head(vision_emb)

        if cf.normalization:
            text_emb = F.normalize(text_emb, dim=-1)
            vision_emb = F.normalize(vision_emb, dim=-1)

        loss = get_loss(cf.loss_type, text_emb, vision_emb, text_emb.size(0), contra_temp)

        if cf.wandb:
            wandb.log({"train_loss": loss.item(), "iteration": iteration})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
        running_loss += loss.item()
        tq.set_postfix({"loss": loss.item(), "best_val": best_val_loss})

        # eval periodica
        if iteration % cf.eval_every == 0:
            with torch.no_grad():
                val_loss, gaps = eval(
                    cf, test_loader, text_encoder, vision_encoder, shared_head,
                    device, iteration, contra_temp
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"[BEST] iter={iteration} val_loss={best_val_loss:.6f}")

                # W&B artifact
                if cf.wandb:
                    log_model_to_wandb(
                        text_encoder=text_encoder,
                        vision_encoder=vision_encoder,
                        shared_head=shared_head,
                        optimizer=optimizer,
                        contra_temp=contra_temp,
                        iteration=iteration,
                        best_val_loss=best_val_loss,
                        cf=cf,
                        artifact_name=wandb_artifact_name,
                        artifact_alias="latest",
                        extra_metadata={
                            "L2M_gap": float(gaps["L2M"]["text_vision"]),
                            "RMG_gap": float(gaps["RMG"]["text_vision"]),
                            "L2I_gap": float(gaps["L2I"]["text_vision"]),
                        }
                    )
                    wandb.log({"best_val_loss": best_val_loss})

            # torna in train
            text_encoder.train()
            vision_encoder.train()
            if shared_head is not None:
                shared_head.train()

            if cf.wandb and cf.contra_temp_learnable:
                wandb.log({"contra_temp": float(contra_temp.item()), "iteration": iteration})

    epoch_loss = running_loss / max(1, iteration)
    print(f"Training completed. Average Loss: {epoch_loss:.4f}")
    if cf.wandb:
        wandb.log({"mean_train_loss": epoch_loss})

    return best_val_loss
