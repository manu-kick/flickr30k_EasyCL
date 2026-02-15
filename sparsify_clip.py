#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import random

import torch 
import json

import torch
import os
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from typing import List, Dict
from openTSNE import TSNE
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import wandb
import time
import open_clip
import math
import umap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
#from uniformity import torch_uniformity1,torch_uniformity_equivalent,uniformity10
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")


# In[2]:
def get_beta(current_step, total_steps, warmup_epoch=20, decay_epoch=50):

    steps_in_one_epoch = total_steps / 100


    if current_step < warmup_epoch*steps_in_one_epoch:
        return 1.0
    elif current_step < (warmup_epoch+decay_epoch)*steps_in_one_epoch:
        return 1.0 - float(current_step - warmup_epoch*steps_in_one_epoch) / float(max(1, decay_epoch*steps_in_one_epoch))
    else:
        return 0.0


def get_alpha(current_step, total_steps, warmup_epoch=20, increment_epoch=50):

    steps_in_one_epoch = total_steps / 100


    if current_step < warmup_epoch*steps_in_one_epoch:
        return 1.0
    elif current_step < (warmup_epoch+increment_epoch)*steps_in_one_epoch:
        return 1.0 + float(current_step - warmup_epoch*steps_in_one_epoch) / float(max(1, increment_epoch*steps_in_one_epoch))
    else:
        return 2.0
    

def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
                                    last_epoch: int = -1, steps_sparsify: int = 462, config: dict = None) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Arguments:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    
    # Question: If we use this scheduler, the max value of lr will be the one set in the optimizer, but this means
    # that lr will be 1e-4 only for a few steps after the warmup period, but in reality we see that if we use a 
    # constant rate of 1e-4, the model performs good, so why use 1e-4 only for a few steps?
    # Cosine with restarts?

    def lr_lambda(current_step):
        # If we are using a warmup with a sparsity loss, we only want to apply the cosine schedule after 
        # the sparsity loss i.e. we want to keep the learning rate constant during the sparsity loss
        if current_step < steps_sparsify and config["only_lunif_epochs"] > 0:
            return 1.0
        elif current_step < num_warmup_steps:
            return (float(current_step) / float(max(1, num_warmup_steps))) + 1e-5
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)) 
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# In[2]:

def harmonic_loss(image_embeds, text_embeds, temperature=0.07):
    n = temperature #1#15#22#10 #22
    #CENTROID-Video Alignment
    text_embedding_exp = text_embeds.unsqueeze(1)  # Shape: (bs, 1, 10)
    vision_embedding_exp = image_embeds.unsqueeze(0)  # Shape: (1, bs, 10)


    tv = torch.norm( text_embedding_exp - vision_embedding_exp, dim=-1 )#torch.matmul(text_embedding, vision_embedding.permute(1,0))
    tv = tv ** (-n)
    tv = tv / torch.sum(tv, dim=1, keepdim=True)
    tv = -torch.log(tv)
    #compute loss by averaging on the diagonal values of tv
    loss_tv = torch.mean(torch.diagonal(tv))
    
    #print(tv)
    vt = torch.norm( text_embedding_exp - vision_embedding_exp, dim=-1 ).T#torch.matmul(vision_embedding, text_embedding.permute(1,0))
    vt = vt ** (-n)
    vt = vt / torch.sum(vt, dim=1, keepdim=True)
    vt = -torch.log(vt)

    loss_vt = torch.mean(torch.diagonal(vt))

    loss = (loss_tv + loss_vt) / 2
    #vt = vt / contra_temp
    batch_size = image_embeds.size(0)
    target = torch.arange(batch_size, device=vt.device)
    #loss_tv = (
    #        F.cross_entropy(tv, target)#, label_smoothing=0.1)
    #        + F.cross_entropy(vt, target)#, label_smoothing=0.1)
    #) / 2
    return loss

class label_smooth_loss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1, n=15):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
        self.n = n
    
    def forward(self, pred, target):
        #pred = pred.log_softmax(dim=1)
        pred = pred ** (-self.n)
        pred = pred / torch.sum(pred, dim=1, keepdim=True)


        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * pred, dim=1).mean()
    
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, n= 15):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n = n

    def forward(self, x, target):
        #logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        logprobs = x ** (-self.n)
        logprobs = logprobs / torch.sum(logprobs, dim=1, keepdim=True)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
def harmonic_loss_label_smoothing(image_embeds, text_embeds, temperature=0.07, label_smoothing=0.1):
    n = temperature
    label_smooth_loss_func = LabelSmoothing(label_smoothing,n)
    # CENTROID-Video Alignment
    text_embedding_exp = text_embeds.unsqueeze(1)  # Shape: (bs, 1, embed_dim)
    vision_embedding_exp = image_embeds.unsqueeze(0)  # Shape: (1, bs, embed_dim)



    tv = torch.norm(text_embedding_exp - vision_embedding_exp, dim=-1)

    loss_tv = label_smooth_loss_func(tv, torch.arange(tv.size(0), device=tv.device))



    vt = torch.norm(text_embedding_exp - vision_embedding_exp, dim=-1).T

    loss_vt = label_smooth_loss_func(vt, torch.arange(vt.size(0), device=vt.device))

    loss = (loss_tv + loss_vt) / 2

    return loss

class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        # Let the temperature be a learnable parameter
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # Otherwise
        # self.temperature = torch.tensor(temperature)

    def forward(self, image_features, text_features):
        # image features: [B,D]
        # text features: [B,D]

        # Normalize
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Similarity matrix and temperature scaling
        logits_per_image = image_features @ text_features.t() / self.temperature
        logits_per_text = text_features @ image_features.t() / self.temperature

        batch_size = image_features.size(0)
        labels = torch.arange(batch_size).to(image_features.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2

def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds: (batch_size, embed_dim)
    text_embeds: (batch_size, embed_dim)
    temperature: scalar float for scaling similarities
    returns: scalar loss (contrastive)
    """
    
    # Similarity matrix, shape (bs, bs)
    logits = image_embeds @ text_embeds.t()
    logits = logits / temperature

    # Targets are just the diagonal (i.e. 0->0, 1->1, ...)
    batch_size = image_embeds.size(0)
    target = torch.arange(batch_size, device=logits.device)

    # CE loss for image->text
    loss_i2t = F.cross_entropy(logits, target)
    # CE loss for text->image
    loss_t2i = F.cross_entropy(logits.t(), target)

    # Average the two directions
    return (loss_i2t + loss_t2i) / 2

def contrastive_loss_roberta(image_embeds, text_embeds, roberta_similarity, temperature=0.07):
    """
    image_embeds: (batch_size, embed_dim)
    text_embeds: (batch_size, embed_dim)
    temperature: scalar float for scaling similarities
    returns: scalar loss (contrastive)
    """
    
    # Similarity matrix, shape (bs, bs)
    logits = image_embeds @ text_embeds.t()
    logits = logits / temperature

    # Targets are just the diagonal (i.e. 0->0, 1->1, ...)
    batch_size = image_embeds.size(0)
    #target = torch.arange(batch_size, device=logits.device)

    # CE loss for image->text
    loss_i2t = F.cross_entropy(logits, roberta_similarity)
    # CE loss for text->image
    loss_t2i = F.cross_entropy(logits.t(), roberta_similarity.t())

    # Average the two directions
    return (loss_i2t + loss_t2i) / 2

def lunif_loss(x, t=2):
    # Compute pairwise distances between all embeddings
    sq_pdist = torch.pdist(x, p=2).pow(2)
    
    # Apply the uniformity loss formula
    return sq_pdist.mul(-t).exp().mean().log()

def sparsify_loss(x):
    # compute pairwise cosine similarity
    cos_sim = x @ x.T

    #matrix with 1 on the main diagonal and -1 elsewhere
    eye = torch.eye(cos_sim.size(0), device=cos_sim.device)
    eye[eye == 0] = -1


    # mse between cosine similarity and eye matrix
    return F.mse_loss(cos_sim, eye)

def random_alignment_loss(x, y):
    alpha = 2
    # randomply shuffle the y embeddings
    idx = torch.randperm(y.size(0))
    y = y[idx]

    return (x - y).norm(dim=1).pow(alpha).mean()

def lalign_loss(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()

# In[3]:


def visualize_embeddings(
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    sample_size: int = 100,
    method: str = 'pca',
    # method: str = 'umap',
    title: str = "Embeddings Visualization",
    save_path: str = None,
    labels: np.ndarray = None,          # <--- New argument (labels for each sample)
    is_cifar10: bool = False            # <--- If True, we do label-based color-coding
):
    """
    Visualizes text and vision embeddings in 2D or 3D using PCA, t-SNE, or UMAP.
    If 'labels' is provided (and is_cifar10 == True), color points by their CIFAR-10 class.

    Args:
        text_embeddings (torch.Tensor): Shape [N, D] containing text embeddings.
        vision_embeddings (torch.Tensor): Shape [N, D] containing vision/image embeddings.
        sample_size (int): If the embeddings contain more than 'sample_size' samples, 
            randomly pick this many for faster plotting. Set -1 to use all.
        method (str): "pca", "tsne", or "umap".
        title (str): Title for the plot.
        save_path (str, optional): If provided, saves the plot to this path.
        labels (np.ndarray): If provided, an array of shape [N] with numeric class labels.
        is_cifar10 (bool): If True, we apply label-based coloring for CIFAR-10.
    """

    # Detach from graph and bring to CPU
    text_np = text_embeddings.detach().cpu().numpy()
    vision_np = vision_embeddings.detach().cpu().numpy()

    # Downsample for faster plotting
    if is_cifar10:
        if sample_size != -1:
            N_text = text_np.shape[0]
            N_vis  = vision_np.shape[0]
            N_samples = min(N_text, N_vis)
            if N_samples > sample_size:
                indices = np.random.choice(N_samples, size=sample_size, replace=False)
                text_np   = text_np[indices]
                vision_np = vision_np[indices]
                # If we have labels, also select subset
                if labels is not None:
                    labels = labels[indices]

    # Combine for joint dimensionality reduction
    all_data = np.concatenate([text_np, vision_np], axis=0)
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3)
        reduced = reducer.fit_transform(all_data)
    elif method.lower() == "tsne":
        from openTSNE import TSNE
        # from sklearn.manifold import TSNE  # Alternatively
        reducer = TSNE(n_components=3, n_jobs=1, n_iter=1000)
        reduced = reducer.fit(all_data)
    elif method.lower() == "umap":
        import umap
        # reducer = umap.UMAP(n_components=3, n_jobs=8)
        # reduced = reducer.fit_transform(all_data)
        reducer = umap.UMAP(n_components=2, n_jobs=8)
        reduced = reducer.fit_transform(all_data)
    else:
        raise NotImplementedError("Only 'pca', 'tsne', and 'umap' are supported.")

    # Split back into text and vision
    text_reduced   = reduced[: len(text_np)]
    vision_reduced = reduced[len(text_np):]

    # Convert to torch Tensors
    text_reduced   = torch.tensor(text_reduced)
    vision_reduced = torch.tensor(vision_reduced)
    
    # Normalize to unit sphere
    text_reduced   = F.normalize(text_reduced, p=2, dim=1)
    vision_reduced = F.normalize(vision_reduced, p=2, dim=1)

    # Plot 3D
    fig = plt.figure(figsize=(10, 8))
    # ax  = fig.add_subplot(111, projection='3d')
    ax  = fig.add_subplot(111)
    
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

    # Draw grey lines between matching text-image pairs
    for i in range(len(text_reduced)):
        ax.plot([text_reduced[i, 0], vision_reduced[i, 0]],
                [text_reduced[i, 1], vision_reduced[i, 1]],
                # [text_reduced[i, 2], vision_reduced[i, 2]],
                c='grey', alpha=0.3, linewidth=0.5, zorder=1)

    # If we have CIFAR-10 style labels, color by class
    if is_cifar10 and labels is not None:
        import matplotlib
        unique_classes = np.unique(labels)
        cmap = matplotlib.colormaps['tab10']

        added_text_labels = set()  # Track which text class labels were added

        for c in unique_classes:
            c = int(c)
            idx_text = (labels == c)
            idx_img = (labels == c)
            color = np.array(cmap(c)).reshape(1, -1)

            # Plot text embeddings (marker 'o')
            if c not in added_text_labels:
                ax.scatter(
                    text_reduced[idx_text, 0][0],
                    text_reduced[idx_text, 1][0],
                    # text_reduced[idx_text, 2][0],
                    c=color,
                    marker='o',
                    alpha=0.95,
                    label=f'{CIFAR10_CLASSES[c]}',
                    zorder=3
                )
                added_text_labels.add(c)  # Mark this text label as added

            # Plot vision embeddings (marker '^')
            ax.scatter(
                vision_reduced[idx_img, 0],
                vision_reduced[idx_img, 1],
                # vision_reduced[idx_img, 2],
                c=color,
                marker='^',
                alpha=0.7,
                zorder=3
            )
    else:
        # Default (old) behavior: text = red, images = blue
        ax.scatter(text_reduced[:, 0], 
                   text_reduced[:, 1], 
                #    text_reduced[:, 2], 
                   c='red', alpha=0.6, label='Text', zorder=3)
        ax.scatter(vision_reduced[:, 0], 
                   vision_reduced[:, 1], 
                #    vision_reduced[:, 2], 
                   c='blue', alpha=0.6, label='Vision', zorder=3)

    # Set axis limits to ±1 for clarity if you like (optional)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    # ax.set_zlim(-1.0, 1.0)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    # ax.set_zlabel("Component 3")
    ax.legend(loc='upper left', bbox_to_anchor=(-0.3, 1.0))


    # Optional: check if saved image is a valid PNG
    def is_valid_png(image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()  
            return True
        except:
            return False

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()

        # If using Weights & Biases, log the image
        if is_valid_png(save_path):
            wandb.log({method: wandb.Image(save_path)})
        else:
            print(f"Invalid PNG file saved at {save_path}")

        # Remove the local image file
        os.remove(save_path)
    else:
        plt.show()


# In[4]:


def compute_centroids(text_embeddings, visual_embeddings):
    """
    Computes the centroid for each pair of samples between text embeddings and visual embeddings
    by calculating the mean of the corresponding feature vectors across the two modalities.

    Parameters:
    - text_embeddings (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing text embeddings.
    - visual_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing visual embeddings.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2, feature_dim) representing the centroid for each pair.
    """

    # Compute centroids by averaging text and visual embeddings
    # Expand the dimensions to allow pairwise computation
    text_expanded = text_embeddings.unsqueeze(1)  # Shape: [batch_size1, 1, feature_dim]
    visual_expanded = visual_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]

    # Compute the centroid by averaging the embeddings
    centroids = (text_expanded + visual_expanded) / 2.0

    # Compute norms of the centroids
    centroid_norms = torch.norm(centroids, dim=-1)

    return centroid_norms, centroids

def compute_centroids_only(text_embeddings, visual_embeddings):
    """
    Computes the centroid for each pair of samples between text embeddings and visual embeddings
    by calculating the mean of the corresponding feature vectors across the two modalities.

    Parameters:
    - text_embeddings (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing text embeddings.
    - visual_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing visual embeddings.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2, feature_dim) representing the centroid for each pair.
    """

    # Compute centroids by averaging text and visual embeddings
    # Expand the dimensions to allow pairwise computation
    #text_expanded = text_embeddings.unsqueeze(1)  # Shape: [batch_size1, 1, feature_dim]
    #visual_expanded = visual_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]

    # Compute the centroid by averaging the embeddings
    centroids = (text_embeddings + visual_embeddings) / 2.0

    return  centroids

def compute_metric_ret(score_matrix: torch.Tensor, ids: List[int], ids_txt: List[int], direction: str = 'forward') -> Dict[str, float]:
    """
    Compute retrieval metrics for either text-to-vision or vision-to-text retrieval.

    Args:
        score_matrix (torch.Tensor): Similarity matrix of shape [N_text, N_image].
        ids (List[int]): List of image IDs.
        ids_txt (List[int]): List of text IDs corresponding to images.
        direction (str): 'forward' for text-to-vision, 'backward' for vision-to-text.

    Returns:
        Dict[str, float]: Dictionary containing retrieval metrics.
    """
    assert score_matrix.shape == (len(ids_txt), len(ids)), f"Score matrix shape {score_matrix.shape} does not match (len(ids_txt), len(ids))"

    if direction == 'forward':  # Text-to-Vision Retrieval
        # Sort each row in descending order
        indice_matrix = score_matrix.sort(dim=-1, descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))

        rank = torch.tensor(rank).to(score_matrix.device)

        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)

        eval_log = {
            'forward_r1': round(vr_r1 * 100, 4),
            'forward_r5': round(vr_r5 * 100, 4),
            'forward_r10': round(vr_r10 * 100, 4),
            #'forward_recall': f'{round(vr_r1 * 100, 1)}/{round(vr_r5 * 100, 1)}/{round(vr_r10 * 100, 1)}',
            'forward_ravg': round((vr_r1 + vr_r5 + vr_r10) / 3 * 100, 4)
        }

    else:  # Vision-to-Text Retrieval
        # Sort each column in descending order
        indice_matrix = score_matrix.sort(dim=0, descending=True)[1].permute(1, 0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices = [idx for idx, id_txt in enumerate(ids_txt) if id_txt == ids[i]]
            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))

        rank = torch.tensor(rank).to(score_matrix.device)

        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)

        eval_log = {
            'backward_r1': round(tr_r1 * 100, 4),
            'backward_r5': round(tr_r5 * 100, 4),
            'backward_r10': round(tr_r10 * 100, 4),
            #'backward_recall': f'{round(tr_r1 * 100,1)}/{round(tr_r5 * 100,1)}/{round(tr_r10 * 100,1)}',
            'backward_ravg': round((tr_r1 + tr_r5 + tr_r10) / 3 * 100, 4)
        }

    return eval_log

def compute_clustering_metrics(feat_t: torch.Tensor, feat_v: torch.Tensor, ids_txt) :
    from pycocotools.coco import COCO

    # File paths
    instances_path = '/mnt/media/giordano/sparsify-clip/data/coco/annotations/instances_val2017.json'
    captions_path = '/mnt/media/giordano/sparsify-clip/data/coco/annotations/captions_val2017.json'

    true_labels = []

    # Load COCO APIs
    coco_instances = COCO(instances_path)
    coco_captions = COCO(captions_path)


    print(f'feat_t shape: {feat_t.shape}')
    print(f'feat_v shape: {feat_v.shape}')

    feat_t_new = []
    feat_v_new = []

    for i, id in enumerate(ids_txt):
        print(id)

        # --- Retrieve OBJECTS ---
        ann_ids = coco_instances.getAnnIds(imgIds=id)
        anns = coco_instances.loadAnns(ann_ids)


        local_ids = set([ann['category_id'] for ann in anns])
        local_labels = []
        #for local_id in local_ids:
        #    store = 0
        #    
        #    if local_id in categories_id:
    #
        #        local_labels.append(local_id)

        if len(local_ids) == 1:
            true_labels.append(list(local_ids)[0])
            feat_t_new.append(feat_t[i])
            feat_v_new.append(feat_v[i])
        else:
            print("More than one object in image", id, ":", local_ids)
            # If you want to handle multiple objects, you can modify this logic
            # For now, we skip this image
            continue


        
        #if len(local_labels) > 0:
        #    # randopmly select one label from local_labels
        #    local_label = np.random.choice(local_labels, 1, replace=False)[0]
        #    true_labels.append(local_label)
        #    feat_t_new.append(feat_t[i])
        #    feat_v_new.append(feat_v[i])




        #print([ann['category_id'] for ann in anns])
        #object_names = [coco_instances.loadCats(ann['category_id'])[0]['name'] for ann in anns]
        #print("Objects in image:", list(set(object_names)))  # Remove duplicates

    feat_t_new = torch.stack(feat_t_new)
    feat_v_new = torch.stack(feat_v_new)

    print(f'feat_t_new shape: {feat_t_new.shape}')
    print(f'feat_v_new shape: {feat_v_new.shape}')
    print("True labels:", true_labels)


    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=10, random_state=0)
    cluster_labels = kmeans.fit_predict(torch.vstack((feat_t_new, feat_v_new)))

    cluster_labels.shape

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
    from sklearn.metrics import v_measure_score


    true_labels_new = true_labels * 2
    ari = adjusted_rand_score(true_labels_new, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_new, cluster_labels)
    hom = homogeneity_score(true_labels_new, cluster_labels)

    preds = kmeans.labels_


    v = v_measure_score(true_labels_new, cluster_labels)

    print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Homogeneity: {hom:.4f}, V-measure: {v:.4f}")


    embeddings = torch.vstack((feat_t_new, feat_v_new))
    # Get unique labels and create mapping to consecutive integers
    unique_labels = sorted(list(set(true_labels_new)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

    # Apply mapping to create consecutive labels from 0 to num_classes-1
    full_labels = torch.tensor([label_mapping[label] for label in true_labels_new], dtype=torch.long)



    D = embeddings.shape[1]
    num_classes = len(torch.unique(full_labels))

    # Custom dataset
    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    dataset = EmbeddingDataset(embeddings, full_labels)

    # Train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    # Linear classifier model
    class LinearProbe(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.fc(x)

    model = LinearProbe(D, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_losses, test_accuracies = [], []

    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        test_accuracies.append(acc)
        #print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {acc:.4f}")

    # k-NN evaluation
    X_train = embeddings[train_set.indices].numpy()
    y_train = full_labels[train_set.indices].numpy()
    X_test = embeddings[test_set.indices].numpy()
    y_test = full_labels[test_set.indices].numpy()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)

    print("\nK-NN Classification Report:")
    print(classification_report(y_test, knn_preds))
    print(f"max accuracy: {max(test_accuracies)}")

    return {
        "ARI": ari,
        "NMI": nmi,
        "Homogeneity": hom,
        "V-measure": v,
        "Max Linear Probe Acc": max(test_accuracies),
        "K-NN Acc": knn_acc
    }


def compute_gap(feat_modality1: torch.Tensor, feat_modality2: torch.Tensor) -> float:
    """
    Compute the Euclidean distance between the centroids of two modalities.

    Args:
        feat_modality1 (torch.Tensor): Feature matrix of modality 1 with shape [N, D].
        feat_modality2 (torch.Tensor): Feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Euclidean distance between centroids.
    """
    # Ensure features are normalized if required
    modality1_centroid = torch.mean(feat_modality1, dim=0)
    modality2_centroid = torch.mean(feat_modality2, dim=0)

    gap = modality1_centroid - modality2_centroid
    norm_gap = torch.norm(gap).item()

    return norm_gap

def compute_mean_angular_value_of_a_modality(feat_modality: torch.Tensor) -> float:
    """
    Compute the mean angular value (mean cosine similarity) of a modality.

    Args:
        feat_modality (torch.Tensor): Feature matrix with shape [N, D].

    Returns:
        float: Mean angular value.
    """
    # Compute cosine similarity matrix
    cos_sim = feat_modality @ feat_modality.T

    # Exclude diagonal elements by creating a mask
    mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
    cos_sim_no_diag = cos_sim[mask]

    mean_cos_sim = cos_sim_no_diag.mean().item()

    return mean_cos_sim

def uniformity(features_modality1: torch.Tensor, features_modality2: torch.Tensor) -> float:
    x = torch.cat([features_modality1, features_modality2], dim=0)
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x - x_center).t(), x - x_center) / N

    mean =  x.mean(0)
    np_mean = mean.data.cpu().numpy()
    np_covariance = covariance.data.cpu().numpy()
   
    ##calculation of part1
    part1 = np.sum(np.multiply(np_mean, np_mean))

    ##calculation of part2
    eps = 1e-8 
    S, Q = np.linalg.eig(np_covariance)
    S = S + eps

    mS = np.sqrt(np.diag(S.clip(min=0)))

    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return -wasserstein_distance 

def centroid_alignment_loss(img_embeds: torch.Tensor, txt_embeds: torch.Tensor, p=2) -> torch.Tensor:
    """
    Compute the distance between the mean image embedding and the mean text embedding.

    Args:
        img_embeds (torch.Tensor): Image embeddings of shape (batch_size, embed_dim).
        txt_embeds (torch.Tensor): Text embeddings of shape (batch_size, embed_dim).
        p (int): Norm order (2 for Euclidean / L2 norm).

    Returns:
        torch.Tensor: A scalar tensor representing the centroid alignment penalty.
    """
    # Compute centroids along the batch dimension
    centroid_img = img_embeds.mean(dim=0)  # shape (embed_dim,)
    centroid_txt = txt_embeds.mean(dim=0)  # shape (embed_dim,)

    # Compute the L2 distance (default) between the centroids
    dist = torch.norm(centroid_img - centroid_txt, p=p)
    return dist

def mean_distance_of_true_pairs(features_modality1: torch.Tensor, features_modality2: torch.Tensor, cosine = True) -> float:
    """
    Compute the mean cosine similarity of true pairs between two modalities.

    Args:
        features_modality1 (torch.Tensor): Normalized feature matrix of modality 1 with shape [N, D].
        features_modality2 (torch.Tensor): Normalized feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Mean cosine similarity of true pairs.
    """
    # Compute cosine similarity matrix
    if cosine:
        cosine_sim = torch.matmul(features_modality1, features_modality2.T)

        # Extract diagonal elements (true pairs)
        cosine_sim_diag = torch.diag(cosine_sim)

        # Compute mean cosine similarity of true pairs
        cosine_tv_mean = torch.mean(cosine_sim_diag).item()

        return cosine_tv_mean

    else:
        # Compute Euclidean distance matrix
        euclidean_dist = torch.cdist(features_modality1, features_modality2)

        # Extract diagonal elements (true pairs)
        euclidean_dist_diag = torch.diag(euclidean_dist)

        # Compute mean Euclidean distance of true pairs
        euclidean_tv_mean = torch.mean(euclidean_dist_diag).item()

        return euclidean_tv_mean


# In[5]:


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, plot_embeddings=True) -> Dict[str, float]:
    """
    Evaluate the (OpenCLIP) model on the given test_loader by computing
    text-to-image and image-to-text retrieval metrics, along with additional metrics.

    Args:
        model (torch.nn.Module): The trained (DataParallel) model.
        test_loader (DataLoader): A DataLoader for the evaluation set.
        device (torch.device): The device (CPU or GPU).

    Returns:
        Dict[str, float]: Dictionary containing all evaluation metrics.
    """
    # Put model into eval mode
    model.eval()

    # Prepare storage for embeddings
    all_image_embeds = []
    all_text_embeds  = []
    all_labels       = []

    # IDs for retrieval
    ids_img = []
    ids_txt = []

    current_index = 0
    
    tokenizer = open_clip.get_tokenizer('RN50')
    
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # No gradient needed during evaluation
    with torch.no_grad():
        for images, captions_list, sample_ids in (tqdm(test_loader, desc="Evaluating") if plot_embeddings else test_loader):
            # Move images to device
            images = images.to(device)
            
            # Convert numerical labels to text class names (only for CIFAR-10)
            if isinstance(captions_list[0], int) or torch.is_tensor(captions_list[0]):
                # Convert numeric label to textual class name
                numeric_labels = captions_list
                captions_list = [CIFAR10_CLASSES[int(lbl)] for lbl in numeric_labels]
                # Store the numeric labels for color-coding
                all_labels.extend(numeric_labels)
            else:
                # For non-CIFAR dataset, all_labels can remain empty or zero-based
                # The code below simply won't color by label if labels is empty.
                numeric_labels = [0]*len(captions_list)  # or skip altogether

            # Tokenize captions    
            text_tokens = tokenizer(captions_list).to(device)
            
            # Extract embeddings using the .module references in DataParallel
            image_embeds = model.module.encode_image(images)
            text_embeds = model.module.encode_text(text_tokens)

            # Move embeddings to CPU for later concatenation
            image_embeds = image_embeds.cpu()
            text_embeds = text_embeds.cpu()

            # Store embeddings
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)

            # Assign unique IDs
            bs = images.size(0)
            #sample_ids = list(range(current_index, current_index + bs))
            ids_img.extend(sample_ids)
            ids_txt.extend(sample_ids)
            current_index += bs

    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # Shape: [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)    # Shape: [N, D]
    all_labels = np.array(all_labels)
    
    # Time taken for UMAP visualization:  7.042090892791748
    # Time taken for TSNE visualization:  51.96275329589844
    # Time taken for PCA visualization:  0.8503968715667725
    
    # If we are working with cifar10, create a true variable
    
    if plot_embeddings:
        '''visualize_embeddings(all_text_embeds, 
                                all_image_embeds, 
                                sample_size=500, 
                                method='umap', 
                                title="CLIP Embeddings Visualization",
                                save_path=f"plots/embeddings_plot_umap_{time.time()}.png",
                                labels=all_labels,
                                is_cifar10=True)'''


        '''visualize_embeddings(all_text_embeds, 
                                all_image_embeds, 
                                sample_size=500, 
                                method='tsne',
                                title="CLIP Embeddings Visualization",
                                save_path=f"plots/embeddings_plot_tsne_{time.time()}.png",
                                labels=all_labels,
                                is_cifar10=False)'''
        
        visualize_embeddings(all_text_embeds, 
                                all_image_embeds, 
                                sample_size=100, 
                                method='umap',
                                title="CLIP Embeddings Visualization",
                                save_path=f"plots/embeddings_plot_pca_{time.time()}.png",
                                labels=all_labels,
                                is_cifar10=False)


    #all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
    #all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)

    # Compute pairwise similarity: [N_text, N_image]
    if config["loss_type"] == "harmonic":
        text_embedding_exp = all_text_embeds.unsqueeze(1)  # Shape: (bs, 1, 10)
        vision_embedding_exp = all_image_embeds.unsqueeze(0)  # Shape: (1, bs, 10)
        similarity_matrix = -torch.norm( text_embedding_exp.to(device) - vision_embedding_exp.to(device), dim=-1 )#torch.matmul(text_embedding, vision_embedding.permute(1,0))
    else:
            # Normalize embeddings to map the embeddings in a sphere of radius 1
        all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
        all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)

        similarity_matrix = torch.matmul(all_text_embeds.to(device), all_image_embeds.t().to(device))


    """SEQUENTIAL COMPUTATION
    # Compute retrieval and additional metrics
    log_forward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='forward')   # Text-to-Vision
    log_backward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='backward') # Vision-to-Text
    gap = compute_gap(all_image_embeds, all_text_embeds)
    mean_ang_image = compute_mean_angular_value_of_a_modality(all_image_embeds)
    mean_ang_text = compute_mean_angular_value_of_a_modality(all_text_embeds)
    uniformity_metric = uniformity(all_image_embeds, all_text_embeds)
    mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds)
    """
    
    def compute_metrics(all_image_embeds, all_text_embeds, similarity_matrix, ids_img, ids_txt):
        if config["loss_type"] == "harmonic":
            mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds, cosine=False)
        else:
            mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds)

        all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
        all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)

        log_forward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='forward')
        log_backward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='backward')
        gap = compute_gap(all_image_embeds, all_text_embeds)
        mean_ang_image = compute_mean_angular_value_of_a_modality(all_image_embeds)
        mean_ang_text = compute_mean_angular_value_of_a_modality(all_text_embeds)
        uniformity_metric = uniformity(all_image_embeds, all_text_embeds)

        clustering_metrics = compute_clustering_metrics(all_text_embeds, all_image_embeds, ids_txt)

        return log_forward, log_backward, gap, mean_ang_image, mean_ang_text, uniformity_metric, mean_cos_true_pairs, clustering_metrics

    #with ThreadPoolExecutor() as executor:
    #    metrics = executor.submit(compute_metrics)
    #    log_forward, log_backward, gap, mean_ang_image, mean_ang_text, uniformity_metric, mean_cos_true_pairs = metrics.result()
    log_forward, log_backward, gap, mean_ang_image, mean_ang_text, uniformity_metric, mean_cos_true_pairs, clustering_metrics = compute_metrics(all_image_embeds, all_text_embeds, similarity_matrix, ids_img, ids_txt)


    # Combine all metrics into final_log
    final_log = {
        **log_forward,
        **log_backward,
        'gap': round(gap, 4),
        'mean_angular_value_image': round(mean_ang_image, 4), # round to 4 decimal places
        'mean_angular_value_text': round(mean_ang_text, 4),
        'uniformity': round(uniformity_metric, 4),
        'mean_cosine_similarity_true_pairs': round(mean_cos_true_pairs, 4),
        **clustering_metrics
    }

    if plot_embeddings:
        print("Evaluation Results:", final_log)
        print()
    
    wandb.log(final_log)

    model.train()
    return final_log


# In[6]:


def train_model(config, train_loader, test_loader, device):

    # Create model & transforms from scratch (no pretrained weights)
    model, _, preprocess = open_clip.create_model_and_transforms(
        config["model"],
        pretrained=None,
        device=device
    )
    
    # Get the tokenizer from the model
    tokenizer = open_clip.get_tokenizer(config["model"])
    
    # Put the model into training mode
    model.train()

    # Require gradients for all parameters to train from scratch
    for param in model.parameters():
        param.requires_grad = True
        
    # Move the model to given device
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Set up training parameters
    lr = config["learning_rate"]
    epochs = config["epochs"]
    temperature = config["anchor_temperature"]
    start_epoch = 0

    # Load the roberta model for anchor-roberta loss
    if config["loss_type"] == "anchor-roberta":
        roberta = SentenceTransformer('stsb-roberta-large').to(device)
    
    # Set up learnable temperature if required
    if config["anchor_temperature_learnable"]:
        temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True)

    loss_fn = ClipLoss(temperature=temperature)

    # Load checkpoint if resuming
    if config["resume_checkpoint"]:
        print(f"Resuming training from {config['resume_checkpoint']} at epoch {config['resume_epoch']}")
        checkpoint = torch.load(config["resume_checkpoint"])
        model.load_state_dict(checkpoint)
        start_epoch = config["resume_epoch"]

    # Set up the parameters and optimizer 
    parameters = [x for x in model.parameters()] #list(model.parameters())
    if config["anchor_temperature_learnable"]:
        print("Using learnable temperature parameter")
        #parameters.append(temperature)
        parameters = [x for x in model.parameters()]+[x for x in loss_fn.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    scaler = torch.GradScaler("cuda") if config["fp16"] else None
    
    # Set up the learning rate scheduler as 20% warmup
    t_total = len(train_loader) * config["epochs"]
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total, config=config)

    # Make a prior evaluation of the model
    print("Evaluating model before training...")
    evaluate_model(model, test_loader, device)

    # BETA init for EXP 7-8-9-10
    beta = 0.0
    alpha = 0.0
    
    # Record start time
    start_time = time.time()
    remaining_time_formatted = "00:00:00"
    
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    current_batch, loss = 0, 0
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        for images, captions_list, sample_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, ETA: {remaining_time_formatted}"):

            current_batch += 1
                        
            # Move data to the primary device
            images = images.to(device)
            captions = captions_list

            #print(f"Processing batch {current_batch} with {len(captions)} samples")

            #print(f"images shape: {images.shape}")
            #print(f"captions: {captions}")

            #if isinstance(captions[0], int) or torch.is_tensor(captions[0]):
            #    captions = [CIFAR10_CLASSES[label] for label in captions]
            
            # Tokenize text
            text_tokens = tokenizer(captions)
            text_tokens = text_tokens.to(device)
            
            with torch.autocast(device_type="cuda", enabled=config["fp16"]):

                # Encode image and text
                image_embeds = model.module.encode_image(images)  # Use .module for methods inside DataParallel
                text_embeds = model.module.encode_text(text_tokens)
                
                
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)


                # EXP 1 AND EXP 2
                if config["loss_type"] == "anchor":
                    #if epoch < config["only_lunif_epochs"]:
                    #    #print(f"Used only lunif loss for epoch {epoch}, batch {current_batch}")
                    #    loss = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                    #else:
                        #print(f"Used only anchor loss for epoch {epoch}, batch {current_batch}")
                    #loss = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    loss = loss_fn(image_embeds, text_embeds)
                
                # EXP 3 AND EXP 5
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+lunif(text)+lunif(img)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                        lalign = lalign_loss(image_embeds, text_embeds)
                        lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                        loss = anchor + lunif + lalign
                
                # EXP 4 AND EXP 6
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+lunif(centroids)":
                
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)

                        centroids = compute_centroids_only(image_embeds, text_embeds)
                        centroids = F.normalize(centroids, dim=-1)
                        lunif_centroids = lunif_loss(centroids)
                        
                        lalign = lalign_loss(image_embeds, text_embeds)

                        loss =  anchor + config["lambda1"] * lalign + config["lambda2"] * lunif_centroids




                # EXP 7
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+BETA*lunif(centroids)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)

                        lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
                        
                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)
                        
                        loss =  anchor + lalign + beta * lunif
  
                
                # EXP 8
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+BETA*lunif(centroids)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)

                        centroids = compute_centroids_only(image_embeds, text_embeds)
                        centroids = F.normalize(centroids, dim=-1)
                        lunif_centroids = lunif_loss(centroids)
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
                        
                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)
                        
                        loss =  anchor + lalign + beta * lunif_centroids

                # EXP 9
                elif config["loss_type"] == "only_lunif_n_then_anchor+ALPHA*lalign+BETA*(lunif(text)+lunif(img))":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)

                        lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
                        
                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)

                        alpha_warmup_epoch = config["alpha_warmup_epoch"]
                        alpha_increment_epoch = config["alpha_increment_epoch"]

                        alpha = get_alpha(current_batch,t_total,alpha_warmup_epoch,alpha_increment_epoch)
                        
                        loss =  anchor + alpha * lalign + beta * lunif
  
                
                # EXP 10
                elif config["loss_type"] == "only_lunif_n_then_anchor+ALPHA*lalign+BETA*lunif(centroids)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)

                        centroids = compute_centroids_only(image_embeds, text_embeds)
                        centroids = F.normalize(centroids, dim=-1)
                        lunif_centroids = lunif_loss(centroids)
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
                        
                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)

                        alpha_warmup_epoch = config["alpha_warmup_epoch"]
                        alpha_increment_epoch = config["alpha_increment_epoch"]

                        alpha = get_alpha(current_batch,t_total,alpha_warmup_epoch,alpha_increment_epoch)
                        
                        loss =  anchor + alpha * lalign + beta * lunif_centroids
            
                ###################################
                # ABLATION STUDIES BASED ON EXP 4
                ##################################
                
                # COMPLETE LOSS: ANCHOR(IMAGE,TEXT) + LALIGN(IMAGE,TEXT) + LUNIF(CENTROIDS)
                elif config["loss_type"] == "ANCHOR(IMAGE,TEXT)+LALIGN(IMAGE,TEXT)+LUNIF(CENTROIDS)":
            
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    
                    lalign = lalign_loss(image_embeds, text_embeds)

                    centroids = compute_centroids_only(image_embeds, text_embeds)
                    centroids = F.normalize(centroids, dim=-1)
                    lunif_centroids = lunif_loss(centroids)
                    
                    loss =  anchor + lalign + lunif_centroids
                
                # ABLATATION 1: ANCHOR(IMAGE,TEXT) + LALIGN(IMAGE,TEXT)
                elif config["loss_type"] == "ANCHOR(IMAGE,TEXT)+LALIGN(IMAGE,TEXT)":
                    
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    lalign = lalign_loss(image_embeds, text_embeds)
                    
                    loss =  anchor + lalign
                
                # ABLATION 2: ANCHOR(IMAGE,TEXT) + LUNIF(CENTROIDS)
                elif config["loss_type"] == "ANCHOR(IMAGE,TEXT)+LUNIF(CENTROIDS)":
                    
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    
                    centroids = compute_centroids_only(image_embeds, text_embeds)
                    centroids = F.normalize(centroids, dim=-1)
                    lunif_centroids = lunif_loss(centroids)
                    
                    loss =  anchor + lunif_centroids   

                elif config["loss_type"] == "only_lunif_n_+lalign+lunif(centroids)":
                    
                    centroids = compute_centroids_only(image_embeds, text_embeds)
                    centroids = F.normalize(centroids, dim=-1)
                    lunif_centroids = lunif_loss(centroids)
                    
                    lalign = lalign_loss(image_embeds, text_embeds)
                    
                    loss =  lalign + lunif_centroids
                
                                
                    
            # Track useful metrics
            if config["anchor_temperature_learnable"]:
                wandb.log({"train_loss": loss.item(),
                           "constrantive_temperature_learnable": loss_fn.temperature.item(),
                           "learning_rate": scheduler.get_last_lr()[0]})
                           
            else:
                wandb.log({"train_loss": loss.item(),
                            #"learning_rate": scheduler.get_last_lr()[0],
                            "beta": beta,
                            "alpha": alpha})
            # Evaluate the model every n batches
            #if current_batch % 100 == 0:
            #    evaluate_model(model, test_loader, device, plot_embeddings=False)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Backprop
            if config["fp16"]:
                scaler.scale(loss).backward()
                # Add gradient clipping
                #torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Add gradient clipping
                #torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate elapsed and estimated total time
            elapsed_time = time.time() - start_time
            progress = (current_batch + epoch * len(train_loader)) / t_total
            remaining_time = elapsed_time * (1 - progress) / progress if progress > 0 else 0

            # Format time for display
            remaining_time_formatted = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            
        
        evaluate_model(model, test_loader, device)
            
        if (epoch+1) % config["save_checkpoint_every_n_epochs"]  == 0:
            torch.save(model.state_dict(), f"models/{config['run_name']}_epoch_{epoch+1}.pt")
            print(f"Model saved at epoch {epoch+1}")
        
    return model


# In[7]:

def get_cifar10_dataloaders(cf, batch_size=128, num_workers=4, data_root='./data'):
    # Get the CIFAR-10 dataset and dataloaders
    
    # Define transformations
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    #])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 images to match CLIP's expected size
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))  # CLIP ImageNet normalization
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Print the len of the dataloaders
    print(f"Train Dataloader samples = {len(train_loader)*batch_size}")
    print(f"Test Dataloader samples = {len(test_loader)*batch_size}")
    
    return train_loader, test_loader


class CocoCaptionsWithIDs(dset.CocoCaptions):
    def __getitem__(self, index):
        image, captions = super().__getitem__(index)
        sample_id = self.ids[index]  # COCO image ID
        return image, captions, sample_id

def get_coco_dataloaders(config):

    # Path to train images and annotations
    train_image_dir = '/home/giordano/coco/images/train2017/'                          # Path to train2017 images
    train_annotation_file = '/home/giordano/coco/annotations/captions_train2017.json'  # Path to train2017 captions

    # Path to test (val) images and annotations
    test_image_dir = '/home/giordano/coco/images/val2017/'                          # Path to val2017 images
    test_annotation_file = '/home/giordano/coco/annotations/captions_val2017.json'  # Path to val2017 captions
    
    # Fixed mean and std for the dataset
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    # Define the transform to be applied to the images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),  # Resize the image to the model's required input size
        transforms.RandomHorizontalFlip(),         # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create the training dataset
    train_coco = CocoCaptionsWithIDs(
        root=train_image_dir,
        annFile=train_annotation_file,
        transform=train_transform
    )

    # Create the test dataset
    test_coco = CocoCaptionsWithIDs(
        root=test_image_dir,
        annFile=test_annotation_file,
        transform=test_transform
    )
    
    if config["num_train_samples"] != -1:
        print(f"Subsetting the training dataset to {config['num_train_samples']} samples")
        # Subset the training dataset
        num_training_samples = config["num_train_samples"]
        subset_indices = list(range(num_training_samples))
        train_coco = Subset(train_coco, subset_indices)
    
    if config["num_test_samples"] != -1:
        print(f"Subsetting the test dataset to {config['num_test_samples']} samples")
        # Subset the test dataset
        num_test_samples = config["num_test_samples"]
        subset_indices = list(range(num_test_samples))
        test_coco = Subset(test_coco, subset_indices)

    # Every image has 5 captions at max, we need to sample one of them
    # Create collate function to sample one caption per image
    def collate_fn(batch):
        images, captions, sample_ids = zip(*batch)
        images = torch.stack(images, 0)
        sel_captions = []
        for list_captions in captions:
            caption = random.choice(list_captions)
            sel_captions.append(caption)
        return images, sel_captions, sample_ids

    # Create DataLoader
    train_batch_size = config["batch_size"]
    test_batch_size = config["batch_size"]
    train_loader = DataLoader(train_coco, batch_size=train_batch_size, shuffle=True , drop_last=True, collate_fn=collate_fn, num_workers=8)
    test_loader  = DataLoader(test_coco , batch_size=test_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=0)
    
    return train_loader, test_loader


# In[8]:


def set_seed(seed: int):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU random numbers
    torch.cuda.manual_seed(seed)  # PyTorch GPU random numbers for a single GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random numbers for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.benchmark = False  # Disable benchmark for deterministic behavior


# In[9]:


def main(config):

    # Initialize your W&B run
    wandb.init(project=config["project_name"], config=config, name=config['run_name']) #, name=f"lambda1_{config['lambda1']}_lambda2_{config['lambda2']}")

    # Set the seed for reproducibility
    set_seed(config["seed"])
    
    # Print the config
    print("Config:", config)
    
    # Print the experiment name
    print("Experiment:", config["run_name"])
    
    # Set the device
    device_id = config["device_id"]
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    print(f"\nLoading the dataset {config['dataset']}...")
    if config["dataset"] == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(config)
    elif config["dataset"] == "coco":
        train_loader, test_loader = get_coco_dataloaders(config)
    print("Dataset loaded.\n")
    
    # Train the model
    print("Training the model...")
    model = train_model(config, train_loader, test_loader, device)
    print("Training complete.\n")
    
    # Final evaluation of the model
    print("Final evaluation of the model...")
    final_log = evaluate_model(model, test_loader, device)
    print("Evaluation complete.\n")
    print("Final evaluation results:", final_log)
    
    # Save the model and upload it to W&B
    torch.save(model.state_dict(), "models/" + config['run_name'] + ".pt")
    #wandb.save(config["run_name"] + ".pt")    
    
    wandb.finish()


# In[ ]:


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the experiment with a config.yaml file")
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml config file or to a folder containing multiple config files")
    parser.add_argument("--device", type=int, required=True, help="GPU id to use")
    args = parser.parse_args()
    
    # Load the config file provided from the command line if the path is a file
    if os.path.isfile(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            # Set the device id

        # lambda1_list = [0.0,0.5,1.0] 
        # lambda2_list = [1.0]   
        # for lambda1 in lambda1_list:
        #     for lambda2 in lambda2_list:
        #         config["lambda1"] = lambda1
        #         config["lambda2"] = lambda2

        #         print(f"Starting experiment with lambda1={lambda1} and lambda2={lambda2}")
        #         # Set the device id
        #         config["device_id"] = args.device
        #         # Convert learning rate to float
        #         config["learning_rate"] = float(config["learning_rate"])

        #          # Start the experiment
        #         main(config)
        config["device_id"] = args.device
        # Convert learning rate to float
        config["learning_rate"] = float(config["learning_rate"])
        # Start the experiment
        config["lambda1"] = 1.0
        config["lambda2"] = 0.0
        main(config)
    # Load all config files in the folder if the path is a folder
    # elif os.path.isdir(args.config):
    #     config_files = [f for f in os.listdir(args.config) if f.endswith(".yaml")]
    #     for file in config_files:
    #         with open(os.path.join(args.config, file), 'r') as f:
    #             config = yaml.safe_load(f)
    #             # Set the device id
    #             config["device_id"] = args.device
    #             # Convert learning rate to float
    #             config["learning_rate"] = float(config["learning_rate"])
    #             # Start the experiment
    #             main(config)
    
    


    

# In[ ]:
    

"""

config = {
    #"resume_checkpoint": "models/model_RN50_anchor+lunif_2025-01-06-23-02-32_epoch_20.pt",
    #"resume_epoch": 20,
    #"run_id": "kx8s0k7i",
    #"resume": "must",
}

# NOT USED IN THE EXPERIMENTS
                elif config["loss_type"] == "lunif":
                    loss = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                
                elif config["loss_type"] == "anchor+lunif+centroids":
                    lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                    centroid_loss = centroid_alignment_loss(image_embeds, text_embeds)
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    loss = anchor + lunif + centroid_loss
                
                elif config["loss_type"] == "alternate":
                    if current_batch % 2 == 0:
                        loss = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    else:
                        loss = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                
                elif config["loss_type"] == "anchor+lunif":
                    lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds) ) / 2
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    loss = anchor + lunif
                
                elif config["loss_type"] == "anchor-roberta":
                    encodings = roberta.encode(list(captions), convert_to_tensor=True, show_progress_bar = False).to(device)
                    feat_roberta = encodings / encodings.norm(dim=-1, keepdim=True) #torch.nn.functional.normalize(encodings,dim=-1)
                    roberta_similarity = torch.matmul(feat_roberta, feat_roberta.permute(1,0))
                    roberta_similarity = roberta_similarity / 0.1
                    roberta_similarity = F.softmax(roberta_similarity, dim=-1)
                    loss = contrastive_loss_roberta(image_embeds, text_embeds, roberta_similarity, temperature=temperature)
                
                elif config["loss_type"] == "anchor+lunif_decaying":
                    decay_factor = 1 - (current_batch / (len(train_loader))) # or len(train_loader) * epochs
                    lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds) ) / 2
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    loss = anchor + decay_factor * lunif
                
                elif config["loss_type"] == "lunif_n_batch+frozen(text_embed)":
                    if current_batch <= config["lalign_n_batch"]:
                        loss = lalign_loss(image_embeds, text_embeds)

                    elif current_batch <= config["lunif_n_batch"]:
                    #if epoch < config["lunif_n_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        #lunif_img = sparsify_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        #lunif_txt = sparsify_loss(text_embeds)

                        #lunif_img = uniformity10(image_embeds.cuda())
                        #lunif_txt = uniformity10(text_embeds.cuda())
                        #lunif_combined = uniformity10(torch.cat([image_embeds, text_embeds], dim=0).cuda())
                        #lunif_combined = lunif_loss(torch.cat([image_embeds, text_embeds], dim=0))
                        #lunif_combined = sparsify_loss(torch.cat([image_embeds, text_embeds], dim=0))
                        #if current_batch<=100:
                        #    loss = lalign(image_embeds, text_embeds)
                        #else:
                        #    loss = lunif_img + lunif_txt

                        #r_align = random_alignment_loss(image_embeds, text_embeds)
                        #align = lalign(image_embeds, text_embeds)
                        loss = (lunif_img + lunif_txt )/2
                        #loss = lunif + r_align
                        #loss = align
                    else: # train on anchor loss with frozen text embeddings
                        #text_embeds = text_embeds.detach()
                        loss = contrastive_loss(image_embeds, text_embeds, temperature=temperature)#+0.1*lalign(image_embeds, text_embeds)
                
"""


# OLD VISUALIZATION FUNCTION
"""
def visualize_embeddings(text_embeddings, vision_embeddings, 
                         sample_size=1000, method='pca', 
                         title="Embeddings Visualization",
                         save_path=None):
    '''
    Visualizes text and vision embeddings in 2D or 3D using PCA, t-SNE, or UMAP.

    Args:
        text_embeddings (torch.Tensor): Shape [N, D] containing text embeddings.
        vision_embeddings (torch.Tensor): Shape [N, D] containing vision/image embeddings.
        sample_size (int): If the embeddings contain more than 'sample_size' samples, 
            randomly pick this many for faster plotting. Set -1 to use all.
        method (str): "pca", "tsne", or "umap".
        title (str): Title for the plot.
        save_path (str, optional): If provided, saves the plot to this path instead of showing it.
    '''
    # Detach from graph and bring to CPU if the tensors require grad
    text_np = text_embeddings.detach().cpu().numpy()
    vision_np = vision_embeddings.detach().cpu().numpy()

    # Optionally downsample for quicker plotting
    if sample_size != -1:
        n_text = text_np.shape[0]
        n_vision = vision_np.shape[0]

        n_samples = min(n_text, n_vision)

        if n_samples > sample_size:
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            text_np = text_np[indices]
            vision_np = vision_np[indices]

    # Combine for joint dimensionality reduction
    all_data = np.concatenate([text_np, vision_np], axis=0)
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=3)
        reduced = reducer.fit_transform(all_data)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=3, n_jobs=1, n_iter=1000)
        reduced = reducer.fit(all_data)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=3, n_jobs=8)
        reduced = reducer.fit_transform(all_data)
    else:
        raise NotImplementedError("Only 'pca', 'tsne', and 'umap' are implemented.")

    # Split back into text and vision
    text_reduced = reduced[: len(text_np)]
    vision_reduced = reduced[len(text_np):]
    
    # Transform to numpy array
    text_reduced = np.array(text_reduced)
    vision_reduced = np.array(vision_reduced)
    
    # Transform to torch tensor
    text_reduced = torch.tensor(text_reduced)
    vision_reduced = torch.tensor(vision_reduced)
    
    # Normalize the reduced embeddings
    text_reduced = F.normalize(text_reduced, dim=-1)
    vision_reduced = F.normalize(vision_reduced, dim=-1)

    # Plot 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(text_reduced[:, 0], text_reduced[:, 1], text_reduced[:, 2], 
               c='red', alpha=0.6, label='Text')
    ax.scatter(vision_reduced[:, 0], vision_reduced[:, 1], vision_reduced[:, 2], 
               c='blue', alpha=0.6, label='Vision')

    # set the axis limits to 1.0 and -1.0
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    
    
    


    

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()
    
    def is_valid_png(image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify if the image is intact
            return True
        except (SyntaxError, IOError, OSError):
            return False

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        if is_valid_png(save_path):
            wandb.log({method: wandb.Image(save_path)})
        else:
            print(f"Invalid PNG file saved at {save_path}")
        
        # Remove the saved plot from disk
        os.remove(save_path)
                
    else:
        plt.show()
"""