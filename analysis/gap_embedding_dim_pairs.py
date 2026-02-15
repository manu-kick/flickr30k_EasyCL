import torch
import matplotlib.pyplot as plt
import os
import wandb

def gap_embedding_dim_pairs(cf, text_embeddings, vision_embeddings, iterations):
    # Shows a 2d graph that plots the embedding dimensions pairs' values. The pair dimension are the ones with largest absolute gap of means between modalities.
    
    modalities_embeddings = dict()
    modalities_embeddings['text'] = torch.Tensor(text_embeddings)
    modalities_embeddings['vision'] = torch.Tensor(vision_embeddings)

    
    # The embeddings should be already normalized
    
    # Get mean of each modality
    modalities_means = {modality: embeddings.mean(dim=0, keepdim=True) for modality, embeddings in modalities_embeddings.items()}
    
    # Compute absolute difference of means between each pair of modalities
    modality_pairs = [("text", "vision")]
    diffs = dict()
    
    # Create a single figure with  subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for mod1, mod2 in modality_pairs:
        diffs[(mod1, mod2)] = torch.abs(modalities_means[mod1] - modalities_means[mod2]).squeeze(0)
        # Get top 2 dimensions with largest difference
        top2_dims = torch.topk(diffs[(mod1, mod2)], k=2).indices.tolist()
        dim1, dim2 = top2_dims[0], top2_dims[1]

        # Plot the 2D scatter plot for the pair of modalities
        ax.scatter(modalities_embeddings[mod1][:, dim1].numpy(), modalities_embeddings[mod1][:, dim2].numpy(), label=mod1, alpha=0.5)
        ax.scatter(modalities_embeddings[mod2][:, dim1].numpy(), modalities_embeddings[mod2][:, dim2].numpy(), label=mod2, alpha=0.5)

        ax.set_xlabel(f'Embedding Dimension {dim1}')
        ax.set_ylabel(f'Embedding Dimension {dim2}')
        ax.set_title(f'Dims ({dim1}, {dim2}) for {mod1} vs {mod2}')
        ax.legend()
        
    plt.suptitle(f'Embedding Dimension Pairs at iteration {iterations}', fontsize=16)
    plt.tight_layout()
    path = cf.plot_path
    os.makedirs(path, exist_ok=True)
    #create a new folder inside: path called 'gap_embedding_dims'
    path = os.path.join(path, 'gap_embedding_dims')
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f'gap_embedding_dims_all_pairs_at_{iterations}.png')
    plt.savefig(save_path)
    plt.close()
    
    if iterations%100 == 0 and cf.wandb :
        wandb.log({"embedding_dims_pairs": wandb.Image(save_path)})
        
        
        