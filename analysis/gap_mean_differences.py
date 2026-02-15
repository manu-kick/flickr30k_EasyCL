# adapted from https://github.com/lmb-freiburg/two-effects-one-trigger/tree/main/analysis


import torch
import matplotlib.pyplot as plt
import os
import wandb

def gap_mean_differences(cf,text_embeddings,vision_embeddings, iterations):
    
    # Obtain the modalities embeddings and labels
    modalities_embeddings = dict()
    modalities_embeddings['text'] = torch.Tensor(text_embeddings)
    modalities_embeddings['vision'] = torch.Tensor(vision_embeddings)
    
    # The embeddings should be already normalized
    
    # Get the mean for each modality
    modalities_means = {modality: embeddings.mean(dim=0, keepdim=True) for modality, embeddings in modalities_embeddings.items()}
    
    # Compute the absolute difference of means between each pair of modalities
    modality_pairs = [("text", "vision")]
    diffs = dict()
    sorted_idx = dict()
    sorted_diffs = dict()
    for mod1, mod2 in modality_pairs:
        diffs[(mod1, mod2)] = torch.abs(modalities_means[mod1] - modalities_means[mod2]).squeeze(0)
        sorted_idx[(mod1, mod2)] = torch.argsort(diffs[(mod1, mod2)], descending=True)
        sorted_diffs[(mod1, mod2)] = diffs[(mod1, mod2)][sorted_idx[(mod1, mod2)]]
        
    # Plot a graph on the x axes the dimension count, on the y axes the absolute difference of means, and a line for each pair of modalities
    # The goal is to show for how many dimension the difference of means is high, and for how many is low, and if there are some dimensions where the difference is particularly high, which could indicate a gap between the modalities in those dimensions.
    fig = plt.figure(figsize=(10, 6))
    for mod1, mod2 in modality_pairs:
        plt.plot(sorted_diffs[(mod1, mod2)].numpy(), label=f"{mod1} vs {mod2}")
    plt.xlabel("Embedding dimensions (sorted by difference)")
    plt.ylabel("Absolute difference of means")
    plt.title(f"Gap of means between modalities at iteration {iterations}")
    plt.legend()
    
    path = cf.plot_path
    os.makedirs(path,exist_ok=True)
    # Create a new folder inside: path called 'gap_mean_differences'
    path = os.path.join(path, 'gap_mean_differences')   
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path,f'gap means at {iterations}.png')
    plt.savefig(save_path)
    plt.close()
    
    if iterations%100 == 0 and cf.wandb :
        wandb.log({"gap_mean_differences": wandb.Image(save_path)})
    
    