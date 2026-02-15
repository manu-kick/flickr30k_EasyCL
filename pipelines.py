from analysis.viz import visualize_3d
from loss import compute_loss_anchor
from loss import *
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from loss import compute_loss_anchor
from tqdm import tqdm
from analysis.gap_mean_differences import  gap_mean_differences
from analysis.gap_embedding_dim_pairs import gap_embedding_dim_pairs
from analysis.fisher_cumulative_expl_var import fisher_and_cumulative_explained_variance
from analysis.intrinsic_dimensions import intrinsic_dimension_mle
from analysis.modality_gap import compute_gap

def get_loss(loss_type, text_embedding, vision_embedding,bs,contra_temp):
    if loss_type ==  'anchor':                     #anchor or volume or centroids
        loss = compute_loss_anchor(text_embedding, vision_embedding, bs, contra_temp)

    elif loss_type ==  'centroids':  
        print("loss not implemented")
        return 0
        # loss = compute_loss_centroids(text_embedding, audio_embedding, vision_embedding, batch_idx, targets, cf, similarity_matrix,contra_temp)

    elif loss_type == 'volume':
        print("loss not implemented")
        return 0
        # loss = compute_loss_volume(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp)

    elif loss_type == 'area':
        print("loss not implemented")
        return 0
        # loss = compute_loss_area(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp)

    elif loss_type == 'anchor_align_unif':
        print(" loss not implemented")
        return 0
        # loss = compute_loss_anchor_lunif_lalign(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp)
    else:    
        print("loss not implemented")
        return 0
    
    return loss

def eval(cf, test_loader, text_encoder, vision_encoder, device, iteration):
    text_encoder.eval()
    vision_encoder.eval()
    text_embeddings = []
    vision_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            images, captions, fns, cap_idxs = batch
            
            text_emb = text_encoder(captions)
            vision_emb = vision_encoder(images.to(device))
            
              
            text_emb = F.normalize(text_emb,dim=-1)
            vision_emb = F.normalize(vision_emb,dim=-1)
            text_embeddings.append(text_emb.cpu().numpy())
            vision_embeddings.append(vision_emb.cpu().numpy())
            
    # Convert lists to numpy arrays
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    vision_embeddings = np.concatenate(vision_embeddings, axis=0)
    
    # Visualization and analysis
    visualize_3d(cf, text_embeddings, vision_embeddings, iteration)
    gap_mean_differences(cf,text_embeddings, vision_embeddings, iteration)
    gap_embedding_dim_pairs(cf,text_embeddings, vision_embeddings, iteration)
    fisher_and_cumulative_explained_variance(cf,text_embeddings, vision_embeddings, iteration)
    intrinsic_dimension_mle(cf,text_embeddings, vision_embeddings, iteration)
    gaps = dict()
    for i in ['L2M','RMG','L2I']:
        gaps[i] = compute_gap(cf,i,text_embeddings,vision_embeddings,iteration)
        
    
    text_embeddings =   torch.from_numpy(text_embeddings) 
    vision_embeddings = torch.from_numpy(vision_embeddings)
    
    val_loss = get_loss(cf.loss_type,text_embeddings, vision_embeddings, text_embeddings.size(0), cf.contra_temp_init)
    print(f"Iteration {iteration} ==> Validation Loss = {val_loss.item():.4f} | L2M Gap: {gaps['L2M']['text_vision']:.4f} | RMG Gap: {gaps['RMG']['text_vision']:.4f} | L2I Gap: {gaps['L2I']['text_vision']:.4f}")
    if cf.wandb:
        wandb.log({"validation_loss": val_loss.item(), 
                   "L2M_gap": gaps['L2M']['text_vision'],
                   "RMG_gap": gaps['RMG']['text_vision'],
                   "L2I_gap": gaps['L2I']['text_vision']} 
                  )
    
def train_model_with_visualization(cf,text_encoder, vision_encoder, train_loader, test_loader,optimizer, device, num_iterations,contra_temp):
    # This function will contain the training loop and visualization code.
    # You can implement the training loop here and add visualization code to plot the embeddings.
    if not cf.contra_temp_learnable:
        contra_temp.requires_grad = False  # Freeze the temperature if not learnable
    
    
    text_encoder.train()
    vision_encoder.train()
    iteration = 0
    with torch.no_grad():
        eval(cf, test_loader, text_encoder, vision_encoder, device, iteration)
        text_encoder.train()
        vision_encoder.train()
        
    running_loss = 0.0
    tq = tqdm(range(num_iterations), total=num_iterations, desc="Training")
    train_iterator = iter(train_loader)
    for batch_idx in tq:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        images, captions, fns, cap_idxs = batch
        images = images.to(device)
        
        # Forward pass
        text_emb = text_encoder(captions)
        vision_emb = vision_encoder(images)
        
        # Normalize embeddings
        if cf.normalization:
            text_emb = F.normalize(text_emb,dim=-1)
            vision_emb = F.normalize(vision_emb,dim=-1)
            
        # Compute loss
        loss = get_loss(cf.loss_type,text_emb, vision_emb, text_emb.size(0), contra_temp)
        
        if cf.wandb:
            wandb.log({"train_loss": loss.item()}
        )
            
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log on tqdm
        tq.set_postfix({"loss": loss.item()})
        iteration += 1
        running_loss += loss.item()
        
        # Evaluate on validation set every cf.eval_every iterations
        if iteration % cf.eval_every == 0:
            eval(cf, test_loader, text_encoder, vision_encoder, device, iteration)
            text_encoder.train()
            vision_encoder.train()
            
            if cf.wandb and cf.contra_temp_learnable:
                wandb.log({"contra_temp": contra_temp.item()})
                
    # average loss for the epoch
    epoch_loss = running_loss / iteration
    print(f"Epoch completed. Average Loss: {epoch_loss:.4f}")
    if cf.wandb:
        wandb.log({"mean_train_loss": epoch_loss})
    
    return 0