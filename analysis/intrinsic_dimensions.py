# We use the maximum likelihood estimate of the intrinsic dimension
# we are referring to the paper: Multi-modal contrastive learning adapts to intrinsic dimensions of shared latent variables
# we refer like them to https://scikit-dimension.readthedocs.io/en/latest/skdim.id.MLE.html

import torch
import numpy as np
from skdim.id import MLE
import wandb

def intrinsic_dimension_mle(cf, text_embeddings, vision_embeddings, iterations,
                            n_neighbors=20, smooth=False, random_subsample=None):
    """
    Levina-Bickel MLE intrinsic dimension estimator (as used via skdim.id in the paper).

    Args:
        text_embeddings/vision_embeddings:
            array-like or torch.Tensor of shape [N, D]
        n_neighbors:
            k used for kNN distances inside the MLE estimator (often 10-30 is a reasonable start)
        smooth:
            if True, skdim returns a smoothed pointwise estimate (averaged over each point's neighborhood)
        random_subsample:
            optional int. If set, subsample that many points before estimating (useful if N is huge).

    Returns:
        A dict with global and pointwise ID estimates for each modality and for the concatenation.
    """

    # 1) Put everything into torch float tensors (on CPU) with shape [N, D]
    modalities_embeddings = {
        "text":  torch.as_tensor(text_embeddings,  dtype=torch.float32, device="cpu"),
        "vision":torch.as_tensor(vision_embeddings,dtype=torch.float32, device="cpu"),
    }

    # 2) Optionally subsample points (ID estimation can be heavy for very large N)
    if random_subsample is not None:
        N = modalities_embeddings["text"].shape[0]
        idx = torch.randperm(N)[:random_subsample]
        for m in modalities_embeddings:
            modalities_embeddings[m] = modalities_embeddings[m][idx]

    # 3) Build datasets we want IDs for:
    #    - each modality separately
    #    - "all" = stack modalities as extra samples (3N x D), similar spirit to your Fisher code
    
    # MLE breaks if there are duplicate points. Given that here the text is just 'one','two'... in a batch more then one sample of a given class must appears. 
    X_text  = torch.unique(modalities_embeddings["text"], dim=0).numpy()   # (N, D)
    X_vision= modalities_embeddings["vision"].numpy() # (N, D)
    X_all   = np.concatenate([X_text, X_vision], axis=0)  # (2N, D)

    # 4) Create estimator
    #    skdim's MLE uses kNN distances internally; n_neighbors controls that k.
    est = MLE( neighborhood_based=True, K=n_neighbors)
    out = {}

    for name, X in [("text", X_text), ("vision", X_vision), ("all", X_all)]:
        # Global ID estimate (single scalar)
        id_global = est.fit_transform(X)  # returns est.dimension_ (global intrinsic dimension) :contentReference[oaicite:1]{index=1}

        # Pointwise ID estimate (one per sample)
        # If you want local/pointwise values for plots/distributions:
        id_pointwise = est.fit_transform_pw(X, n_neighbors=n_neighbors, smooth=smooth)  # returns est.dimension_ (global intrinsic dimension) :contentReference[oaicite:2]{index=2}

        out[name] = {
            "id_global": float(id_global),
            "id_pointwise": np.asarray(id_pointwise),
        }
        
    
    if cf.wandb and iterations%100 == 0:
        wandb.log({f"intrinsic_dimension_text": float(out['text']['id_global'])})
        wandb.log({f"intrinsic_dimension_vision": float(out['vision']['id_global'])})
        wandb.log({f"intrinsic_dimension_all": float(out['all']['id_global'])})