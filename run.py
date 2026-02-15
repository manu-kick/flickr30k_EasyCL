# main.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib



import wandb

from models.text_encoder import TextEncoder
from models.img_encoder import VisionEncoder
from dataloader import get_dataloaders
from pipelines import train_model_with_visualization

from config_loader import load_configs_from_dir
matplotlib.use("Agg")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_experiment(cf):
    # set working directory to the script's directory (for python debugger vscode)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    seed_everything(cf.seed)

    run_wandb = None
    if cf.wandb:
        run_wandb = wandb.init(
            settings=wandb.Settings(init_timeout=120),
            project=f"{cf.dataset_name}_ContrastiveLearning",
            name=cf.run,
            config=cf.log_config(),
            reinit=True,  # IMPORTANT: allow multiple runs in same process
        )

    device = cf.device

    # Encoders
    text_encoder = TextEncoder(word2vec_model_path=cf.w2v_path, embedding_size=cf.embedding_dim).to(device)
    vision_encoder = VisionEncoder(input_channels=3, output_dim=cf.embedding_dim).to(device)

    contra_temp = nn.Parameter(torch.tensor(cf.contra_temp_init, device=device))
    params = list(text_encoder.parameters()) + list(vision_encoder.parameters())
    if cf.contra_temp_learnable:
        params += [contra_temp]

    optimizer = torch.optim.Adam(params, lr=cf.lr)

    train_loader, val_loader, test_loader = get_dataloaders(cf)

    train_model_with_visualization(
        cf,
        text_encoder,
        vision_encoder,
        train_loader,
        val_loader,
        optimizer,
        device=device,
        num_iterations=cf.num_iterations,
        contra_temp=contra_temp
    )

    if run_wandb is not None:
        run_wandb.finish()


def main():
    configs = load_configs_from_dir("./config_dir")

    print(f"Found {len(configs)} configs in {'./config_dir'}:")
    for i, (path, cf) in enumerate(configs, 1):
        print(f"\n=== Running experiment {i}/{len(configs)} ===")
        print(f"Config: {path}")
        run_experiment(cf)


main()
