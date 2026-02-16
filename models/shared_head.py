import torch
import torch.nn as nn


class SharedHead(nn.Module):
    def __init__(self, d_in: int, d_shared: int = None, dropout: float = 0.0):
        super().__init__()
        d_shared = d_in if d_shared is None else d_shared

        # se d_shared == d_in, è una “refine head” nello stesso spazio
        self.ln1 = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_shared)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_shared)

        # init “gentile”
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.ln1(z)
        x = self.drop(self.act(self.fc1(x)))
        x = self.ln2(x)
        return x
