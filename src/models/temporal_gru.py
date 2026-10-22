"""GRU next-clip embedding predictor for E2."""
import torch
from torch import nn


class GRUTemporalPredictor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, batch_first=True,
                          dropout=float(dropout) if self.num_layers > 1 else 0.0)
        self.predictor = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, context):
        if context.ndim != 3 or context.shape[-1] != self.input_dim:
            raise ValueError(f'Expected (batch, context, {self.input_dim}), got {tuple(context.shape)}')
        output, _ = self.gru(context)
        return self.predictor(output[:, -1, :])


def next_embedding_mse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(f'Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}')
    return torch.mean((prediction.float() - target.float()) ** 2)
