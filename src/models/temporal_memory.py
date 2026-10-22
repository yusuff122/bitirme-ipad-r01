"""GRU temporal predictor with a learnable normal-pattern prototype memory."""
import torch
from torch import nn
import torch.nn.functional as F


class TemporalMemoryPredictor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_memory_slots=64, memory_dim=256):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_memory_slots = int(num_memory_slots)
        self.memory_dim = int(memory_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.memory_slots = nn.Parameter(F.normalize(torch.randn(self.num_memory_slots, self.memory_dim), dim=1))
        self.query = nn.Linear(self.hidden_dim, self.memory_dim, bias=False)
        self.reconstruction = nn.Sequential(nn.Linear(self.hidden_dim + self.memory_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.input_dim))

    def forward(self, context):
        if context.ndim != 3 or context.shape[-1] != self.input_dim:
            raise ValueError(f'Expected (batch, context, {self.input_dim}), got {tuple(context.shape)}')
        output, _ = self.gru(context)
        h = output[:, -1, :]
        q = F.normalize(self.query(h), dim=-1)
        slots = F.normalize(self.memory_slots, dim=-1)
        similarities = q @ slots.T
        attention = torch.softmax(similarities, dim=-1)
        memory = attention @ slots
        prediction = self.reconstruction(torch.cat([h, memory], dim=-1))
        return {'prediction': prediction, 'hidden': h, 'attention': attention, 'memory': memory, 'memory_slots': slots}


def memory_losses(output, target, memory_slots):
    """Prediction MSE; compactness=1-cos(q, retrieved m); diversity=mean squared off-diagonal cosine."""
    prediction = F.mse_loss(output['prediction'].float(), target.float())
    q = F.normalize(output['hidden'], dim=-1)
    m = F.normalize(output['memory'], dim=-1)
    compactness = (1.0 - (q * m).sum(dim=-1)).mean()
    slots = F.normalize(memory_slots, dim=-1)
    gram = slots @ slots.T
    off_diag = gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    diversity = off_diag.square().sum() / (gram.shape[0] * (gram.shape[0] - 1))
    return prediction, compactness, diversity
