from torch import nn

class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, layers=2, heads=4):
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x): return self.head(self.encoder(self.input(x))).squeeze(-1)
