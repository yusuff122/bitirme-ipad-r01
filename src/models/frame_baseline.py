import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.embedding_dim = model.fc.in_features
        self.weights_name = weights.name if weights is not None else None
        self.eval()
        for p in self.parameters(): p.requires_grad = False
    @torch.inference_mode()
    def forward(self, x): return self.features(x).flatten(1)

def knn_scores(query, memory, k=1, chunk_size=512):
    if memory.ndim != 2 or query.ndim != 2 or query.shape[1] != memory.shape[1]:
        raise ValueError('query and memory must be 2D tensors with matching feature dimensions')
    if memory.shape[0] < 1:
        raise ValueError('memory bank must contain at least one embedding')
    if k < 1:
        raise ValueError('k must be >= 1')
    outputs=[]
    for batch in query.split(chunk_size):
        distances = torch.cdist(batch.float(), memory.float())
        outputs.append(distances.topk(min(k, memory.shape[0]), largest=False).values.mean(dim=1))
    return torch.cat(outputs)
