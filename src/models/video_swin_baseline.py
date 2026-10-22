"""Frozen torchvision Video Swin Tiny feature extractor."""
import torch
from torch import nn


MODEL_NAME = 'swin3d_t'
WEIGHTS_NAME = 'Swin3D_T_Weights.KINETICS400_V1'


class VideoSwinFeatureExtractor(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True, weights_name=WEIGHTS_NAME):
        super().__init__()
        if model_name != MODEL_NAME:
            raise ValueError(f'Unsupported Video Swin model {model_name!r}; expected {MODEL_NAME!r}')
        if not pretrained:
            raise ValueError('Randomly initialized Video Swin is forbidden for E1 feature extraction')
        if weights_name != WEIGHTS_NAME:
            raise ValueError(f'Unsupported weights {weights_name!r}; expected {WEIGHTS_NAME!r}')
        try:
            from torchvision.models.video import Swin3D_T_Weights, swin3d_t
        except ImportError as exc:
            raise RuntimeError('Installed torchvision does not provide swin3d_t weights API') from exc
        self.weights = Swin3D_T_Weights.KINETICS400_V1
        self.weights_name = f'{type(self.weights).__name__}.{self.weights.name}'
        self.model_name = model_name
        self.model = swin3d_t(weights=self.weights)
        self.embedding_dim = int(self.model.head.in_features)
        self.model.head = nn.Identity()  # output after final norm/pooling, before classification
        self.model.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def make_transform(self):
        """Official deterministic resize/crop/RGB normalization for these weights."""
        return self.weights.transforms()

    @torch.inference_mode()
    def forward(self, clips):
        if clips.ndim != 5:
            raise ValueError(f'Expected B,C,T,H,W clips, got shape {tuple(clips.shape)}')
        features = self.model(clips).float()
        if features.ndim != 2 or features.shape[1] != self.embedding_dim:
            raise RuntimeError(f'Unexpected feature output shape {tuple(features.shape)}')
        return features
