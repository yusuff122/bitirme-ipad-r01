"""Partially fine-tuned Kinetics Video Swin Tiny with a contrastive projection head."""
import torch
from torch import nn


MODEL_NAME='swin3d_t'
WEIGHTS_NAME='Swin3D_T_Weights.KINETICS400_V1'


class AdaptedVideoSwin(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if not pretrained: raise ValueError('E4 requires official pretrained Kinetics400 weights')
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t
        self.weights=Swin3D_T_Weights.KINETICS400_V1
        self.backbone=swin3d_t(weights=self.weights)
        self.backbone.head=nn.Identity()
        self.embedding_dim=768
        self.projection_head=nn.Sequential(nn.Linear(768,256),nn.ReLU(inplace=True),nn.Linear(256,128))
        for p in self.backbone.parameters(): p.requires_grad_(False)
        # torchvision Swin3D Tiny: features[6] is the final Swin stage.
        for p in self.backbone.features[6].parameters(): p.requires_grad_(True)
        for p in self.backbone.norm.parameters(): p.requires_grad_(True)
        for p in self.projection_head.parameters(): p.requires_grad_(True)

    def make_transform(self): return self.weights.transforms()

    def forward(self, clips):
        if clips.ndim!=5 or clips.shape[1]!=3: raise ValueError(f'Expected B,C,T,H,W, got {tuple(clips.shape)}')
        features=self.backbone(clips).float()
        if features.ndim!=2 or features.shape[1]!=768: raise ValueError(f'Unexpected Swin feature shape: {tuple(features.shape)}')
        projection=torch.nn.functional.normalize(self.projection_head(features),dim=1)
        return features,projection

    def trainable_parameter_names(self): return [n for n,p in self.named_parameters() if p.requires_grad]


def info_nce_loss(projections, temperature=.07):
    """Symmetric batch InfoNCE: diagonal pairs positive, different-sequence rows negative."""
    if projections.ndim!=3 or projections.shape[1]!=2: raise ValueError('Expected normalized projections (B,2,128)')
    a=torch.nn.functional.normalize(projections[:,0],dim=-1); b=torch.nn.functional.normalize(projections[:,1],dim=-1)
    logits=(a@b.T)/float(temperature)
    labels=torch.arange(len(a),device=a.device)
    return .5*(torch.nn.functional.cross_entropy(logits,labels)+torch.nn.functional.cross_entropy(logits.T,labels)),logits
