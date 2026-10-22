import torch
from torch import nn

class VideoSwinFeatureExtractor(nn.Module):
    """Lazy timm wrapper; raises a clear error if the installed timm lacks a video model."""
    def __init__(self, model_name='swin3d_tiny', pretrained=True):
        super().__init__()
        try:
            import timm
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        except Exception as exc:
            raise RuntimeError(f'Could not load Video Swin model {model_name!r}; check timm model availability/weights: {exc}') from exc
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad=False
    @torch.inference_mode()
    def forward(self, clips): return self.model(clips)

def clips_to_tensor(clips, transform):
    # clips: list of RGB numpy frames; output expected by video models: B,C,T,H,W
    import numpy as np
    tensors=[]
    for clip in clips:
        tensors.append(torch.stack([transform(frame) for frame in clip], dim=1))
    return torch.stack(tensors)
