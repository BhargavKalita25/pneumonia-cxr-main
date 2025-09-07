import torch
from torch import nn
import numpy as np, cv2

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        layer = dict([*self.model.named_modules()])[target_layer]
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)
    def _fwd(self, m, inp, out): self.activations = out.detach()
    def _bwd(self, m, gin, gout): self.gradients = gout[0].detach()
    def __call__(self, x):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:,0].sum(); score.backward()
        grads, acts = self.gradients, self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights*acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam); cam = (cam-cam.min())/(cam.max()+1e-8)
        return cam.cpu().numpy()[0,0], torch.sigmoid(logits).item()

def overlay_cam(rgb_img, cam, alpha=0.45):
    h,w = rgb_img.shape[:2]
    heat = (cam*255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:,:,::-1]
    over = (alpha*heat+(1-alpha)*rgb_img).clip(0,255).astype(np.uint8)
    return over
