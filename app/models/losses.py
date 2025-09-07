import torch, torch.nn.functional as F

def weighted_bce_with_logits(logits, targets, pos_weight=None):
    return F.binary_cross_entropy_with_logits(
        logits, targets.unsqueeze(1), pos_weight=pos_weight)

def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits).clamp(1e-6,1-1e-6)
    t = targets.unsqueeze(1)
    ce = F.binary_cross_entropy(p,t,reduction="none")
    pt = t*p+(1-t)*(1-p)
    return (alpha*(1-pt)**gamma*ce).mean()
