import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomASLLossBinary(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps=0.1,
                 reduction='mean', pos_weight=None):
        super().__init__()
        self.gamma_pos  = gamma_pos
        self.gamma_neg  = gamma_neg
        self.eps        = eps
        self.reduction  = reduction
        self.register_buffer('pos_weight', pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, y_hat, y, mask=None):
        """
        y_hat : (B, T)   *or* (B, T, 2)  – raw logits
        y     : (B, T)                – 0 / 1,  (<0 => ignore)
        """
        # ── 1. convert 2-logit input to single positive-logit ──────────
        if y_hat.dim() == y.dim() + 1:
            # assume last dim==2 : [logit_neg, logit_pos]
            y_hat = y_hat[..., 1]          # keep positive-class logit

        # ── 2. flatten and remove padding (y < -0.5) ──────────────────
        y_hat = y_hat.reshape(-1)
        y     = y.reshape(-1)

        valid = y > -0.5
        y_hat = y_hat[valid]
        y     = y[valid]

        # ── 3. sigmoid → probabilities ────────────────────────────────
        p     = torch.sigmoid(y_hat)

        # ── 4. label-smoothing (if eps>0) ─────────────────────────────
        if self.eps > 0:
            y_smooth = y    * (1 - self.eps) + 0.5 * self.eps
        else:
            y_smooth = y

        # ── 5. asymmetric focal weights ───────────────────────────────
        pt_pos = 1 - p               # for positive targets
        pt_neg = p                   # for negative targets

        w = torch.ones_like(y_hat)
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        w[pos_mask] = pt_pos[pos_mask].pow(self.gamma_pos)
        w[neg_mask] = pt_neg[neg_mask].pow(self.gamma_neg)

        # ── 6. BCE component (with smoothing) ─────────────────────────
        eps_ = 1e-8
        loss = -(
            y_smooth * torch.log(p + eps_) +
            (1 - y_smooth) * torch.log(1 - p + eps_)
        )

        # ── 7. apply focal weight and optional positive scaling ───────
        loss = loss * w
        loss[pos_mask] *= self.pos_weight

        # ── 8. reduction ──────────────────────────────────────────────
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:           # 'none'
            return loss


def masked_focal_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                      alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Compute the masked focal loss for binary classification.
    - logits: model outputs (unnormalized) of shape (batch, output_length)
    - targets: ground truth labels (0 or 1) of shape (batch, output_length)
    - mask: binary mask of shape (batch, output_length), with 1 for valid positions and 0 for border positions to ignore
    - alpha: balancing factor for positive class (default 0.25)
    - gamma: focusing parameter (default 2.0)
    """
    B, T, C = logits.shape
    logits = logits.reshape(-1, C)           # (B*T, 2)
    targets = targets.reshape(-1).long()     # (B*T,)
    mask = mask.reshape(-1)

    # convert to one-hot
    targets_onehot = F.one_hot(targets, num_classes=C).float()  # (B*T, 2)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # focal factor
    pt = (probs * targets_onehot).sum(dim=-1)        # p_t for each sample
    alpha_t = alpha * targets_onehot[:,1] + (1-alpha) * targets_onehot[:,0]
    focal_factor = alpha_t * (1 - pt) ** gamma

    ce = -(targets_onehot * log_probs).sum(dim=-1)    # cross-entropy per sample
    loss = focal_factor * ce

    # mask out border positions
    loss = loss * mask
    return loss.sum() / mask.sum()
