import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBCELoss(nn.Module):
    
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, y_hat, y):
        
        y_hat = y_hat.view(-1)
        y = y.view(-1)

        y_hat = y_hat[y > -0.5]
        y = y[y > -0.5]
        
        return self.loss(y_hat, y) 


class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(CustomBCEWithLogitsLoss, self).__init__()
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))
        
    def forward(self, y_hat, y):
        
        y_hat = y_hat.view(-1)
        y = y.view(-1)

        y_hat = y_hat[y > -0.5]
        y = y[y > -0.5]
        
        return self.loss(y_hat, y)


class CustomAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(CustomAsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        
        mask = y > -0.5
        x = x[mask]
        y = y[mask]

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()



class BoundaryAwareAsymmetricLoss(nn.Module):
    """
    Boundary-aware ASL
    (1) Custom Asymmetric Loss + (2) 1D boundary BCE
    """
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        boundary_weight=1.0
    ):
        super(BoundaryAwareAsymmetricLoss, self).__init__()
        
        # ASL 부분을 그대로 내부에서 사용
        self.asl = CustomAsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            eps=eps,
            disable_torch_grad_focal_loss=disable_torch_grad_focal_loss
        )
        
        self.boundary_weight = boundary_weight

    def forward(self, logits, targets):
        main_loss = self.asl(logits, targets)
        targets = targets.unsqueeze(1)

        pred_prob = torch.sigmoid(logits)

        pred_diff = pred_prob[:, :, 1:] - pred_prob[:, :, :-1] 
        target_diff = targets[:, :, 1:] - targets[:, :, :-1]   

        pred_edge = pred_diff.abs()
        target_edge = target_diff.abs().clamp(max=1)  # 0 or 1 로 바운딩

        boundary_loss = F.l1_loss(pred_edge, target_edge)

        total_loss = main_loss + self.boundary_weight * boundary_loss
        return total_loss