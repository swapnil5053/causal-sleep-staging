import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    Reduces the relative loss for well-classified examples, putting more
    focus on hard, misclassified instances (like minority classes N1 and N3).
    Formula: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0.
            alpha (list or Tensor, optional): Class weights to address imbalance.
            reduction (str): Reduction method: 'mean', 'sum', or 'none'. Default: 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Predicted class logits. Shape: (batch_size, L, num_classes) or (N, num_classes).
            targets (Tensor): Ground truth labels. Shape: (batch_size, L) or (N,).
        
        Returns:
            Tensor: Computed focal loss.
        """
        # Reshape to 2D logit matrix (N, num_classes) and 1D label vector (N,)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Numerically stable log softmax
        log_pt = F.log_softmax(logits, dim=-1)
        # Extract log probability of correct class for each sample
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)
        
        # Compute focal loss term
        loss = -((1 - pt) ** self.gamma) * log_pt
        
        # Apply class balancing weight alpha
        if self.alpha is not None:
            # keep the class weights on whatever device the batch is on
            alpha_t = self.alpha.to(targets.device)[targets]
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss.
    Standard Cross Entropy Loss scaled by class weights (usually inverse class frequencies).
    """
    def __init__(self, weights=None, reduction='mean'):
        """
        Args:
            weights (list or Tensor, optional): Weight scaling factor for each class.
            reduction (str): Loss reduction method. Default: 'mean'.
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Predicted class logits. Shape: (batch_size, L, num_classes) or (N, num_classes).
            targets (Tensor): Ground truth labels. Shape: (batch_size, L) or (N,).
        
        Returns:
            Tensor: Cross entropy loss.
        """
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        weights = self.weights.to(logits.device) if self.weights is not None else None
        return F.cross_entropy(logits, targets, weight=weights, reduction=self.reduction)
