import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    Classifier module.
    Takes attended per-second feature representations and projects them to
    logits across the sleep staging target classes.
    """
    def __init__(self, embed_dim, num_classes=5):
        """
        Args:
            embed_dim (int): Dimensionality of the incoming features.
            num_classes (int): Number of target classes. Default: 5 (W, N1, N2, N3, REM).
        """
        super(Classifier, self).__init__()
        # Simple Linear projection to logits (softmax is applied in loss function / evaluation)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): Attended sequence features of shape (batch_size, L, embed_dim).
        
        Returns:
            Tensor: Class logits of shape (batch_size, L, num_classes).
        """
        # x is (batch_size, L, embed_dim) -> output is (batch_size, L, num_classes)
        return self.fc(x)
