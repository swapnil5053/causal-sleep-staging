import torch
import torch.nn as nn
from src.model.causal_conv import CausalConv1d

class MultiResolutionCNN(nn.Module):
    """
    Multi-Resolution CNN module.
    Runs two parallel 1D causal convolutions with different kernel sizes:
    - High-frequency branch (kernel ~ 50) to capture sleep spindles and K-complexes.
    - Low-frequency branch (kernel ~ 400) to capture slow waves.
    
    The outputs of both branches are concatenated along the channel dimension
    and pooled down to a 1-second resolution.
    """
    def __init__(self, in_channels=1, channels_1=16, channels_2=16, downsample_factor=100, dropout=0.2):
        """
        Args:
            in_channels (int): Input signal channels. Default: 1.
            channels_1 (int): Channels for high-freq branch.
            channels_2 (int): Channels for low-freq branch.
            downsample_factor (int): Factor by which to downsample (100 Hz to 1 Hz).
            dropout (float): Dropout probability.
        """
        super(MultiResolutionCNN, self).__init__()
        
        # High-frequency branch (small receptive field)
        self.branch_high = nn.Sequential(
            CausalConv1d(in_channels, channels_1, kernel_size=50),
            nn.BatchNorm1d(channels_1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Low-frequency branch (large receptive field)
        self.branch_low = nn.Sequential(
            CausalConv1d(in_channels, channels_2, kernel_size=400),
            nn.BatchNorm1d(channels_2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Causal Downsampling (MaxPool with kernel=stride=downsample_factor)
        # Pooling is left-aligned, ensuring only past/present points in the window are pooled.
        self.pool = nn.MaxPool1d(kernel_size=downsample_factor, stride=downsample_factor)
        
        # Total output channels
        self.out_channels = channels_1 + channels_2

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor representing continuous raw signal.
                       Shape: (batch_size, 1, L * 100) where L is the sequence length in seconds.
        
        Returns:
            Tensor: Feature representation at 1-second intervals.
                    Shape: (batch_size, out_channels, L).
        """
        h_high = self.branch_high(x)
        h_low = self.branch_low(x)
        
        # Concatenate branches along the channel dimension
        h = torch.cat([h_high, h_low], dim=1) # (batch_size, ch_1 + ch_2, L * 100)
        
        # Downsample to 1-second steps
        out = self.pool(h) # (batch_size, out_channels, L)
        return out
