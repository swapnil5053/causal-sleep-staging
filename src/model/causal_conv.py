import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer that prevents information flow from future steps.
    Applies asymmetric padding to the left of the input signal before calling PyTorch's Conv1d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, causal=True, **kwargs):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            dilation (int): Dilation rate. Default: 1.
            **kwargs: Other arguments passed to nn.Conv1d (e.g. groups, bias).
        """
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        # Total padding needed to keep the sequence length unchanged.
        # Causal: all of it on the left, so no future sample can reach the output.
        # Non-causal: split either side, which lets the kernel see the future. Used only
        # for the causality ablation, where everything else is held fixed.
        self.causal_padding = (kernel_size - 1) * dilation
        
        # Standard convolution with no padding (we pad manually in forward)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, seq_len).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, seq_len).
        """
        # F.pad format for 1D: (pad_left, pad_right) on the last dimension
        if self.causal:
            x_padded = F.pad(x, (self.causal_padding, 0))
        else:
            left = self.causal_padding // 2
            x_padded = F.pad(x, (left, self.causal_padding - left))
        return self.conv(x_padded)
