import torch
import torch.nn as nn
from src.model.causal_conv import CausalConv1d

class TemporalResidualBlock(nn.Module):
    """
    A single residual block for Temporal Convolutional Networks (TCN)
    featuring causal dilated convolutions, batch normalization, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        """
        Args:
            in_channels (int): Input channel dimension.
            out_channels (int): Output channel dimension.
            kernel_size (int): Convolution kernel size. Default: 3.
            dilation (int): Dilation rate. Default: 1.
            dropout (float): Dropout probability.
        """
        super(TemporalResidualBlock, self).__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Linear projection projection shortcut if channels do not match
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature sequence of shape (batch_size, in_channels, L).
        
        Returns:
            Tensor: Output feature sequence of shape (batch_size, out_channels, L).
        """
        residual = x
        
        # First dilated causal conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second dilated causal conv layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Project residual if channel count differs
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        return self.relu_out(out + residual)


class TemporalModel(nn.Module):
    """
    TCN stack that processes downsampled second-by-second sleep features.
    Configurable depth, channel sizes, and dilation schedule.
    """
    def __init__(self, in_channels, channel_list=[32, 32, 32], kernel_size=3, dilations=[1, 2, 4], dropout=0.2):
        """
        Args:
            in_channels (int): Input feature size (concatenated MRCNN channels).
            channel_list (list of int): Output channels of each residual block.
            kernel_size (int): Conv kernel size. Default: 3.
            dilations (list of int): Dilation rate for each residual block.
            dropout (float): Dropout rate.
        """
        super(TemporalModel, self).__init__()
        
        assert len(channel_list) == len(dilations), "Length of channel_list and dilations must match!"
        
        layers = []
        curr_channels = in_channels
        for i in range(len(channel_list)):
            layers.append(
                TemporalResidualBlock(
                    in_channels=curr_channels,
                    out_channels=channel_list[i],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout
                )
            )
            curr_channels = channel_list[i]
            
        self.network = nn.Sequential(*layers)
        self.out_channels = curr_channels

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features of shape (batch_size, in_channels, L).
        
        Returns:
            Tensor: Modeled sequence of shape (batch_size, out_channels, L).
        """
        return self.network(x)
