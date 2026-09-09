import torch
import torch.nn as nn
from src.model.mrcnn import MultiResolutionCNN
from src.model.temporal import TemporalModel
from src.model.attention import CausalSelfAttention
from src.model.classifier import Classifier
from src.model.gru import GRUSequenceEncoder

class SleepStagingModel(nn.Module):
    """
    Integrated Sleep Staging Model wiring:
    Continuous Raw EEG (batch, L, 100) 
      -> Flatten to continuous (batch, 1, L * 100)
      -> Multi-Resolution CNN (MRCNN) (batch, channels_1 + channels_2, L)
      -> Causal Dilated Conv Stack (TCN) (batch, tcn_channels[-1], L)
      -> Transpose (batch, L, tcn_channels[-1])
      -> Causal Multi-Head Self-Attention (MHA) (batch, L, tcn_channels[-1])
      -> Classifier (FC + Softmax) (batch, L, num_classes)
    """
    def __init__(self, config=None, **kwargs):
        """
        Args:
            config (dict, optional): Config dict containing 'model' hyperparameters.
            **kwargs: Direct hyperparameter overrides.
        """
        super(SleepStagingModel, self).__init__()
        
        # Load hyperparameters from config dictionary or direct overrides
        model_cfg = config.get("model", {}) if config is not None else {}
        
        mrcnn_ch1 = kwargs.get("mrcnn_channels_1", model_cfg.get("mrcnn_channels_1", 16))
        mrcnn_ch2 = kwargs.get("mrcnn_channels_2", model_cfg.get("mrcnn_channels_2", 16))
        
        architecture = kwargs.get("architecture", model_cfg.get("architecture", "tcn_attention"))
        tcn_channels = kwargs.get("tcn_channels", model_cfg.get("tcn_channels", [32, 32, 32]))
        tcn_kernel = kwargs.get("tcn_kernel_size", model_cfg.get("tcn_kernel_size", 3))
        tcn_dilations = kwargs.get("tcn_dilations", model_cfg.get("tcn_dilations", [1, 2, 4]))
        tcn_dropout = kwargs.get("tcn_dropout", model_cfg.get("tcn_dropout", 0.2))
        
        attn_heads = kwargs.get("attn_num_heads", model_cfg.get("attn_num_heads", 4))
        attn_dropout = kwargs.get("attn_dropout", model_cfg.get("attn_dropout", 0.1))
        
        num_classes = kwargs.get("num_classes", model_cfg.get("num_classes", 5))

        # Causality switch. True is the real model. False keeps every layer, channel and
        # parameter identical but lets convolutions pad symmetrically and drops the attention
        # mask, so the only difference is access to future signal. Used to measure the cost
        # of the causal constraint on this exact architecture.
        causal = kwargs.get("causal", model_cfg.get("causal", True))
        self.causal = causal
        self.architecture = architecture
        
        # 1. Multi-Resolution CNN
        self.mrcnn = MultiResolutionCNN(
            in_channels=1,
            channels_1=mrcnn_ch1,
            channels_2=mrcnn_ch2,
            downsample_factor=100, # downsample 100 Hz to 1 Hz
            dropout=tcn_dropout,
            causal=causal
        )
        
        if architecture == "gru":
            self.temporal = GRUSequenceEncoder(
                input_size=self.mrcnn.out_channels,
                hidden_size=kwargs.get("gru_hidden_size", model_cfg.get("gru_hidden_size", 32)),
                num_layers=kwargs.get("gru_num_layers", model_cfg.get("gru_num_layers", 2)),
                dropout=kwargs.get("gru_dropout", model_cfg.get("gru_dropout", 0.1)),
                causal=causal,
            )
            self.attention = None
        elif architecture == "tcn_attention":
            # 2. Temporal Convolutional Network
            self.temporal = TemporalModel(
                in_channels=self.mrcnn.out_channels,
                channel_list=tcn_channels,
                kernel_size=tcn_kernel,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
                causal=causal
            )

            # 3. Causal Multi-Head Self-Attention
            self.attention = CausalSelfAttention(
                embed_dim=self.temporal.out_channels,
                num_heads=attn_heads,
                dropout=attn_dropout,
                causal=causal
            )
        else:
            raise ValueError(
                f"Unknown model architecture {architecture!r}; "
                "expected 'tcn_attention' or 'gru'"
            )
        
        # 4. Final Staging Classifier
        self.classifier = Classifier(
            embed_dim=self.temporal.out_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input raw EEG sequence of shape (batch_size, L, 100).
                        L represents the context sequence length in seconds.
                        100 represents the raw samples per second (100 Hz).
        
        Returns:
            Tensor: Logits for each second of the sequence.
                    Shape: (batch_size, L, num_classes).
        """
        batch_size, L, samples_per_sec = x.size()
        
        # Flatten raw EEG sequence into a single continuous channel representation
        # shape: (batch_size, 1, L * 100)
        x_continuous = x.view(batch_size, 1, L * samples_per_sec)
        
        # Extract features and downsample to 1 Hz
        # shape: (batch_size, mrcnn_out_channels, L)
        features = self.mrcnn(x_continuous)
        
        # Model temporal sequence dynamics using dilated causal convs
        # shape: (batch_size, tcn_out_channels, L)
        temporal_features = self.temporal(features)

        if self.attention is None:
            attended_features = temporal_features
        else:
            # Transpose for multi-head self-attention: (batch_size, L, tcn_out_channels)
            temporal_features = temporal_features.transpose(1, 2)
            attended_features = self.attention(temporal_features)
        
        # Classify each second of the sequence
        # shape: (batch_size, L, num_classes)
        logits = self.classifier(attended_features)
        
        return logits

if __name__ == "__main__":
    # Quick sanity check on parameter budget and dimensions
    model = SleepStagingModel(
        mrcnn_channels_1=16,
        mrcnn_channels_2=16,
        tcn_channels=[32, 32, 32],
        tcn_kernel_size=3,
        tcn_dilations=[1, 2, 4],
        attn_num_heads=4,
        num_classes=5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"--- SleepStagingModel Sanity Check ---")
    print(f"Total Parameters: {total_params} (~{total_params/1000:.1f}K)")
    print(f"Target Parameter Budget: ~48K")
    print(f"Status: {'PASSED' if total_params <= 48000 else 'OVER BUDGET'}")
    
    # Test batch forward pass
    dummy_input = torch.randn(8, 30, 100) # batch_size=8, L=30 seconds, 100 Hz
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape (logits): {output.shape} (Expected: [8, 30, 5])")
