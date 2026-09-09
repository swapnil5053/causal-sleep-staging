import torch.nn as nn


class GRUSequenceEncoder(nn.Module):
    """Encode one-Hz EEG features with a causal recurrent sequence model.

    A unidirectional GRU is strictly causal. The bidirectional form is retained for
    the matched non-causal control arm and exposes future context by construction.
    """

    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.1, causal=True):
        super().__init__()
        self.causal = causal
        self.hidden_size = hidden_size
        self.bidirectional = not causal
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.out_channels = hidden_size * (2 if self.bidirectional else 1)
        self.layernorm = nn.LayerNorm(self.out_channels)

    def forward(self, x):
        """Return encoded features for ``(batch, channels, seconds)`` input."""
        sequence = x.transpose(1, 2)
        encoded, _ = self.gru(sequence)
        return self.layernorm(encoded)