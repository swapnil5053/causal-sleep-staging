import torch.nn as nn


class GRUSequenceEncoder(nn.Module):
    """Encode the one-Hz feature sequence with a recurrent model.

    A unidirectional GRU is strictly causal: the hidden state at second t is a function
    of the input sequence through t only, so no mask is needed. The bidirectional form is
    the matched control and exposes future context by construction.

    Input and output are both (batch, channels, seconds), matching TemporalModel, so the
    two encoders are interchangeable inside SleepStagingModel.

    The control is run at a reduced per-direction hidden size so the two arms match on
    parameter count. At equal hidden size the bidirectional arm would carry roughly twice
    the parameters, which would confound the causality comparison with capacity.
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
        """Args:
            x (Tensor): feature sequence of shape (batch_size, in_channels, L).

        Returns:
            Tensor: encoded sequence of shape (batch_size, out_channels, L).
        """
        encoded, _ = self.gru(x.transpose(1, 2))
        return self.layernorm(encoded).transpose(1, 2)
