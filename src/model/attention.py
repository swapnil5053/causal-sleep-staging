import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """
    Causal (masked) Multi-Head Self-Attention layer.
    Ensures that for any time step i, the model only attends to steps j <= i (past and present).
    Used to model complex context dependencies across seconds in sleep recordings.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim (int): Input feature size. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            dropout (float): Attention dropout.
        """
        super(CausalSelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Sequence features of shape (batch_size, L, embed_dim).
        
        Returns:
            Tensor: Attended features of shape (batch_size, L, embed_dim).
        """
        L = x.size(1)
        
        # Create a boolean causal mask of shape (L, L)
        # where elements above the diagonal are True (masked out) and others are False (allowed).
        # We construct it dynamically on the device of x.
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        # Run Multi-Head Attention
        # attn_output shape: (batch_size, L, embed_dim)
        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            need_weights=False
        )
        
        # Pre-LN residual connection or Post-LN residual connection
        # Post-LN: layernorm(x + attn_out) is standard in standard PyTorch Transformer blocks.
        return self.layernorm(x + attn_out)
