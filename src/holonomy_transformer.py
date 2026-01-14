"""
HOLONOMY TRANSFORMER
====================
A geometrically-native neural architecture for consistent reasoning.

Author: Logan Napolitano
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HoTConfig:
    """Configuration for Holonomy Transformer."""
    vocab_size: int = 50257
    d_base: int = 512          # Base manifold dimension
    d_fiber: int = 32          # Fiber dimension
    d_ff: int = 2048           # FFN hidden dimension
    n_heads: int = 8           # Attention heads
    n_layers: int = 6          # Transformer blocks
    lie_rank: int = 8          # Lie algebra generators
    max_seq_len: int = 2048    # Maximum sequence length
    dropout: float = 0.1
    holonomy_weight: float = 1.0
    curvature_weight: float = 1.0
    waypoint_threshold: float = 0.1


@dataclass
class FiberSection:
    """A section of the principal fiber bundle."""
    base: torch.Tensor       # [batch, seq, d_base]
    fiber: torch.Tensor      # [batch, seq, d_fiber, d_fiber]
    connection: torch.Tensor # [batch, seq, lie_rank]
    generators: torch.Tensor # [lie_rank, d_fiber, d_fiber]


class FiberBundleEmbedding(nn.Module):
    """Embed tokens as sections of a fiber bundle."""
    
    def __init__(self, config: HoTConfig):
        super().__init__()
        self.config = config
        
        self.base_embed = nn.Embedding(config.vocab_size, config.d_base)
        self.fiber_embed = nn.Embedding(config.vocab_size, config.d_fiber * config.d_fiber)
        self.connection_embed = nn.Embedding(config.vocab_size, config.lie_rank)
        
        # Antisymmetric Lie algebra generators
        generators = torch.zeros(config.lie_rank, config.d_fiber, config.d_fiber)
        for i in range(config.lie_rank):
            A = torch.randn(config.d_fiber, config.d_fiber) * 0.01
            generators[i] = A - A.T
        self.register_buffer('generators', generators)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.base_embed.weight, std=0.02)
        with torch.no_grad():
            identity = torch.eye(self.config.d_fiber).flatten()
            self.fiber_embed.weight.copy_(
                identity.unsqueeze(0).expand(self.config.vocab_size, -1) +
                torch.randn_like(self.fiber_embed.weight) * 0.01
            )
        nn.init.normal_(self.connection_embed.weight, std=0.01)
    
    def forward(self, tokens: torch.Tensor) -> FiberSection:
        batch, seq_len = tokens.shape
        
        base = self.base_embed(tokens)
        fiber_flat = self.fiber_embed(tokens)
        fiber = fiber_flat.view(batch, seq_len, self.config.d_fiber, self.config.d_fiber)
        
        # Project to SO(n)
        U, _, Vh = torch.linalg.svd(fiber)
        fiber = U @ Vh
        
        connection = self.connection_embed(tokens)
        
        return FiberSection(base, fiber, connection, self.generators)


class ParallelTransportAttention(nn.Module):
    """Attention based on parallel transport holonomy cost."""
    
    def __init__(self, config: HoTConfig):
        super().__init__()
        self.config = config
        self.d_head = config.d_base // config.n_heads
        
        self.W_q = nn.Linear(config.d_base, config.d_base)
        self.W_k = nn.Linear(config.d_base, config.d_base)
        self.W_v = nn.Linear(config.d_base, config.d_base)
        self.W_o = nn.Linear(config.d_base, config.d_base)
        
        self.dropout = nn.Dropout(config.dropout)
        self.lambda_hol = nn.Parameter(torch.tensor(config.holonomy_weight))
    
    def forward(self, section: FiberSection, mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = section.base.shape
        
        Q = self.W_q(section.base).view(batch, seq_len, self.config.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(section.base).view(batch, seq_len, self.config.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(section.base).view(batch, seq_len, self.config.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Holonomy cost matrix
        holonomy_cost = self._compute_holonomy(section)
        scores = scores - self.lambda_hol * holonomy_cost.unsqueeze(1)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.config.d_base)
        output = self.W_o(output)
        
        # Update fiber
        avg_attn = attn.mean(dim=1)
        new_fiber = torch.einsum('bij,bjkl->bikl', avg_attn, section.fiber)
        U, _, Vh = torch.linalg.svd(new_fiber)
        new_fiber = U @ Vh
        
        total_hol = (attn * holonomy_cost.unsqueeze(1)).sum()
        
        return FiberSection(output, new_fiber, section.connection, section.generators), total_hol
    
    def _compute_holonomy(self, section: FiberSection) -> torch.Tensor:
        batch, seq_len, _ = section.connection.shape
        
        A = torch.einsum('bsr,rij->bsij', section.connection, section.generators)
        T = torch.matrix_exp(A * 0.1)
        T_inv = T.transpose(-1, -2)
        
        holonomy = torch.zeros(batch, seq_len, seq_len, device=section.base.device)
        identity = torch.eye(self.config.d_fiber, device=section.base.device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    round_trip = T[:, j] @ T_inv[:, i] @ T[:, i] @ T_inv[:, j]
                    holonomy[:, i, j] = torch.norm(round_trip - identity, dim=(-2, -1))
        
        return holonomy


class CurvatureGatedFFN(nn.Module):
    """Feedforward with curvature-based gating."""
    
    def __init__(self, config: HoTConfig):
        super().__init__()
        self.config = config
        
        self.W1 = nn.Linear(config.d_base, config.d_ff)
        self.W2 = nn.Linear(config.d_ff, config.d_base)
        self.dropout = nn.Dropout(config.dropout)
        self.lambda_curv = nn.Parameter(torch.tensor(config.curvature_weight))
    
    def forward(self, section: FiberSection):
        curvature = self._compute_curvature(section)
        gate = torch.sigmoid(1 - self.lambda_curv * curvature)
        
        hidden = F.gelu(self.W1(section.base))
        hidden = gate.unsqueeze(-1) * hidden
        hidden = self.dropout(hidden)
        output = self.W2(hidden)
        
        return FiberSection(output, section.fiber, section.connection, section.generators), curvature.sum()
    
    def _compute_curvature(self, section: FiberSection) -> torch.Tensor:
        A = torch.einsum('bsr,rij->bsij', section.connection, section.generators)
        
        dA = torch.zeros_like(A)
        dA[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / 2
        dA[:, 0] = A[:, 1] - A[:, 0]
        dA[:, -1] = A[:, -1] - A[:, -2]
        
        F_curv = dA + torch.matmul(A, A)
        return torch.norm(F_curv, dim=(-2, -1))


class HoTBlock(nn.Module):
    """Single Holonomy Transformer block."""
    
    def __init__(self, config: HoTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_base)
        self.attn = ParallelTransportAttention(config)
        self.ln2 = nn.LayerNorm(config.d_base)
        self.ffn = CurvatureGatedFFN(config)
    
    def forward(self, section: FiberSection, mask: Optional[torch.Tensor] = None):
        losses = {}
        
        # Attention
        normed = FiberSection(self.ln1(section.base), section.fiber, section.connection, section.generators)
        attn_out, hol_loss = self.attn(normed, mask)
        losses['holonomy'] = hol_loss
        
        section = FiberSection(section.base + attn_out.base, attn_out.fiber, section.connection, section.generators)
        
        # FFN
        normed = FiberSection(self.ln2(section.base), section.fiber, section.connection, section.generators)
        ffn_out, curv_loss = self.ffn(normed)
        losses['curvature'] = curv_loss
        
        section = FiberSection(section.base + ffn_out.base, ffn_out.fiber, section.connection, section.generators)
        
        return section, losses


class HolonomyTransformer(nn.Module):
    """The Holonomy Transformer - geometrically-native consistency."""
    
    def __init__(self, config: HoTConfig):
        super().__init__()
        self.config = config
        
        self.embed = FiberBundleEmbedding(config)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_base)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([HoTBlock(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.d_base)
        self.head = nn.Linear(config.d_base, config.vocab_size, bias=False)
        self.head.weight = self.embed.base_embed.weight  # Tie weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Causal mask
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
        if attention_mask is not None:
            causal = causal * attention_mask.unsqueeze(1)
        
        # Embed
        section = self.embed(input_ids)
        pos = self.pos_embed(torch.arange(seq_len, device=device))
        section = FiberSection(
            self.dropout(section.base + pos),
            section.fiber, section.connection, section.generators
        )
        
        # Blocks
        total_hol, total_curv = 0.0, 0.0
        for block in self.blocks:
            section, losses = block(section, causal)
            total_hol += losses['holonomy']
            total_curv += losses['curvature']
        
        # Output
        hidden = self.ln_f(section.base)
        logits = self.head(hidden)
        
        output = {
            'logits': logits,
            'holonomy_loss': total_hol,
            'curvature_loss': total_curv,
        }
        
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output['lm_loss'] = lm_loss
            output['loss'] = lm_loss + 0.1 * total_hol + 0.1 * total_curv
        
        return output
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0):
        self.eval()
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.forward(generated)
                logits = outputs['logits'][:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated


if __name__ == '__main__':
    # Quick test
    config = HoTConfig(vocab_size=1000, d_base=256, d_fiber=16, n_layers=2)
    model = HolonomyTransformer(config)
    
    x = torch.randint(0, 1000, (2, 32))
    out = model(x, labels=x)
    
    print(f"Logits shape: {out['logits'].shape}")
    print(f"LM Loss: {out['lm_loss']:.4f}")
    print(f"Holonomy Loss: {out['holonomy_loss']:.4f}")
    print(f"Curvature Loss: {out['curvature_loss']:.4f}")
    print(f"Total Loss: {out['loss']:.4f}")
    print("\nHolonomy Transformer initialized successfully!")
