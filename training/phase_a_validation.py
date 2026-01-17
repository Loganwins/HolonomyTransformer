#!/usr/bin/env python3
"""
CONTROL FIELD HOT — PHASE A: ARCHITECTURE VALIDATION
=====================================================
Small standalone model (~50M params) to prove CF-HoT works before scaling.

This validates:
1. Control field learns meaningful risk signals
2. Gating actually affects attention patterns  
3. Perplexity matches or beats baseline
4. Training is stable

Hardware: RTX 3090 (24GB) — this will use ~6-8GB, leaving room for Phase B
Training time: ~4-6 hours for full validation

Author: Logan Napolitano + Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import math
import json
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CFHoTConfig:
    """Small config optimized for fast validation on 3090."""
    # Architecture (small but meaningful)
    vocab_size: int = 50257      # GPT-2 tokenizer
    d_model: int = 512           # Hidden dimension
    n_heads: int = 8             # Attention heads
    n_layers: int = 8            # Transformer layers
    d_ff: int = 2048             # FFN intermediate
    max_seq_len: int = 512       # Context length
    dropout: float = 0.1
    
    # Control Field (the innovation)
    d_fiber: int = 32            # Consistency subspace
    d_control: int = 64          # Predictor hidden dim
    momentum: float = 0.9        # EMA decay
    lambda_gate: float = 1.0     # Attention gate scale (learnable)
    lambda_curv: float = 0.1     # Curvature gate scale
    
    # Loss weights (kept weak per GPT-5.2's advice)
    lambda_hol_loss: float = 1e-4   # Holonomy regularization
    lambda_curv_loss: float = 1e-5  # Curvature regularization
    
    # Toggle for ablation
    use_control_field: bool = True
    
    def num_params(self) -> int:
        """Estimate parameter count."""
        embed = self.vocab_size * self.d_model
        attn_per_layer = 4 * self.d_model * self.d_model  # Q, K, V, O
        ffn_per_layer = 2 * self.d_model * self.d_ff
        cf_per_layer = (self.d_model * self.d_fiber + 
                        (self.d_model + self.d_fiber) * self.d_control + 
                        self.d_control * 1)  # fiber proj + predictor
        
        base = embed + self.n_layers * (attn_per_layer + ffn_per_layer)
        cf = self.n_layers * cf_per_layer if self.use_control_field else 0
        return base + cf


# ============================================================================
# VECTORIZED CONTROL FIELD
# ============================================================================

class ControlFieldModule(nn.Module):
    """
    The core innovation: predict consistency risk and accumulate into control field.
    
    Key equations:
        φ_t = W_fiber · x_t                           (fiber projection)
        Δh_t = softplus(MLP([x_t; φ_t]))              (holonomy prediction)
        h_t = α · h_{t-1} + (1-α) · Δh_t             (control field accumulation)
        g_t = σ(-λ · h_t)                            (attention gate)
    
    Vectorized via causal convolution for GPU efficiency.
    """
    
    def __init__(self, config: CFHoTConfig):
        super().__init__()
        self.d_fiber = config.d_fiber
        self.momentum = config.momentum
        
        # Fiber projection: compress hidden state to consistency subspace
        self.fiber_proj = nn.Linear(config.d_model, config.d_fiber)
        
        # Holonomy predictor: estimate local consistency risk
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model + config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1),
            nn.Softplus()  # Risk is non-negative
        )
        
        # Initialize predictor output small (start with minimal gating)
        nn.init.zeros_(self.predictor[-2].bias)
        nn.init.normal_(self.predictor[-2].weight, std=0.01)
        
        # Learnable gate scale
        self.lambda_gate = nn.Parameter(torch.tensor(config.lambda_gate))
        
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [batch, seq, d_model]
        
        Returns:
            gate: [batch, seq] attention gate values (0-1)
            field: [batch, seq] accumulated control field
            delta_h: [batch, seq] predicted holonomy increments
            fiber: [batch, seq, d_fiber] fiber states for curvature
        """
        batch, seq_len, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype
        
        # Project to fiber space
        fiber = self.fiber_proj(hidden)  # [batch, seq, d_fiber]
        
        # Predict holonomy increments
        combined = torch.cat([hidden, fiber], dim=-1)
        delta_h = self.predictor(combined).squeeze(-1)  # [batch, seq]
        
        # Clamp delta_h to prevent explosion (risk should be small)
        delta_h = torch.clamp(delta_h, 0, 1)
        
        # Vectorized EMA accumulation via causal convolution
        # Build kernel: [(1-α)·α^{n-1}, (1-α)·α^{n-2}, ..., (1-α)]
        powers = self.momentum ** torch.arange(seq_len, device=device, dtype=dtype)
        kernel = (1 - self.momentum) * powers.flip(0)
        kernel = kernel.view(1, 1, -1)
        
        # Causal convolution
        delta_h_padded = F.pad(delta_h.unsqueeze(1), (seq_len - 1, 0))
        field = F.conv1d(delta_h_padded, kernel).squeeze(1)  # [batch, seq]
        
        # Compute gate: high field → low gate → suppress attention
        # Clamp field to prevent explosion
        field = torch.clamp(field, 0, 10)
        
        gate = torch.sigmoid(-self.lambda_gate * field)  # [batch, seq]
        
        return gate, field, delta_h, fiber


# ============================================================================
# ATTENTION WITH CONTROL FIELD GATING
# ============================================================================

class ControlFieldAttention(nn.Module):
    """Multi-head attention with control field gating."""
    
    def __init__(self, config: CFHoTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = self.d_head ** -0.5
        
        # Standard attention projections
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Control field (if enabled)
        if config.use_control_field:
            self.control_field = ControlFieldModule(config)
        else:
            self.control_field = None
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, seq, d_model]
            mask: [seq, seq] causal mask
        
        Returns:
            output: [batch, seq, d_model]
            metrics: dict with control field statistics
        """
        batch, seq_len, _ = x.shape
        
        # QKV projections
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Control field gating
        metrics = {}
        if self.control_field is not None:
            gate, field, delta_h, fiber = self.control_field(x)
            
            # Apply gate in log-space (numerically stable)
            log_gate = torch.log(gate.clamp(min=1e-8))
            scores = scores + log_gate.unsqueeze(1).unsqueeze(2)  # Broadcast to all heads
            
            metrics = {
                'gate': gate,
                'field': field,
                'delta_h': delta_h,
                'fiber': fiber
            }
        
        # Softmax and apply attention
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Combine values
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.W_o(output)
        
        return output, metrics


# ============================================================================
# CURVATURE-GATED FFN
# ============================================================================

class CurvatureGatedFFN(nn.Module):
    """FFN with curvature gating — suppresses output in high-curvature fiber regions."""
    
    def __init__(self, config: CFHoTConfig):
        super().__init__()
        self.config = config
        
        # Standard FFN
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Curvature gate scale (learnable)
        if config.use_control_field:
            self.lambda_curv = nn.Parameter(torch.tensor(config.lambda_curv))
    
    def forward(
        self, 
        x: torch.Tensor,
        fiber: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq, d_model]
            fiber: [batch, seq, d_fiber] from control field
        
        Returns:
            output: [batch, seq, d_model]
            curvature: [batch, seq] if computed
        """
        # Standard FFN forward
        hidden = self.fc1(x)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        
        curvature = None
        
        # Curvature gating
        if fiber is not None and self.config.use_control_field:
            batch, seq_len, d_fiber = fiber.shape
            
            # Compute curvature as finite difference: ||φ_{t+1} - φ_{t-1}|| / 2
            curvature = torch.zeros(batch, seq_len, device=fiber.device, dtype=fiber.dtype)
            if seq_len > 2:
                curvature[:, 1:-1] = torch.norm(fiber[:, 2:] - fiber[:, :-2], dim=-1) / 2
                curvature[:, 0] = torch.norm(fiber[:, 1] - fiber[:, 0], dim=-1)
                curvature[:, -1] = torch.norm(fiber[:, -1] - fiber[:, -2], dim=-1)
            
            # Gate: high curvature → suppress output
            gate = torch.sigmoid(1 - self.lambda_curv * curvature)
            hidden = hidden * gate.unsqueeze(-1)
        
        output = self.fc2(hidden)
        output = self.dropout(output)
        
        return output, curvature


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class CFHoTBlock(nn.Module):
    """Single transformer block with control field integration."""
    
    def __init__(self, config: CFHoTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = ControlFieldAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = CurvatureGatedFFN(config)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Pre-norm attention with control field
        attn_out, metrics = self.attn(self.ln1(x), mask)
        x = x + attn_out
        
        # Pre-norm FFN with curvature gating
        fiber = metrics.get('fiber', None)
        ffn_out, curvature = self.ffn(self.ln2(x), fiber)
        x = x + ffn_out
        
        if curvature is not None:
            metrics['curvature'] = curvature
        
        return x, metrics


# ============================================================================
# FULL MODEL
# ============================================================================

class CFHoTModel(nn.Module):
    """
    Control Field Holonomy Transformer — Small Validation Model
    
    ~50M parameters, designed to prove the architecture works
    before scaling to 8B.
    """
    
    def __init__(self, config: CFHoTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            CFHoTBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embed.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Causal mask (cached)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        )
        
        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[CFHoT] Parameters: {n_params:,}")
        print(f"[CFHoT] Control Field: {'ENABLED' if config.use_control_field else 'DISABLED'}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq]
            labels: [batch, seq] (optional, for computing loss)
        
        Returns:
            dict with logits, loss, and control field metrics
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        # Causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Forward through blocks, accumulating metrics
        total_holonomy = 0.0
        total_curvature = 0.0
        mean_gate = 0.0
        
        for block in self.blocks:
            x, metrics = block(x, mask)
            
            if 'delta_h' in metrics:
                total_holonomy = total_holonomy + metrics['delta_h'].sum()
            if 'curvature' in metrics:
                total_curvature = total_curvature + metrics['curvature'].sum()
            if 'gate' in metrics:
                mean_gate = mean_gate + metrics['gate'].mean()
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        output = {
            'logits': logits,
            'total_holonomy': total_holonomy,
            'total_curvature': total_curvature,
            'mean_gate': mean_gate / self.config.n_layers if self.config.use_control_field else 1.0
        }
        
        # Loss
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Total loss with weak regularization
            total_loss = lm_loss
            if self.config.use_control_field:
                norm = batch * seq_len * self.config.n_layers
                hol_reg = self.config.lambda_hol_loss * total_holonomy / norm
                curv_reg = self.config.lambda_curv_loss * total_curvature / norm
                total_loss = total_loss + hol_reg + curv_reg
            
            output['lm_loss'] = lm_loss
            output['loss'] = total_loss
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> torch.Tensor:
        """Simple generation for testing."""
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            output = self(idx_cond)
            logits = output['logits'][:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ============================================================================
# DATA LOADING
# ============================================================================

def load_wikitext(tokenizer, seq_len: int = 512, split: str = 'train'):
    """Load WikiText-103 for training."""
    from datasets import load_dataset
    
    print(f"[Data] Loading WikiText-103 ({split})...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    
    # Tokenize and concatenate
    all_tokens = []
    for example in dataset:
        if example['text'].strip():
            tokens = tokenizer.encode(example['text'])
            all_tokens.extend(tokens)
    
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"[Data] Total tokens: {len(tokens):,}")
    
    # Create sequences
    n_seqs = len(tokens) // seq_len
    tokens = tokens[:n_seqs * seq_len].view(n_seqs, seq_len)
    
    return tokens


class TokenDataset(Dataset):
    """Simple dataset wrapper."""
    
    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        x = self.tokens[idx]
        return x, x.clone()


# ============================================================================
# TRAINING
# ============================================================================

def train_phase_a(
    output_dir: str = './phase_a_results',
    batch_size: int = 16,
    gradient_accumulation: int = 2,
    max_steps: int = 20000,
    eval_interval: int = 1000,
    save_interval: int = 5000,
    lr: float = 3e-4,
    compare_baseline: bool = True
):
    """
    Phase A: Train small CF-HoT to validate architecture.
    
    This proves:
    1. CF-HoT trains stably
    2. Control field learns non-trivial values
    3. Perplexity is competitive with baseline
    """
    print("=" * 70)
    print("PHASE A: CONTROL FIELD HOT ARCHITECTURE VALIDATION")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Config
    config = CFHoTConfig()
    print(f"\n[Config] d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"[Config] d_fiber={config.d_fiber}, d_control={config.d_control}")
    print(f"[Config] Estimated params: {config.num_params():,}")
    
    # Create CF-HoT model
    print("\n[Model] Creating CF-HoT...")
    cf_hot = CFHoTModel(config).to(device)
    
    # Create baseline for comparison
    baseline = None
    if compare_baseline:
        print("\n[Model] Creating Baseline...")
        baseline_config = CFHoTConfig()
        baseline_config.use_control_field = False
        baseline = CFHoTModel(baseline_config).to(device)
    
    # Data
    print("\n[Data] Loading WikiText-103...")
    train_tokens = load_wikitext(tokenizer, config.max_seq_len, 'train')
    val_tokens = load_wikitext(tokenizer, config.max_seq_len, 'validation')
    
    train_dataset = TokenDataset(train_tokens)
    val_dataset = TokenDataset(val_tokens)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    # Optimizers
    cf_optimizer = torch.optim.AdamW(cf_hot.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    cf_scaler = GradScaler()
    
    if baseline:
        bl_optimizer = torch.optim.AdamW(baseline.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
        bl_scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING CF-HoT")
    print("=" * 70)
    
    cf_hot.train()
    train_iter = iter(train_loader)
    
    history = {
        'cf_hot': {'train_loss': [], 'val_loss': [], 'val_ppl': [], 'holonomy': [], 'gate': []},
        'baseline': {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    }
    
    for step in range(1, max_steps + 1):
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        # CF-HoT forward/backward
        with autocast():
            output = cf_hot(x, labels=y)
            loss = output['loss'] / gradient_accumulation
        
        cf_scaler.scale(loss).backward()
        
        if step % gradient_accumulation == 0:
            cf_scaler.unscale_(cf_optimizer)
            torch.nn.utils.clip_grad_norm_(cf_hot.parameters(), 1.0)
            cf_scaler.step(cf_optimizer)
            cf_scaler.update()
            cf_optimizer.zero_grad()
        
        # Logging
        if step % 100 == 0:
            lm_loss = output['lm_loss'].item()
            holonomy = output['total_holonomy'].item() if isinstance(output['total_holonomy'], torch.Tensor) else output['total_holonomy']
            gate = output['mean_gate'].item() if isinstance(output['mean_gate'], torch.Tensor) else output['mean_gate']
            
            print(f"Step {step:5d} | Loss: {lm_loss:.4f} | PPL: {math.exp(lm_loss):.2f} | "
                  f"Holonomy: {holonomy:.2f} | Gate: {gate:.3f}")
            
            history['cf_hot']['train_loss'].append((step, lm_loss))
            history['cf_hot']['holonomy'].append((step, holonomy))
            history['cf_hot']['gate'].append((step, gate))
        
        # Evaluation
        if step % eval_interval == 0:
            print("\n[Eval] Evaluating CF-HoT...")
            cf_hot.eval()
            
            val_loss = 0.0
            n_batches = 0
            
            with torch.no_grad():
                for vx, vy in val_loader:
                    if n_batches >= 50:  # Quick eval
                        break
                    vx, vy = vx.to(device), vy.to(device)
                    with autocast():
                        out = cf_hot(vx, labels=vy)
                    val_loss += out['lm_loss'].item()
                    n_batches += 1
            
            val_loss /= n_batches
            val_ppl = math.exp(val_loss)
            
            print(f"[Eval] CF-HoT: Val Loss = {val_loss:.4f}, PPL = {val_ppl:.2f}")
            history['cf_hot']['val_loss'].append((step, val_loss))
            history['cf_hot']['val_ppl'].append((step, val_ppl))
            
            cf_hot.train()
            print()
        
        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = output_path / f'cf_hot_step_{step}.pt'
            torch.save({
                'step': step,
                'model_state_dict': cf_hot.state_dict(),
                'optimizer_state_dict': cf_optimizer.state_dict(),
                'config': config,
                'history': history
            }, ckpt_path)
            print(f"[Save] Checkpoint: {ckpt_path}")
    
    # Train baseline if requested
    if baseline:
        print("\n" + "=" * 70)
        print("TRAINING BASELINE")
        print("=" * 70)
        
        baseline.train()
        train_iter = iter(train_loader)
        
        for step in range(1, max_steps + 1):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            
            with autocast():
                output = baseline(x, labels=y)
                loss = output['loss'] / gradient_accumulation
            
            bl_scaler.scale(loss).backward()
            
            if step % gradient_accumulation == 0:
                bl_scaler.unscale_(bl_optimizer)
                torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
                bl_scaler.step(bl_optimizer)
                bl_scaler.update()
                bl_optimizer.zero_grad()
            
            if step % 100 == 0:
                lm_loss = output['lm_loss'].item()
                print(f"Step {step:5d} | Loss: {lm_loss:.4f} | PPL: {math.exp(lm_loss):.2f}")
                history['baseline']['train_loss'].append((step, lm_loss))
            
            if step % eval_interval == 0:
                baseline.eval()
                val_loss = 0.0
                n_batches = 0
                
                with torch.no_grad():
                    for vx, vy in val_loader:
                        if n_batches >= 50:
                            break
                        vx, vy = vx.to(device), vy.to(device)
                        with autocast():
                            out = baseline(vx, labels=vy)
                        val_loss += out['lm_loss'].item()
                        n_batches += 1
                
                val_loss /= n_batches
                val_ppl = math.exp(val_loss)
                
                print(f"[Eval] Baseline: Val Loss = {val_loss:.4f}, PPL = {val_ppl:.2f}\n")
                history['baseline']['val_loss'].append((step, val_loss))
                history['baseline']['val_ppl'].append((step, val_ppl))
                
                baseline.train()
    
    # Final comparison
    print("\n" + "=" * 70)
    print("PHASE A RESULTS")
    print("=" * 70)
    
    if history['cf_hot']['val_ppl'] and history['baseline']['val_ppl']:
        cf_final_ppl = history['cf_hot']['val_ppl'][-1][1]
        bl_final_ppl = history['baseline']['val_ppl'][-1][1]
        
        print(f"CF-HoT Final PPL:   {cf_final_ppl:.2f}")
        print(f"Baseline Final PPL: {bl_final_ppl:.2f}")
        print(f"Improvement: {(bl_final_ppl - cf_final_ppl) / bl_final_ppl * 100:.2f}%")
        
        # Check control field behavior
        if history['cf_hot']['holonomy']:
            final_holonomy = history['cf_hot']['holonomy'][-1][1]
            final_gate = history['cf_hot']['gate'][-1][1]
            print(f"\nControl Field Stats:")
            print(f"  Final Holonomy: {final_holonomy:.2f}")
            print(f"  Final Gate Mean: {final_gate:.3f}")
            
            if final_gate < 0.95 and final_gate > 0.05:
                print("  ✓ Gate is non-trivial (not collapsed)")
            else:
                print("  ⚠ Gate may be collapsed")
    
    # Save final results
    results_path = output_path / 'phase_a_results.json'
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[Save] Results: {results_path}")
    
    return cf_hot, baseline, history


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase A: CF-HoT Architecture Validation')
    parser.add_argument('--output_dir', type=str, default='./phase_a_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--no_baseline', action='store_true')
    
    args = parser.parse_args()
    
    train_phase_a(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        compare_baseline=not args.no_baseline
    )
