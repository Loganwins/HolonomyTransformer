#!/usr/bin/env python3
"""
CF-HoT with FIXED GATING
========================
The problem: Uniform gates (0.499 everywhere) cancel out after softmax.
The fix: Force gate variance through normalized gating.

Instead of: scores + log(gate)  [constant cancels in softmax]
We use: scores + scale * (gate - mean) / std  [forced discrimination]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import time
import random
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CFHoTConfig:
    model_path: str = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
    output_dir: str = "./results/cfhot_fixed_gating"
    
    # Architecture
    d_fiber: int = 16
    d_control: int = 64
    ema_momentum: float = 0.995
    
    # FIXED: Gate modulation scale (not temperature)
    # This scales the normalized gate signal
    gate_scale: float = 0.5  # Start conservative
    
    # Training
    max_steps: int = 1500
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 256
    
    # Learning rates
    lr_lora: float = 2e-5
    lr_cfhot: float = 5e-4  # Higher for adapters
    weight_decay: float = 0.01
    
    # Loss weights
    lambda_hol: float = 0.01
    lambda_diversity: float = 0.1  # NEW: Reward gate variance
    
    # Monitoring
    log_every: int = 10
    save_every: int = 200
    eval_every: int = 100


# =============================================================================
# CF-HoT ADAPTER with NORMALIZED GATING
# =============================================================================

class CFAdapter(nn.Module):
    """
    Fixed adapter that outputs position-discriminative gates.
    """
    def __init__(self, d_model: int, config: CFHoTConfig):
        super().__init__()
        self.config = config
        
        # Fiber projection
        self.fiber_proj = nn.Linear(d_model, config.d_fiber, bias=False)
        
        # Holonomy predictor - outputs raw risk score (can be any value)
        self.predictor = nn.Sequential(
            nn.Linear(d_model + config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1)
            # NO Softplus - allow negative values for better discrimination
        )
        
        # Learnable scale for gating strength
        self.gate_scale = nn.Parameter(torch.tensor(config.gate_scale))
        
        # Initialize for diversity
        nn.init.normal_(self.fiber_proj.weight, std=0.02)
        nn.init.xavier_normal_(self.predictor[0].weight)
        nn.init.xavier_normal_(self.predictor[2].weight)
        # Bias init to create initial variance
        nn.init.normal_(self.predictor[2].bias, std=0.1)
    
    def forward(
        self, 
        hidden: torch.Tensor, 
        prev_field: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: gate_normalized, field, risk, gate_raw_std
        
        The gate is now NORMALIZED per sequence:
        gate_norm = (risk - mean) / std
        
        This FORCES discrimination - some positions positive, some negative.
        """
        B, S, D = hidden.shape
        orig_dtype = hidden.dtype
        h = hidden.float()
        
        # Fiber projection
        fiber = self.fiber_proj(h)
        
        # Risk prediction (raw, unbounded)
        combined = torch.cat([h, fiber], dim=-1)
        risk = self.predictor(combined).squeeze(-1)  # [B, S]
        
        # EMA accumulation
        alpha = self.config.ema_momentum
        if prev_field is None:
            field = (1 - alpha) * risk
        else:
            if prev_field.shape[1] < S:
                pad = torch.zeros(B, S - prev_field.shape[1], device=h.device, dtype=h.dtype)
                prev_field = torch.cat([prev_field, pad], dim=1)
            elif prev_field.shape[1] > S:
                prev_field = prev_field[:, :S]
            field = alpha * prev_field.float() + (1 - alpha) * risk
        
        # NORMALIZED GATING - forces variance
        field_mean = field.mean(dim=1, keepdim=True)
        if field.shape[1] > 1:
            field_std = field.std(dim=1, keepdim=True) + 1e-6
            gate_normalized = (field - field_mean) / field_std
            raw_std = field.std(dim=1).mean()
        else:
            # Single token: can't normalize, use zero (neutral)
            gate_normalized = torch.zeros_like(field)
            raw_std = torch.tensor(0.0, device=field.device)
        
        return (
            gate_normalized.to(orig_dtype),
            field.to(orig_dtype),
            risk.abs().mean(),  # Risk magnitude for loss
            raw_std.to(orig_dtype)
        )


# =============================================================================
# CF-HoT WRAPPER with NORMALIZED GATING
# =============================================================================

class CFHoTWrapper(nn.Module):
    def __init__(self, model: nn.Module, config: CFHoTConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        self.n_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size
        
        self.adapters = nn.ModuleList([
            CFAdapter(self.d_model, config) for _ in range(self.n_layers)
        ])
        
        # State
        self.fields: List[Optional[torch.Tensor]] = [None] * self.n_layers
        self.gate_values: List[float] = []
        self.gate_stds: List[float] = []
        self.total_risk: float = 0.0
        
        self._patch_attention()
        
        param_count = sum(p.numel() for p in self.adapters.parameters())
        print(f"[CF-HoT Fixed] Adapters: {param_count:,} parameters")
        print(f"[CF-HoT Fixed] Using NORMALIZED gating (forces variance)")
    
    def _get_layers(self):
        if hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'model'):
                if hasattr(self.model.base_model.model, 'model'):
                    return self.model.base_model.model.model.layers
                return self.model.base_model.model.layers
            return self.model.base_model.layers
        return self.model.model.layers
    
    def _patch_attention(self):
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            self._patch_layer_attention(layer.self_attn, idx)
    
    def _patch_layer_attention(self, attn: nn.Module, layer_idx: int):
        original_forward = attn.forward
        wrapper = self
        
        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple] = None,
            **kwargs
        ):
            # Get normalized gate
            gate_norm, field, risk, gate_std = wrapper.adapters[layer_idx](
                hidden_states,
                wrapper.fields[layer_idx]
            )
            
            wrapper.fields[layer_idx] = field
            wrapper.total_risk = wrapper.total_risk + risk
            wrapper.gate_values.append(gate_norm.mean().item())
            wrapper.gate_stds.append(gate_std.item())
            
            # Model config
            config = wrapper.model.config
            num_heads = config.num_attention_heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
            head_dim = config.hidden_size // num_heads
            
            bsz, q_len, _ = hidden_states.shape
            
            # Q, K, V
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)
            
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # Rotary
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            # KV cache
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            
            kv_len = k.shape[2]
            present = (k, v) if use_cache else None
            
            # GQA
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            
            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            
            # Causal mask
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # =================================================================
            # FIXED CF-HoT INTERVENTION
            # Uses normalized gate (mean=0, std=1) scaled by learnable param
            # This FORCES some positions to get positive bias, others negative
            # =================================================================
            
            # Handle KV cache: pad with zeros (neutral after normalization)
            if gate_norm.shape[1] < kv_len:
                pad = torch.zeros(bsz, kv_len - gate_norm.shape[1], 
                                  device=gate_norm.device, dtype=gate_norm.dtype)
                gate_extended = torch.cat([pad, gate_norm], dim=1)
            else:
                gate_extended = gate_norm[:, :kv_len]
            
            # Get learnable scale
            scale = wrapper.adapters[layer_idx].gate_scale
            
            # Apply: scores = scores + scale * normalized_gate
            # Negative gate_norm -> suppress attention
            # Positive gate_norm -> boost attention
            gate_bias = scale * gate_extended  # [B, kv_len]
            gate_bias = gate_bias.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, kv_len]
            
            scores = scores + gate_bias
            # =================================================================
            
            # Softmax and output
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            output = torch.matmul(attn_weights, v)
            
            output = output.transpose(1, 2).reshape(bsz, q_len, -1)
            output = attn.o_proj(output)
            
            return output, present
        
        attn.forward = patched_forward
    
    def reset_state(self):
        self.fields = [None] * self.n_layers
        self.gate_values = []
        self.gate_stds = []
        self.total_risk = 0.0
    
    def get_diversity_loss(self) -> torch.Tensor:
        """Penalize low gate variance (reward discrimination)."""
        if not self.gate_stds:
            return torch.tensor(0.0)
        mean_std = sum(self.gate_stds) / len(self.gate_stds)
        # We want HIGH std, so penalize low std
        return 1.0 / (mean_std + 0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        self.reset_state()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        return outputs, self.total_risk, self.gate_values, self.gate_stds


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# TRAINING
# =============================================================================

def main():
    config = CFHoTConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("CF-HoT FIXED GATING TRAINING")
    print("=" * 70)
    print("Fix: Normalized gating forces position discrimination")
    print("     gate_bias = scale * (field - mean) / std")
    print("=" * 70)
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    device = next(model.parameters()).device
    print(f"Device: {device}")
    
    # LoRA
    print("Adding LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))
    model.print_trainable_parameters()
    
    # CF-HoT
    print("Adding CF-HoT (fixed gating)...")
    cfhot = CFHoTWrapper(model, config)
    cfhot.adapters = cfhot.adapters.to(device).float()
    
    # Data
    print("Loading data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex['text'] for ex in ds if len(ex['text']) > 50]
    random.shuffle(texts)
    print(f"Loaded {len(texts)} samples")
    
    # Optimizer
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': config.lr_lora},
        {'params': cfhot.adapters.parameters(), 'lr': config.lr_cfhot}
    ], weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps, eta_min=1e-6
    )
    
    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    model.train()
    cfhot.adapters.train()
    
    step = 0
    data_idx = 0
    acc_loss, acc_lm, acc_risk, acc_div = 0, 0, 0, 0
    start_time = time.time()
    
    while step < config.max_steps:
        batch_texts = [texts[(data_idx + i) % len(texts)] for i in range(config.batch_size)]
        data_idx += config.batch_size
        
        encoded = tokenizer(
            batch_texts, truncation=True, max_length=config.max_length,
            padding='max_length', return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward
        outputs, risk, gate_vals, gate_stds = cfhot(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )
        
        lm_loss = outputs.loss
        diversity_loss = cfhot.get_diversity_loss()
        
        # Total loss
        loss = lm_loss + config.lambda_hol * risk + config.lambda_diversity * diversity_loss
        
        (loss / config.grad_accum).backward()
        
        acc_loss += loss.item()
        acc_lm += lm_loss.item()
        acc_risk += risk.item() if isinstance(risk, torch.Tensor) else risk
        acc_div += diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
        
        step += 1
        
        if step % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                list(lora_params) + list(cfhot.adapters.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            eta = (config.max_steps - step) / (step / elapsed) / 3600 if step > 0 else 0
            
            mean_gate = sum(gate_vals) / len(gate_vals) if gate_vals else 0
            mean_std = sum(gate_stds) / len(gate_stds) if gate_stds else 0
            
            # Get scale values
            scales = [a.gate_scale.item() for a in cfhot.adapters]
            mean_scale = sum(scales) / len(scales)
            
            print(
                f"Step {step:5d} | "
                f"Loss: {acc_loss/config.log_every:.4f} | "
                f"LM: {acc_lm/config.log_every:.4f} | "
                f"Risk: {acc_risk/config.log_every:.2f} | "
                f"Div: {acc_div/config.log_every:.2f} | "
                f"FieldStd: {mean_std:.4f} | "
                f"Scale: {mean_scale:.3f} | "
                f"ETA: {eta:.1f}h"
            )
            
            acc_loss, acc_lm, acc_risk, acc_div = 0, 0, 0, 0
        
        # Checkpoint
        if step % config.save_every == 0:
            ckpt_dir = os.path.join(config.output_dir, f"checkpoint_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            torch.save({
                'cfhot_state': cfhot.adapters.state_dict(),
                'step': step
            }, os.path.join(ckpt_dir, "cfhot.pt"))
            print(f">>> Saved: {ckpt_dir}")
        
        # Evaluation
        if step % config.eval_every == 0:
            model.eval()
            cfhot.adapters.eval()
            
            print("\n--- Evaluation ---")
            prompts = [
                "The will to power, as described by Nietzsche, is",
                "In the beginning, there was",
            ]
            
            for prompt in prompts:
                cfhot.reset_state()
                with torch.no_grad():
                    inp = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                    out = model.generate(
                        inp, max_new_tokens=80,
                        do_sample=True, temperature=0.8, top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                print(f"  {text[:200]}...")
            
            if cfhot.gate_stds:
                print(f"  Field stds: {cfhot.gate_stds[-5:]}")
            print("--- End Eval ---\n")
            
            model.train()
            cfhot.adapters.train()
    
    # Final save
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    torch.save({
        'cfhot_state': cfhot.adapters.state_dict(),
        'step': step
    }, os.path.join(final_dir, "cfhot.pt"))
    
    print("\n" + "=" * 70)
    print(f"DONE! Saved to {final_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
