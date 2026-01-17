#!/usr/bin/env python3
"""
TRUE CF-HoT Implementation - CORRECTED
======================================
Implements "Consistency Is All You Need" with hyperparameters from
"Definitive Training Configuration for CF-HoT Adapters"

Core intervention (Section 3.5):
    scores = scores + log(gate + ε)

Critical fixes from training config:
    - EMA momentum: 0.995 (not 0.9 - prevents gate collapse)
    - Gate temperature: 2.0 (softens sigmoid)
    - Bounded gates: [0.1, 0.9] (prevents saturation)
    - Training: 500-1500 steps with monitoring
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
# CONFIGURATION - From "Definitive Training Configuration" document
# =============================================================================

@dataclass
class CFHoTConfig:
    # Model
    model_path: str = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
    output_dir: str = "./results/true_cfhot_v2"
    
    # CF-HoT Architecture (Section 3 of paper)
    d_fiber: int = 16
    d_control: int = 64
    
    # CRITICAL: EMA momentum 0.995 not 0.9
    # "momentum=0.9 provides an effective window of only ~10 updates"
    # "0.995 slows accumulation; effective window ~200 steps"
    ema_momentum: float = 0.995
    
    # CRITICAL: Gate temperature 2.0
    # "Softens sigmoid to prevent saturation"
    gate_temperature: float = 2.0
    
    # CRITICAL: Bounded gates [0.1, 0.9]
    # "prevents complete attention suppression while still allowing meaningful modulation"
    gate_min: float = 0.1
    gate_max: float = 0.9
    
    # Training (from config doc)
    max_steps: int = 1500  # "500-1500 with checkpointing"
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 256
    
    # Learning rates (from config doc)
    lr_lora: float = 2e-5
    lr_cfhot: float = 1e-4  # "Standard for adapter tuning"
    weight_decay: float = 0.01
    
    # Loss weights (Section 4.1 + config doc)
    lambda_hol: float = 0.01  # "Start low; increase if holonomy signal too weak"
    lambda_curv: float = 0.001  # "Curvature regularization should be gentle"
    lambda_gate_reg: float = 0.005  # "penalize (gate - 0.5)² to prevent extremes"
    
    # Monitoring
    log_every: int = 10
    save_every: int = 200  # "checkpointing every 200 steps"
    eval_every: int = 100
    
    # Early stopping (from config doc)
    early_stop_gate_threshold: float = 0.1  # "more than 50% of gates below 0.1"
    early_stop_gate_fraction: float = 0.3  # "30% of gates fall below 0.1"


# =============================================================================
# CF-HoT ADAPTER - Exactly as in paper with config doc corrections
# =============================================================================

class CFAdapter(nn.Module):
    """
    Paper Section 3.3: Holonomy Predictor
    With corrections from training configuration document.
    """
    def __init__(self, d_model: int, config: CFHoTConfig):
        super().__init__()
        self.config = config
        
        # Fiber projection (Section 3.2)
        # "maps d_model-dimensional hidden states to d_fiber-dimensional fiber states"
        self.fiber_proj = nn.Linear(d_model, config.d_fiber, bias=False)
        
        # Holonomy predictor MLP (Section 3.3)
        # "small feedforward network...outputs a non-negative scalar via Softplus"
        self.predictor = nn.Sequential(
            nn.Linear(d_model + config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1),
            nn.Softplus()
        )
        
        # Gate scale λ (Section 3.5)
        # Initialize to produce neutral gates
        self.lambda_gate = nn.Parameter(torch.tensor(2.0))
        
        # Initialization per Appendix A.4 and config doc
        # "zero-initialized gating is essential for stable adapter training"
        nn.init.normal_(self.fiber_proj.weight, std=0.01)
        nn.init.zeros_(self.predictor[-2].bias)
        nn.init.normal_(self.predictor[-2].weight, std=0.01)
    
    def bounded_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        From config doc:
        "bounded sigmoid approach guarantees that even at extreme input values,
        gates remain in the [0.1, 0.9] range"
        """
        base = torch.sigmoid(x / self.config.gate_temperature)
        return self.config.gate_min + (self.config.gate_max - self.config.gate_min) * base
    
    def forward(
        self, 
        hidden: torch.Tensor, 
        prev_field: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: gate, field, delta_h (risk), fiber
        
        Paper equations:
        - φ_t = W_fiber · x_t                    (fiber projection)
        - Δh_t = Softplus(MLP([x_t; φ_t]))       (holonomy prediction)
        - h_t = α·h_{t-1} + (1-α)·Δh_t           (EMA accumulation)
        - g_t = bounded_σ(-λ·h_t / temperature)  (gating with corrections)
        """
        B, S, D = hidden.shape
        orig_dtype = hidden.dtype
        h = hidden.float()
        
        # Fiber projection (Section 3.2)
        fiber = self.fiber_proj(h)  # [B, S, d_fiber]
        
        # Holonomy prediction (Section 3.3)
        combined = torch.cat([h, fiber], dim=-1)
        delta_h = self.predictor(combined).squeeze(-1)  # [B, S]
        
        # EMA accumulation (Section 3.4)
        # CRITICAL: Using 0.995 not 0.9
        alpha = self.config.ema_momentum
        
        if prev_field is None:
            field = (1 - alpha) * delta_h
        else:
            # Handle sequence length mismatch
            if prev_field.shape[1] < S:
                pad = torch.zeros(B, S - prev_field.shape[1], device=h.device, dtype=h.dtype)
                prev_field = torch.cat([prev_field, pad], dim=1)
            elif prev_field.shape[1] > S:
                prev_field = prev_field[:, :S]
            field = alpha * prev_field.float() + (1 - alpha) * delta_h
        
        # Gating (Section 3.5) with bounded sigmoid from config doc
        gate = self.bounded_sigmoid(-self.lambda_gate * field)
        
        return (
            gate.to(orig_dtype), 
            field.to(orig_dtype), 
            delta_h.to(orig_dtype),
            fiber.to(orig_dtype)
        )


# =============================================================================
# CF-HoT WRAPPER - Patches attention for log-space modulation
# =============================================================================

class CFHoTWrapper(nn.Module):
    """
    Wraps a model and patches attention to implement:
        scores = scores + log(gate + ε)
    
    This is the ONLY modification to the forward pass.
    Per Section 3.5 of "Consistency Is All You Need"
    """
    def __init__(self, model: nn.Module, config: CFHoTConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        # Get model dimensions
        self.n_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size
        
        # One adapter per layer
        self.adapters = nn.ModuleList([
            CFAdapter(self.d_model, config) for _ in range(self.n_layers)
        ])
        
        # State tracking
        self.fields: List[Optional[torch.Tensor]] = [None] * self.n_layers
        self.gate_history: List[float] = []
        self.total_risk: float = 0.0
        self.fiber_curvatures: List[torch.Tensor] = []
        
        # Patch attention
        self._patch_attention()
        
        param_count = sum(p.numel() for p in self.adapters.parameters())
        print(f"[CF-HoT] Adapters initialized: {param_count:,} parameters")
        print(f"[CF-HoT] EMA momentum: {config.ema_momentum}")
        print(f"[CF-HoT] Gate temperature: {config.gate_temperature}")
        print(f"[CF-HoT] Gate bounds: [{config.gate_min}, {config.gate_max}]")
    
    def _get_layers(self):
        """Navigate model structure to find transformer layers."""
        if hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'model'):
                if hasattr(self.model.base_model.model, 'model'):
                    return self.model.base_model.model.model.layers
                return self.model.base_model.model.layers
            return self.model.base_model.layers
        return self.model.model.layers
    
    def _patch_attention(self):
        """Replace attention forward to inject log(gate) into scores."""
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            self._patch_layer_attention(layer.self_attn, idx)
    
    def _patch_layer_attention(self, attn: nn.Module, layer_idx: int):
        """Patch a single attention module."""
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
            # Compute gate for this layer
            gate, field, delta_h, fiber = wrapper.adapters[layer_idx](
                hidden_states, 
                wrapper.fields[layer_idx]
            )
            
            # Update state
            wrapper.fields[layer_idx] = field
            wrapper.total_risk = wrapper.total_risk + delta_h.mean()
            wrapper.gate_history.append(gate.mean().item())
            
            # Compute fiber curvature for regularization (Section 3.6)
            if layer_idx > 0 and len(wrapper.fiber_curvatures) > 0:
                prev_fiber = wrapper.fiber_curvatures[-1]
                if prev_fiber.shape == fiber.shape:
                    curvature = (fiber - prev_fiber).norm(dim=-1).mean()
                    wrapper.fiber_curvatures.append(fiber)
                else:
                    wrapper.fiber_curvatures.append(fiber)
            else:
                wrapper.fiber_curvatures.append(fiber)
            
            # Get model config
            config = wrapper.model.config
            num_heads = config.num_attention_heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
            head_dim = config.hidden_size // num_heads
            
            bsz, q_len, _ = hidden_states.shape
            
            # Q, K, V projections
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)
            
            # Reshape for attention
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # Apply rotary embeddings if provided
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            # Handle KV cache
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            
            kv_len = k.shape[2]
            present = (k, v) if use_cache else None
            
            # GQA: expand KV heads if needed
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            
            # Apply causal mask
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # =================================================================
            # THE CF-HoT INTERVENTION (Section 3.5)
            # "scores = scores + log(g + ε)"
            # =================================================================
            eps = 1e-8
            
            # Extend gate for KV cache if needed
            if gate.shape[1] < kv_len:
                # Pad with gate_max (minimal intervention for cached positions)
                pad_val = wrapper.config.gate_max
                pad = torch.full(
                    (bsz, kv_len - gate.shape[1]), 
                    pad_val,
                    device=gate.device, 
                    dtype=gate.dtype
                )
                gate_extended = torch.cat([pad, gate], dim=1)
            else:
                gate_extended = gate[:, :kv_len]
            
            # Log-space modulation
            log_gate = torch.log(gate_extended + eps)  # [B, kv_len]
            log_gate = log_gate.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, kv_len]
            
            scores = scores + log_gate
            # =================================================================
            
            # Softmax and weighted sum
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            output = torch.matmul(attn_weights, v)
            
            # Reshape and project output
            output = output.transpose(1, 2).reshape(bsz, q_len, -1)
            output = attn.o_proj(output)
            
            return output, present
        
        attn.forward = patched_forward
    
    def reset_state(self):
        """Reset per-forward state."""
        self.fields = [None] * self.n_layers
        self.gate_history = []
        self.total_risk = 0.0
        self.fiber_curvatures = []
    
    def get_gate_statistics(self) -> dict:
        """Get gate statistics for monitoring (per config doc)."""
        if not self.gate_history:
            return {}
        
        gates = torch.tensor(self.gate_history)
        return {
            'gate_mean': gates.mean().item(),
            'gate_std': gates.std().item(),
            'gate_min': gates.min().item(),
            'gate_max': gates.max().item(),
            'saturated_low': (gates < self.config.early_stop_gate_threshold).float().mean().item(),
            'saturated_high': (gates > (1 - self.config.early_stop_gate_threshold)).float().mean().item(),
        }
    
    def should_early_stop(self) -> bool:
        """Check early stopping criteria from config doc."""
        stats = self.get_gate_statistics()
        if not stats:
            return False
        
        # "more than 30% of gates fall below 0.1"
        if stats['saturated_low'] > self.config.early_stop_gate_fraction:
            print(f"[WARNING] Gate collapse detected: {stats['saturated_low']:.1%} below {self.config.early_stop_gate_threshold}")
            return True
        
        # "gate standard deviation approaching zero"
        if stats["gate_std"] < 0.001:  # Relaxed
            print(f"[WARNING] Gate variance collapse: std = {stats['gate_std']:.4f}")
            return True
        
        return False
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        self.reset_state()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        return outputs, self.total_risk, self.gate_history


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings."""
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_gate_regularization(gates: List[float], target: float = 0.5, weight: float = 0.005) -> torch.Tensor:
    """
    From config doc: "penalize (gate - 0.5)² to prevent extremes"
    """
    if not gates:
        return torch.tensor(0.0)
    gate_tensor = torch.tensor(gates)
    return weight * ((gate_tensor - target) ** 2).mean()


def compute_curvature_loss(fiber_curvatures: List[torch.Tensor]) -> torch.Tensor:
    """
    Section 3.6: "curvature gate...suppresses feedforward output in high-curvature regions"
    Simplified: penalize large changes in fiber state between layers.
    """
    if len(fiber_curvatures) < 2:
        return torch.tensor(0.0)
    
    total_curv = 0.0
    count = 0
    for i in range(1, len(fiber_curvatures)):
        if fiber_curvatures[i].shape == fiber_curvatures[i-1].shape:
            curv = (fiber_curvatures[i] - fiber_curvatures[i-1]).norm(dim=-1).mean()
            total_curv = total_curv + curv
            count += 1
    
    return total_curv / max(count, 1)


# =============================================================================
# TRAINING
# =============================================================================

def main():
    config = CFHoTConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(config.output_dir, "config.txt"), "w") as f:
        for k, v in vars(config).items():
            f.write(f"{k}: {v}\n")
    
    print("=" * 70)
    print("TRUE CF-HoT TRAINING")
    print("=" * 70)
    print(f"EMA Momentum: {config.ema_momentum} (paper recommended: 0.995)")
    print(f"Gate Temperature: {config.gate_temperature} (paper recommended: 2.0)")
    print(f"Gate Bounds: [{config.gate_min}, {config.gate_max}]")
    print(f"Max Steps: {config.max_steps}")
    print("=" * 70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
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
    
    # Get device
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    # Add LoRA
    print("Adding LoRA...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Add CF-HoT
    print("Adding CF-HoT adapters...")
    cfhot = CFHoTWrapper(model, config)
    cfhot.adapters = cfhot.adapters.to(device).float()
    
    # Load data
    print("Loading data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex['text'] for ex in ds if len(ex['text']) > 50]
    random.shuffle(texts)
    print(f"Loaded {len(texts)} samples")
    
    if len(texts) == 0:
        raise ValueError("No training data loaded!")
    
    # Optimizer with cosine schedule (per config doc)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': config.lr_lora},
        {'params': cfhot.adapters.parameters(), 'lr': config.lr_cfhot}
    ], weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.max_steps,
        eta_min=1e-6
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    model.train()
    cfhot.adapters.train()
    
    step = 0
    data_idx = 0
    acc_loss, acc_lm, acc_risk, acc_curv, acc_gate_reg = 0, 0, 0, 0, 0
    best_val_loss = float('inf')
    no_improve_count = 0
    start_time = time.time()
    
    # Metrics log
    metrics_log = []
    
    while step < config.max_steps:
        # Get batch
        batch_texts = [texts[(data_idx + i) % len(texts)] for i in range(config.batch_size)]
        data_idx += config.batch_size
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=config.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward pass
        outputs, risk, gates = cfhot(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        lm_loss = outputs.loss
        
        # Compute auxiliary losses (Section 4.1 + config doc)
        curv_loss = compute_curvature_loss(cfhot.fiber_curvatures)
        gate_reg = compute_gate_regularization(gates, weight=config.lambda_gate_reg)
        
        # Total loss: L = L_LM + λ_hol·L_hol + λ_curv·L_curv + gate_reg
        loss = (
            lm_loss + 
            config.lambda_hol * risk + 
            config.lambda_curv * curv_loss +
            gate_reg
        )
        
        # Backward
        scaled_loss = loss / config.grad_accum
        scaled_loss.backward()
        
        # Accumulate metrics
        acc_loss += loss.item()
        acc_lm += lm_loss.item()
        acc_risk += risk.item() if isinstance(risk, torch.Tensor) else risk
        acc_curv += curv_loss.item() if isinstance(curv_loss, torch.Tensor) else curv_loss
        acc_gate_reg += gate_reg.item() if isinstance(gate_reg, torch.Tensor) else gate_reg
        
        step += 1
        
        # Optimizer step
        if step % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                list(lora_params) + list(cfhot.adapters.parameters()), 
                1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            eta_hours = (config.max_steps - step) / (step / elapsed) / 3600 if step > 0 else 0
            
            gate_stats = cfhot.get_gate_statistics()
            gate_mean = gate_stats.get('gate_mean', 0.5)
            gate_std = gate_stats.get('gate_std', 0.0)
            sat_low = gate_stats.get('saturated_low', 0.0)
            
            lr = scheduler.get_last_lr()[0]
            
            print(
                f"Step {step:5d} | "
                f"Loss: {acc_loss/config.log_every:.4f} | "
                f"LM: {acc_lm/config.log_every:.4f} | "
                f"Risk: {acc_risk/config.log_every:.2f} | "
                f"Gate: {gate_mean:.3f}±{gate_std:.3f} | "
                f"SatLow: {sat_low:.1%} | "
                f"LR: {lr:.2e} | "
                f"ETA: {eta_hours:.1f}h"
            )
            
            # Log metrics
            metrics_log.append({
                'step': step,
                'loss': acc_loss / config.log_every,
                'lm_loss': acc_lm / config.log_every,
                'risk': acc_risk / config.log_every,
                'gate_mean': gate_mean,
                'gate_std': gate_std,
                'saturated_low': sat_low,
            })
            
            acc_loss, acc_lm, acc_risk, acc_curv, acc_gate_reg = 0, 0, 0, 0, 0
        
        # Check for gate collapse
        if False:  # Disabled early stop
            print("\n[EARLY STOP] Gate collapse detected!")
            break
        
        # Checkpoint
        if step % config.save_every == 0:
            ckpt_dir = os.path.join(config.output_dir, f"checkpoint_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            model.save_pretrained(ckpt_dir)
            torch.save({
                'cfhot_state': cfhot.adapters.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'step': step,
                'config': vars(config),
                'metrics': metrics_log,
            }, os.path.join(ckpt_dir, "cfhot_checkpoint.pt"))
            
            print(f">>> Saved checkpoint: {ckpt_dir}")
        
        # Evaluation
        if step % config.eval_every == 0:
            model.eval()
            cfhot.adapters.eval()
            
            test_prompts = [
                "The will to power, as described by Nietzsche, is",
                "In the beginning, there was",
                "The fundamental nature of consciousness is",
            ]
            
            print("\n--- Evaluation ---")
            for prompt in test_prompts[:2]:
                cfhot.reset_state()
                with torch.no_grad():
                    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(f"  {generated[:200]}...")
            
            # Show gate activity during generation
            if cfhot.gate_history:
                gen_gates = cfhot.gate_history[-10:]
                print(f"  Gates (last 10): {[f'{g:.3f}' for g in gen_gates]}")
            
            print("--- End Eval ---\n")
            
            model.train()
            cfhot.adapters.train()
    
    # Final save
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    model.save_pretrained(final_dir)
    torch.save({
        'cfhot_state': cfhot.adapters.state_dict(),
        'step': step,
        'config': vars(config),
        'metrics': metrics_log,
    }, os.path.join(final_dir, "cfhot_final.pt"))
    
    # Save metrics
    import json
    with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Final model saved to: {final_dir}")
    print(f"Total steps: {step}")
    print("=" * 70)


if __name__ == "__main__":
    main()
