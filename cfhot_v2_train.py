#!/usr/bin/env python3
"""
CF-HoT v2 Training Script — Research-Informed Configuration
============================================================
Based on comprehensive analysis of:
- EMA momentum best practices (0.995 vs 0.9)
- Gate saturation prevention (temperature scaling, bounded gates)
- LLaMA-Adapter zero-init principles
- Early stopping criteria for gating mechanisms

Changes from original:
1. EMA momentum: 0.9 → 0.995 (prevents rapid gate collapse)
2. Gate temperature: 1.0 → 2.0 (softens sigmoid)
3. Bounded gates: [0.1, 0.9] (prevents complete suppression)
4. Gate regularization loss (penalizes extreme values)
5. Momentum warmup schedule
6. Comprehensive gate monitoring
7. Early stopping based on gate health

Author: Logan Napolitano + Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import math
import json
import time
import os
from pathlib import Path


# ============================================================================
# CONFIGURATION — RESEARCH-INFORMED DEFAULTS
# ============================================================================

@dataclass
class CFAdapterConfigV2:
    """Improved configuration based on research findings."""
    
    # Architecture (unchanged)
    d_model: int = 4096          # Will be set from base model
    n_layers: int = 32           # Will be set from base model
    d_fiber: int = 16            # Consistency subspace dimension
    d_control: int = 64          # Predictor hidden dimension
    
    # CRITICAL CHANGES
    momentum: float = 0.995      # Was 0.9 — now much slower accumulation
    momentum_warmup_steps: int = 100  # Warm up from 0.9 to 0.995
    gate_temperature: float = 2.0     # Softens sigmoid to prevent saturation
    gate_min: float = 0.1        # Minimum gate value (prevents complete suppression)
    gate_max: float = 0.9        # Maximum gate value (prevents no effect)
    
    # Loss weights (conservative)
    lambda_hol_loss: float = 0.01    # Was higher — reduced to prevent over-regularization
    lambda_curv_loss: float = 0.001  # Gentle curvature regularization
    lambda_gate_reg: float = 0.005   # NEW: penalizes gates far from 0.5
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100
    checkpoint_every: int = 25   # More frequent for short runs
    batch_size: int = 2
    gradient_accumulation: int = 4
    
    # Early stopping
    gate_collapse_threshold: float = 0.3  # Stop if >30% gates below gate_min


# ============================================================================
# IMPROVED CONTROL FIELD MODULE
# ============================================================================

class ControlFieldAdapterV2(nn.Module):
    """
    Improved CF adapter with:
    - Temperature-scaled gating
    - Bounded output range
    - Slower EMA accumulation
    """
    
    def __init__(self, config: CFAdapterConfigV2, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Fiber projection
        self.fiber_proj = nn.Linear(config.d_model, config.d_fiber)
        
        # Holonomy predictor
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model + config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1),
            nn.Softplus()
        )
        
        # IMPORTANT: Initialize predictor output near zero (zero-init principle)
        nn.init.zeros_(self.predictor[-2].bias)
        nn.init.normal_(self.predictor[-2].weight, std=0.001)  # Smaller init
        
        # Learnable gate scale (start at 1.0)
        self.lambda_gate = nn.Parameter(torch.tensor(1.0))
    
    def get_momentum(self, step: int) -> float:
        """Momentum warmup: start at 0.9, ramp to target over warmup period."""
        if step < self.config.momentum_warmup_steps:
            progress = step / self.config.momentum_warmup_steps
            return 0.9 + progress * (self.config.momentum - 0.9)
        return self.config.momentum
    
    def bounded_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid bounded to [gate_min, gate_max] to prevent saturation."""
        base = torch.sigmoid(x / self.config.gate_temperature)
        return self.config.gate_min + (self.config.gate_max - self.config.gate_min) * base
    
    def forward(
        self, 
        hidden: torch.Tensor, 
        prev_field: Optional[torch.Tensor],
        step: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [batch, seq, d_model] - attention output
            prev_field: [batch, seq] - previous control field (or None)
            step: current training step (for momentum warmup)
        
        Returns:
            gate: [batch, seq] - bounded attention gate
            field: [batch, seq] - accumulated control field
            risk: [batch, seq] - predicted holonomy risk
            fiber: [batch, seq, d_fiber] - fiber states
        """
        batch, seq_len, _ = hidden.shape
        
        # Project to fiber space
        fiber = self.fiber_proj(hidden)
        
        # Predict risk (holonomy increment)
        combined = torch.cat([hidden, fiber], dim=-1)
        risk = self.predictor(combined).squeeze(-1)
        risk = torch.clamp(risk, 0, 1)  # Bounded risk
        
        # Get current momentum (with warmup)
        momentum = self.get_momentum(step)
        
        # EMA accumulation with shape handling
        if prev_field is None or prev_field.shape != risk.shape:
            field = risk
        else:
            field = momentum * prev_field + (1 - momentum) * risk
        
        # Bounded gate computation
        gate = self.bounded_sigmoid(-self.lambda_gate * field)
        
        return gate, field, risk, fiber


class CFAdapterStackV2(nn.Module):
    """Stack of CF adapters for all layers."""
    
    def __init__(self, config: CFAdapterConfigV2):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleList([
            ControlFieldAdapterV2(config, i) for i in range(config.n_layers)
        ])
    
    def __getitem__(self, idx):
        return self.adapters[idx]
    
    def __len__(self):
        return len(self.adapters)


# ============================================================================
# HOOKED MODEL WRAPPER
# ============================================================================

class CFHoTLlamaHookedV2:
    """
    Wrapper that hooks CF adapters into a frozen LLaMA model.
    Improved with step tracking for momentum warmup.
    """
    
    def __init__(self, base_model, config: CFAdapterConfigV2):
        self.base_model = base_model
        self.config = config
        self.cf_adapters = CFAdapterStackV2(config)
        self.control_field = None
        self.current_step = 0
        self.hooks = []
        self._install_hooks()
        
        n_params = sum(p.numel() for p in self.cf_adapters.parameters())
        print(f"[CFHoT-v2] Adapter params: {n_params:,}")
        print(f"[CFHoT-v2] Momentum: {config.momentum} (warmup from 0.9)")
        print(f"[CFHoT-v2] Gate temp: {config.gate_temperature}")
        print(f"[CFHoT-v2] Gate range: [{config.gate_min}, {config.gate_max}]")
    
    def _install_hooks(self):
        """Install forward hooks on attention layers."""
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            hook = layer.self_attn.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output
            
            gate, field, risk, fiber = self.cf_adapters[layer_idx](
                attn_output,
                self.control_field,
                self.current_step
            )
            
            self.control_field = field.detach()
            self.last_gate = gate
            self.last_risk = risk
            
            # Apply gate
            gated_output = attn_output * gate.unsqueeze(-1)
            
            if isinstance(output, tuple):
                return (gated_output,) + output[1:]
            return gated_output
        
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def train(self):
        self.cf_adapters.train()
    
    def eval(self):
        self.cf_adapters.eval()
    
    def reset_field(self):
        self.control_field = None


# ============================================================================
# METRICS AND MONITORING
# ============================================================================

def compute_gate_metrics(model: CFHoTLlamaHookedV2) -> Dict[str, float]:
    """Compute comprehensive gate health metrics."""
    if not hasattr(model, 'last_gate'):
        return {}
    
    gate = model.last_gate
    config = model.config
    
    metrics = {
        'gate_mean': gate.mean().item(),
        'gate_std': gate.std().item(),
        'gate_min': gate.min().item(),
        'gate_max': gate.max().item(),
        'saturated_low': (gate < config.gate_min + 0.05).float().mean().item(),
        'saturated_high': (gate > config.gate_max - 0.05).float().mean().item(),
        'healthy_range': ((gate > 0.3) & (gate < 0.7)).float().mean().item(),
    }
    
    if hasattr(model, 'last_risk'):
        metrics['risk_mean'] = model.last_risk.mean().item()
        metrics['risk_max'] = model.last_risk.max().item()
    
    return metrics


def check_early_stopping(metrics: Dict[str, float], config: CFAdapterConfigV2) -> Tuple[bool, str]:
    """Check if training should stop early based on gate health."""
    if metrics.get('saturated_low', 0) > config.gate_collapse_threshold:
        return True, f"Gate collapse: {metrics['saturated_low']:.1%} gates below minimum"
    
    if metrics.get('gate_std', 1) < 0.01:
        return True, f"Gate variance collapsed: std={metrics['gate_std']:.4f}"
    
    return False, ""


def compute_gate_regularization(model: CFHoTLlamaHookedV2) -> torch.Tensor:
    """Penalize gates far from neutral (0.5)."""
    if not hasattr(model, 'last_gate'):
        return torch.tensor(0.0)
    
    gate = model.last_gate
    target = (model.config.gate_min + model.config.gate_max) / 2  # 0.5 for default
    return ((gate - target) ** 2).mean()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(tokenizer, seq_len: int = 512, max_samples: int = 1000):
    """Load WikiText-2 for quick training."""
    print("[Data] Loading WikiText-2...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    all_tokens = []
    for example in dataset:
        if example['text'].strip():
            tokens = tokenizer.encode(example['text'])
            all_tokens.extend(tokens)
        if len(all_tokens) >= max_samples * seq_len:
            break
    
    tokens = torch.tensor(all_tokens[:max_samples * seq_len], dtype=torch.long)
    tokens = tokens[:len(tokens) // seq_len * seq_len].view(-1, seq_len)
    
    print(f"[Data] Loaded {len(tokens)} sequences of length {seq_len}")
    return tokens


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_cfhot_v2(
    model_path: str = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5',
    output_dir: str = './results/cfhot_v2',
    config: Optional[CFAdapterConfigV2] = None
):
    """
    Train CF-HoT v2 adapters with improved configuration.
    """
    print("=" * 70)
    print("CF-HoT v2 TRAINING — RESEARCH-INFORMED CONFIGURATION")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Config
    if config is None:
        config = CFAdapterConfigV2()
    
    print(f"\n[Config]")
    print(f"  Momentum: {config.momentum} (warmup from 0.9 over {config.momentum_warmup_steps} steps)")
    print(f"  Gate temperature: {config.gate_temperature}")
    print(f"  Gate bounds: [{config.gate_min}, {config.gate_max}]")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Lambda holonomy: {config.lambda_hol_loss}")
    print(f"  Lambda curvature: {config.lambda_curv_loss}")
    print(f"  Lambda gate reg: {config.lambda_gate_reg}")
    
    # Load tokenizer
    print(f"\n[Model] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model (frozen, quantized)
    print("[Model] Loading base model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16,
    )
    
    # Update config with model dimensions
    config.d_model = base_model.config.hidden_size
    config.n_layers = base_model.config.num_hidden_layers
    
    # Create CF-HoT wrapper
    print("[Model] Creating CF-HoT v2 adapters...")
    cf_model = CFHoTLlamaHookedV2(base_model, config)
    cf_model.cf_adapters = cf_model.cf_adapters.to(device).half()
    cf_model.train()
    
    # Load data
    train_tokens = load_training_data(tokenizer, seq_len=512, max_samples=500)
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_tokens, train_tokens.clone()),
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Optimizer with cosine schedule
    optimizer = torch.optim.AdamW(
        cf_model.cf_adapters.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.max_steps,
        eta_min=config.learning_rate * 0.01
    )
    
    # Training history
    history = {
        'step': [],
        'lm_loss': [],
        'total_loss': [],
        'gate_mean': [],
        'gate_std': [],
        'risk_mean': [],
        'saturated_low': [],
        'lr': [],
    }
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    train_iter = iter(train_loader)
    optimizer.zero_grad()
    
    for step in range(1, config.max_steps + 1):
        cf_model.current_step = step
        
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Reset control field for each batch
        cf_model.reset_field()
        
        # Forward pass
        outputs = base_model(input_ids=x, labels=y)
        lm_loss = outputs.loss
        
        # Auxiliary losses
        gate_reg = compute_gate_regularization(cf_model)
        
        # Risk regularization (if available)
        risk_reg = torch.tensor(0.0, device=device)
        if hasattr(cf_model, 'last_risk'):
            risk_reg = cf_model.last_risk.mean()
        
        # Total loss
        total_loss = (
            lm_loss + 
            config.lambda_hol_loss * risk_reg +
            config.lambda_gate_reg * gate_reg
        )
        
        # Backward
        (total_loss / config.gradient_accumulation).backward()
        
        if step % config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(cf_model.cf_adapters.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if step % 10 == 0 or step == 1:
            metrics = compute_gate_metrics(cf_model)
            current_lr = scheduler.get_last_lr()[0]
            current_momentum = cf_model.cf_adapters.adapters[0].get_momentum(step)
            
            print(f"Step {step:4d} | "
                  f"LM: {lm_loss.item():.4f} | "
                  f"Gate: {metrics.get('gate_mean', 0):.3f}±{metrics.get('gate_std', 0):.3f} | "
                  f"Risk: {metrics.get('risk_mean', 0):.2f} | "
                  f"Sat↓: {metrics.get('saturated_low', 0):.1%} | "
                  f"Mom: {current_momentum:.3f} | "
                  f"LR: {current_lr:.2e}")
            
            # Record history
            history['step'].append(step)
            history['lm_loss'].append(lm_loss.item())
            history['total_loss'].append(total_loss.item())
            history['gate_mean'].append(metrics.get('gate_mean', 0))
            history['gate_std'].append(metrics.get('gate_std', 0))
            history['risk_mean'].append(metrics.get('risk_mean', 0))
            history['saturated_low'].append(metrics.get('saturated_low', 0))
            history['lr'].append(current_lr)
            
            # Early stopping check
            should_stop, reason = check_early_stopping(metrics, config)
            if should_stop:
                print(f"\n⚠️  EARLY STOPPING: {reason}")
                break
        
        # Checkpoint
        if step % config.checkpoint_every == 0:
            ckpt_path = output_path / f'cfhot_v2_step_{step}.pt'
            torch.save({
                'step': step,
                'adapter_state_dict': cf_model.cf_adapters.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'history': history,
            }, ckpt_path)
            print(f"[Save] {ckpt_path}")
    
    # Final save
    final_path = output_path / 'cfhot_v2_final.pt'
    torch.save({
        'step': step,
        'adapter_state_dict': cf_model.cf_adapters.state_dict(),
        'config': config,
        'history': history,
    }, final_path)
    
    # Save history as JSON
    history_path = output_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final checkpoint: {final_path}")
    print(f"Training history: {history_path}")
    
    # Summary
    if history['gate_mean']:
        print(f"\nGate Statistics:")
        print(f"  Initial: {history['gate_mean'][0]:.3f} ± {history['gate_std'][0]:.3f}")
        print(f"  Final:   {history['gate_mean'][-1]:.3f} ± {history['gate_std'][-1]:.3f}")
        print(f"  Saturated (low): {history['saturated_low'][-1]:.1%}")
    
    return cf_model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_generation(cf_model, tokenizer, base_model, prompts: List[str]):
    """Generate from prompts and display results."""
    print("\n" + "=" * 70)
    print("GENERATION EVALUATION")
    print("=" * 70)
    
    cf_model.eval()
    
    for prompt in prompts:
        cf_model.reset_field()
        inputs = tokenizer(prompt, return_tensors='pt').to(base_model.device)
        
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # Deterministic for comparison
                pad_token_id=tokenizer.eos_token_id
            )
        
        print(f"\n>> {prompt}")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("-" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CF-HoT v2 Training')
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/nvme2/ubermesnchetien4/models/merged-final-v5')
    parser.add_argument('--output_dir', type=str, default='./results/cfhot_v2')
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.995)
    parser.add_argument('--gate_temp', type=float, default=2.0)
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Create config with CLI overrides
    config = CFAdapterConfigV2()
    config.max_steps = args.max_steps
    config.learning_rate = args.lr
    config.momentum = args.momentum
    config.gate_temperature = args.gate_temp
    
    # Train
    cf_model, history = train_cfhot_v2(
        model_path=args.model_path,
        output_dir=args.output_dir,
        config=config
    )
    
    # Optional evaluation
    if args.eval:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        prompts = [
            "The will to power, as described by Nietzsche, is",
            "The fundamental problem with artificial general intelligence is",
            "To solve a differential equation, one must first",
        ]
        
        evaluate_generation(cf_model, tokenizer, cf_model.base_model, prompts)
