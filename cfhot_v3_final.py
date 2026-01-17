#!/usr/bin/env python3
"""
CF-HoT v3 FINAL — The "Goldilocks" Configuration
================================================
Fixes the "dimmer switch" bug from v2.
1. Gate Init: Starts at 1.0 (Pass-through), not 0.5.
2. Range: [0.1, 1.0]. We allow full attention if risk is low.
3. Momentum: 0.95. Fast enough to learn in 500 steps, slow enough to prevent 5000-step collapse.
4. Logic: Inverse Risk Gating (Risk increases -> Gate decreases).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import json
from pathlib import Path

# ================= CONFIGURATION =================
@dataclass
class CFConfig:
    d_model: int = 4096
    n_layers: int = 32
    d_fiber: int = 16
    d_control: int = 64
    
    # THE GOLDILOCKS PARAMS
    momentum: float = 0.95       # Balanced for 300-1000 step runs
    gate_temp: float = 2.0       # Softens the curve
    gate_min: float = 0.1        # Don't let it close completely
    gate_max: float = 1.0        # Allow FULL attention (fix for v2 bug)
    
    # Loss Weights
    lambda_hol: float = 0.05     # Stronger signal to learn quickly
    lambda_curv: float = 0.001
    
    # Training
    lr: float = 2e-4             # Slightly higher for adapter
    max_steps: int = 600         # The new target (not 100, not 5000)
    batch_size: int = 2
    grad_acc: int = 4
    ckpt_every: int = 100

# ================= ADAPTER MODULE =================
class CFAdapter(nn.Module):
    def __init__(self, config: CFConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.fiber_proj = nn.Linear(config.d_model, config.d_fiber)
        
        # Risk Predictor
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model + config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1),
            nn.Softplus() # Outputs positive risk
        )
        
        # Zero-init the predictor so we start with 0 risk
        nn.init.zeros_(self.predictor[-2].bias)
        nn.init.zeros_(self.predictor[-2].weight)
        
        # Sensitivity parameter
        self.risk_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, hidden, prev_field):
        # 1. Compute Risk
        fiber = self.fiber_proj(hidden)
        risk = self.predictor(torch.cat([hidden, fiber], dim=-1)).squeeze(-1)
        
        # 2. Update Control Field (EMA)
        if prev_field is None or prev_field.shape != risk.shape:
            field = risk
        else:
            field = self.config.momentum * prev_field + (1 - self.config.momentum) * risk
            
        # 3. Compute Gate (Inverse: High Field -> Low Gate)
        # We map Field=0 -> Gate=1.0 (Pass-through)
        # We map Field=High -> Gate=0.1 (Suppressed)
        
        scaled_risk = field * self.risk_scale
        # Sigmoid(0) = 0.5, we want 1.0. 
        # So we use: Gate = 1.0 - (Bounded Sigmoid of Risk)
        
        # Simple bounded inverse logic
        suppression = torch.sigmoid(scaled_risk / self.config.gate_temp) # 0.5 to 1.0
        # Remap suppression: if risk is 0, suppression should be 0.
        # Let's use Tanh for safer 0-start
        suppression = torch.tanh(scaled_risk) # 0 to 1
        
        gate_val = 1.0 - (suppression * (1.0 - self.config.gate_min))
        
        return gate_val, field, risk

class CFModel(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.config = config
        self.adapters = nn.ModuleList([CFAdapter(config, i) for i in range(config.n_layers)])
        self.field = None
        self.hooks = []
        
    def setup(self):
        for i, layer in enumerate(self.base.model.layers):
            self.hooks.append(layer.self_attn.register_forward_hook(self.make_hook(i)))
            
    def make_hook(self, idx):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            gate, field, risk = self.adapters[idx](out, self.field)
            self.field = field.detach()
            self.last_gate = gate
            # Multiplicative gating starting at 1.0
            return (out * gate.unsqueeze(-1),) + (output[1:] if isinstance(output, tuple) else ())
        return hook

    def train(self): 
        for a in self.adapters: a.train()
    def eval(self):
        for a in self.adapters: a.eval()
    def reset(self): self.field = None

# ================= TRAINING =================
def train():
    print(">>> LOADING MODEL...")
    model_path = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'
    tok = AutoTokenizer.from_pretrained(model_path)
    if not tok.pad_token: tok.pad_token = tok.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        model_path, 
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map='auto'
    )
    
    config = CFConfig(d_model=base.config.hidden_size, n_layers=base.config.num_hidden_layers)
    model = CFModel(base, config)
    model.setup()
    model.train()
    model.adapters.to('cuda').half()
    
    print(">>> LOADING DATA...")
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tokens = []
    for txt in data['text']:
        if txt.strip(): tokens.extend(tok.encode(txt))
        if len(tokens) > 1000 * 512: break
    
    t_tensor = torch.tensor(tokens[:(len(tokens)//512)*512]).view(-1, 512)
    loader = DataLoader(torch.utils.data.TensorDataset(t_tensor), batch_size=config.batch_size, shuffle=True)
    
    opt = torch.optim.AdamW(model.adapters.parameters(), lr=config.lr)
    
    print(f">>> STARTING GOLDILOCKS RUN ({config.max_steps} steps)")
    iter_data = iter(loader)
    
    for step in range(1, config.max_steps+1):
        try: batch = next(iter_data)[0].to('cuda')
        except: 
            iter_data = iter(loader)
            batch = next(iter_data)[0].to('cuda')
            
        model.reset()
        out = base(input_ids=batch, labels=batch)
        
        # Loss: LM + Risk Penalty
        # We want to minimize Risk (Field magnitude)
        risk_loss = model.field.mean() if model.field is not None else 0
        loss = out.loss + config.lambda_hol * risk_loss
        
        loss.backward()
        
        if step % config.grad_acc == 0:
            opt.step()
            opt.zero_grad()
            
        if step % 10 == 0:
            g_mean = model.last_gate.mean().item()
            print(f"Step {step:3d} | Loss: {out.loss.item():.4f} | Gate: {g_mean:.4f} | Risk: {risk_loss:.4f}")
            
        if step % config.ckpt_every == 0:
            torch.save(model.adapters.state_dict(), f"results/cfhot_v3_step{step}.pt")
            
    print(">>> TRAINING COMPLETE. SAVING FINAL.")
    torch.save(model.adapters.state_dict(), "results/cfhot_v3_final.pt")

if __name__ == "__main__":
    train()
