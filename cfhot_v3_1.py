#!/usr/bin/env python3
"""
CF-HoT v3.1 — Float32 Stability Fix
===================================
1. Force-casts hidden states to Float32 before Adapter processing (Prevents NaN).
2. Clamps Risk values to [0, 10] (Prevents Exploding Gradients).
3. Properly aggregates Risk Loss across all layers (Fixes detached gradient bug).
4. Adds Gradient Clipping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from dataclasses import dataclass
import sys

# ================= CONFIGURATION =================
@dataclass
class CFConfig:
    d_model: int = 4096
    n_layers: int = 32
    d_fiber: int = 16
    d_control: int = 64
    
    momentum: float = 0.95
    gate_temp: float = 2.0
    gate_min: float = 0.1
    gate_max: float = 1.0
    
    lambda_hol: float = 0.05
    lr: float = 2e-4
    max_steps: int = 600
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
            nn.Softplus()
        )
        
        nn.init.zeros_(self.predictor[-2].bias)
        nn.init.zeros_(self.predictor[-2].weight)
        self.risk_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, hidden, prev_field):
        # STABILITY: Cast to Float32
        h_f32 = hidden.to(torch.float32)
        
        # 1. Compute Risk
        fiber = self.fiber_proj(h_f32)
        risk = self.predictor(torch.cat([h_f32, fiber], dim=-1)).squeeze(-1)
        
        # STABILITY: Clamp Risk
        risk = torch.clamp(risk, 0.0, 10.0)
        
        # 2. Update Control Field (EMA)
        if prev_field is None:
            field = risk
        else:
            # Handle shape mismatch just in case (e.g. prompt length change)
            if prev_field.shape != risk.shape:
                field = risk
            else:
                field = self.config.momentum * prev_field + (1 - self.config.momentum) * risk
            
        # 3. Compute Gate
        scaled_risk = field * self.risk_scale
        suppression = torch.tanh(scaled_risk)
        gate_val = 1.0 - (suppression * (1.0 - self.config.gate_min))
        
        return gate_val.to(hidden.dtype), field, risk

class CFModel(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.config = config
        self.adapters = nn.ModuleList([CFAdapter(config, i) for i in range(config.n_layers)])
        self.field = None
        self.hooks = []
        self.layer_risks = [] # Store risks for loss calculation
        
    def setup(self):
        for i, layer in enumerate(self.base.model.layers):
            self.hooks.append(layer.self_attn.register_forward_hook(self.make_hook(i)))
            
    def make_hook(self, idx):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            
            # Pass hidden state to adapter
            gate, field, risk = self.adapters[idx](out, self.field)
            
            # Update state
            self.field = field # Keep graph connected? No, usually detach for EMA to prevent massive backprop through layers
            # Actually, standard EMA detaches prev_step, but here we are depth-accumulating?
            # Phase B detached. Let's DETACH field to match Phase B.
            self.field = field.detach() 
            
            # Store risk for loss (Keep graph connected here!)
            self.layer_risks.append(risk)
            
            self.last_gate = gate
            return (out * gate.unsqueeze(-1),) + (output[1:] if isinstance(output, tuple) else ())
        return hook

    def reset(self): 
        self.field = None
        self.layer_risks = []
    
    def train(self): 
        for a in self.adapters: a.train()
    def eval(self):
        for a in self.adapters: a.eval()

# ================= TRAINING =================
def train():
    print(">>> LOADING MODEL (Float16 + Float32 Mixed Precision)...")
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
    
    # Keep adapters in Float32 for stability
    model.adapters.to('cuda').to(torch.float32)
    
    print(">>> LOADING DATA...")
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tokens = []
    for txt in data['text']:
        if txt.strip(): tokens.extend(tok.encode(txt))
        if len(tokens) > 1000 * 512: break
    
    t_tensor = torch.tensor(tokens[:(len(tokens)//512)*512]).view(-1, 512)
    loader = DataLoader(torch.utils.data.TensorDataset(t_tensor), batch_size=config.batch_size, shuffle=True)
    
    opt = torch.optim.AdamW(model.adapters.parameters(), lr=config.lr)
    
    print(f">>> STARTING STABLE RUN ({config.max_steps} steps)")
    iter_data = iter(loader)
    
    for step in range(1, config.max_steps+1):
        try: batch = next(iter_data)[0].to('cuda')
        except: 
            iter_data = iter(loader)
            batch = next(iter_data)[0].to('cuda')
            
        model.reset()
        out = base(input_ids=batch, labels=batch)
        
        # Compute Risk Loss from collected risks (connected graph)
        if len(model.layer_risks) > 0:
            avg_risk = torch.stack(model.layer_risks).mean()
        else:
            avg_risk = torch.tensor(0.0).to('cuda')

        loss = out.loss + config.lambda_hol * avg_risk
        
        if torch.isnan(loss):
            print(f"!!! NAN DETECTED AT STEP {step} !!!")
            break
            
        loss.backward()
        
        if step % config.grad_acc == 0:
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
        if step % 10 == 0:
            g_mean = model.last_gate.float().mean().item()
            r_mean = avg_risk.item()
            print(f"Step {step:3d} | Loss: {out.loss.item():.4f} | Gate: {g_mean:.4f} | Risk: {r_mean:.4f}")
            
        if step % config.ckpt_every == 0:
            torch.save(model.adapters.state_dict(), f"results/cfhot_v3_1_step{step}.pt")
            
    print(">>> TRAINING COMPLETE. SAVING FINAL.")
    torch.save(model.adapters.state_dict(), "results/cfhot_v3_1_final.pt")

if __name__ == "__main__":
    train()
