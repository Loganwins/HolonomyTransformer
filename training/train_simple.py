#!/usr/bin/env python3
"""CF-HoT Training - Simplified v2"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from dataclasses import dataclass
import os, time

MODEL_PATH = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
OUTPUT_DIR = "./results/cfhot_simple"
MAX_STEPS = 50000
BATCH_SIZE = 2
GRAD_ACCUM = 4
LOG_EVERY = 10
SAVE_EVERY = 2000

@dataclass  
class CFConfig:
    d_model: int = 4096
    n_layers: int = 32
    d_fiber: int = 16
    d_control: int = 64

class CFAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fiber = nn.Linear(cfg.d_model, cfg.d_fiber, bias=False)
        self.pred = nn.Sequential(nn.Linear(cfg.d_model + cfg.d_fiber, cfg.d_control), nn.GELU(), 
                                   nn.Linear(cfg.d_control, 1), nn.Softplus())
        self.lam = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, h, prev=None):
        B, S, _ = h.shape
        h32 = h.float()
        fib = self.fiber(h32)
        risk = self.pred(torch.cat([h32, fib], -1)).squeeze(-1)
        field = 0.1 * risk if prev is None else 0.9 * prev.float() + 0.1 * risk
        gate = torch.sigmoid(-self.lam * field)
        return gate.to(h.dtype), field.to(h.dtype), risk.mean()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map='auto')
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                              lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    
    cf_cfg = CFConfig(d_model=model.config.hidden_size, n_layers=model.config.num_hidden_layers)
    cf_adapters = nn.ModuleList([CFAdapter(cf_cfg) for _ in range(cf_cfg.n_layers)]).cuda().float()
    print(f"CF-HoT params: {sum(p.numel() for p in cf_adapters.parameters()):,}")
    
    print("Loading data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda x: len(x['text']) > 50)
    texts = [x['text'] for x in ds]
    print(f"Samples: {len(texts)}")
    
    opt = torch.optim.AdamW(
        [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 2e-5},
         {'params': cf_adapters.parameters(), 'lr': 1e-4}]
    )
    
    print("="*60)
    print(f"Training | Steps: {MAX_STEPS}")
    print("="*60)
    
    model.train()
    cf_adapters.train()
    step = 0
    accum = 0
    t0 = time.time()
    idx = 0
    
    while step < MAX_STEPS:
        # Manual batching
        batch_texts = texts[idx:idx+BATCH_SIZE]
        idx = (idx + BATCH_SIZE) % len(texts)
        
        # Tokenize and move to GPU immediately
        enc = tok(batch_texts, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        input_ids = enc['input_ids'].cuda()
        attn_mask = enc['attention_mask'].cuda()
        labels = input_ids.clone()
        
        # Forward
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels, output_hidden_states=True)
        lm_loss = out.loss
        
        # CF risk
        total_risk = torch.tensor(0.0, device='cuda')
        field = None
        for i, ad in enumerate(cf_adapters):
            if i < len(out.hidden_states):
                gate, field, risk = ad(out.hidden_states[i], field)
                total_risk = total_risk + risk
        
        loss = lm_loss + 1e-4 * total_risk
        (loss / GRAD_ACCUM).backward()
        accum += loss.item()
        
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cf_adapters.parameters()), 1.0)
            opt.step()
            opt.zero_grad()
        
        step += 1
        
        if step % LOG_EVERY == 0:
            eta = (MAX_STEPS - step) / (step / (time.time() - t0)) / 3600
            print(f"Step {step:5d} | Loss: {accum/LOG_EVERY:.4f} | LM: {lm_loss.item():.4f} | "
                  f"Risk: {total_risk.item():.1f} | Gate: {gate.mean().item():.3f} | ETA: {eta:.1f}h")
            accum = 0
        
        if step % SAVE_EVERY == 0:
            p = os.path.join(OUTPUT_DIR, f"ckpt_{step}")
            os.makedirs(p, exist_ok=True)
            model.save_pretrained(p)
            torch.save(cf_adapters.state_dict(), os.path.join(p, "cf.pt"))
            print(f"Saved: {p}")
    
    print("Done!")

if __name__ == "__main__":
    main()
