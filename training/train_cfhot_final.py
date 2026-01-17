#!/usr/bin/env python3
"""CF-HoT 24h Training - WORKING"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os, time, random

MODEL_PATH = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
OUTPUT_DIR = "./results/cfhot_24h"
MAX_STEPS = 50000
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_LENGTH = 512
LOG_EVERY = 10
SAVE_EVERY = 2000
EVAL_EVERY = 500

class CFAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fiber = nn.Linear(d_model, 16, bias=False)
        self.pred = nn.Sequential(nn.Linear(d_model + 16, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())
        self.lam = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.fiber.weight, std=0.01)
        nn.init.zeros_(self.pred[-2].bias)
    
    def forward(self, h, prev=None):
        B, S, D = h.shape
        h32 = h.float()
        fib = self.fiber(h32)
        risk = self.pred(torch.cat([h32, fib], -1)).squeeze(-1)
        if prev is None:
            field = 0.1 * risk
        else:
            if prev.shape[1] < S:
                prev = F.pad(prev, (0, S - prev.shape[1]))
            elif prev.shape[1] > S:
                prev = prev[:, :S]
            field = 0.9 * prev.float() + 0.1 * risk
        gate = torch.sigmoid(-self.lam * field)
        return gate.to(h.dtype), field.to(h.dtype), risk.mean()

class CFHoT(nn.Module):
    def __init__(self, n_layers, d_model):
        super().__init__()
        self.adapters = nn.ModuleList([CFAdapter(d_model) for _ in range(n_layers)])
    
    def forward(self, hidden_states):
        total_risk = 0.0
        field = None
        gates = []
        for i, ad in enumerate(self.adapters):
            if i < len(hidden_states):
                gate, field, risk = ad(hidden_states[i], field)
                total_risk = total_risk + risk
                gates.append(gate.mean().item())
        return total_risk, sum(gates)/len(gates) if gates else 0.5

def unlikelihood_loss(logits, input_ids, window=10):
    B, S, V = logits.shape
    loss = torch.tensor(0.0, device=logits.device)
    cnt = 0
    for b in range(B):
        for t in range(window, S):
            recent = set(input_ids[b, t-window:t].tolist())
            cur = input_ids[b, t].item()
            if cur in recent and cur != 0:
                p = F.softmax(logits[b, t-1].float(), -1)[cur]
                loss = loss - torch.log(1 - p + 1e-8)
                cnt += 1
    return loss / max(cnt, 1)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    
    print("Loading model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    # Get the device the model is actually on
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    print("Adding LoRA...")
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    
    print("Adding CF-HoT...")
    cfhot = CFHoT(model.config.num_hidden_layers, model.config.hidden_size).to(device).float()
    print(f"CF-HoT params: {sum(p.numel() for p in cfhot.parameters()):,}")
    
    print("Loading data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex['text'] for ex in ds if len(ex['text']) > 50]
    random.shuffle(texts)
    print(f"Loaded {len(texts)} samples")
    
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': lora_params, 'lr': 2e-5}, {'params': cfhot.parameters(), 'lr': 1e-4}], weight_decay=0.01)
    
    print("="*60)
    print(f"TRAINING | Steps: {MAX_STEPS} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}")
    print("="*60)
    
    model.train()
    cfhot.train()
    step = 0
    idx = 0
    acc_loss, acc_lm, acc_risk, acc_ul = 0, 0, 0, 0
    t0 = time.time()
    
    while step < MAX_STEPS:
        batch = [texts[(idx + i) % len(texts)] for i in range(BATCH_SIZE)]
        idx += BATCH_SIZE
        
        enc = tok(batch, truncation=True, max_length=MAX_LENGTH, padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        
        out = model(input_ids=ids, attention_mask=mask, labels=ids, output_hidden_states=True)
        lm = out.loss
        risk, gate = cfhot(out.hidden_states)
        ul = unlikelihood_loss(out.logits, ids)
        
        loss = lm + 1e-4 * risk + 0.1 * ul
        (loss / GRAD_ACCUM).backward()
        
        acc_loss += loss.item()
        acc_lm += lm.item()
        acc_risk += risk.item() if isinstance(risk, torch.Tensor) else risk
        acc_ul += ul.item()
        
        step += 1
        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + list(cfhot.parameters()), 1.0)
            opt.step()
            opt.zero_grad()
        
        if step % LOG_EVERY == 0:
            eta = (MAX_STEPS - step) / (step / (time.time() - t0)) / 3600
            print(f"Step {step:5d} | Loss: {acc_loss/LOG_EVERY:.4f} | LM: {acc_lm/LOG_EVERY:.4f} | Risk: {acc_risk/LOG_EVERY:.1f} | UL: {acc_ul/LOG_EVERY:.4f} | Gate: {gate:.3f} | ETA: {eta:.1f}h")
            acc_loss, acc_lm, acc_risk, acc_ul = 0, 0, 0, 0
        
        if step % SAVE_EVERY == 0:
            p = os.path.join(OUTPUT_DIR, f"ckpt_{step}")
            os.makedirs(p, exist_ok=True)
            model.save_pretrained(p)
            torch.save(cfhot.state_dict(), os.path.join(p, "cfhot.pt"))
            print(f">>> Saved {p}")
        
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                inp = tok("The will to power, as described by Nietzsche, is", return_tensors='pt').input_ids.to(device)
                out = model.generate(inp, max_new_tokens=80, do_sample=True, temperature=0.8, pad_token_id=tok.eos_token_id)
                print(f"--- Eval: {tok.decode(out[0], skip_special_tokens=True)[:180]}...")
            model.train()
    
    p = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(p, exist_ok=True)
    model.save_pretrained(p)
    torch.save(cfhot.state_dict(), os.path.join(p, "cfhot.pt"))
    print(f"DONE! Saved to {p}")

if __name__ == "__main__":
    main()
