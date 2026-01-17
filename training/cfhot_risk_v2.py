#!/usr/bin/env python3
"""
CF-HoT Phase 1: Train Risk Predictor (FIXED)
=============================================
Fixed: Class weighting to handle imbalanced data (most tokens aren't repeats)
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
from dataclasses import dataclass
from typing import Tuple

@dataclass 
class Config:
    model_path: str = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
    output_dir: str = "./results/cfhot_risk_v2"
    d_fiber: int = 16
    d_control: int = 64
    max_steps: int = 3000
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 256
    lr_lora: float = 2e-5
    lr_predictor: float = 1e-4
    weight_decay: float = 0.01
    rep_window: int = 32
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 200


class RiskPredictor(nn.Module):
    def __init__(self, d_model: int, n_layers: int, config: Config):
        super().__init__()
        self.config = config
        self.n_layers = n_layers
        
        self.fiber_projs = nn.ModuleList([
            nn.Linear(d_model, config.d_fiber, bias=False)
            for _ in range(n_layers)
        ])
        
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        
        # Output LOGITS, not probabilities
        self.predictor = nn.Sequential(
            nn.Linear(config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1)
            # NO sigmoid here - we'll use BCEWithLogitsLoss
        )
        
        for proj in self.fiber_projs:
            nn.init.normal_(proj.weight, std=0.02)
    
    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        fibers = []
        for i, (proj, h) in enumerate(zip(self.fiber_projs, hidden_states)):
            if i < len(hidden_states):
                fiber = proj(h.float())
                fibers.append(fiber)
        
        weights = F.softmax(self.layer_weights[:len(fibers)], dim=0)
        aggregated = sum(w * f for w, f in zip(weights, fibers))
        
        logits = self.predictor(aggregated).squeeze(-1)  # [B, S] LOGITS
        return logits


def compute_repetition_labels_fast(input_ids: torch.Tensor, window: int = 32) -> torch.Tensor:
    B, S = input_ids.shape
    device = input_ids.device
    labels = torch.zeros(B, S, device=device)
    
    for offset in range(1, min(window + 1, S)):
        if offset < S:
            matches = (input_ids[:, offset:] == input_ids[:, :-offset]).float()
            labels[:, offset:] = torch.maximum(labels[:, offset:], matches)
    
    return labels


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("CF-HoT RISK PREDICTOR v2 (CLASS-WEIGHTED)")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    device = next(model.parameters()).device
    print(f"Device: {device}")
    
    print("Adding LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))
    model.print_trainable_parameters()
    
    print("Adding Risk Predictor...")
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    risk_predictor = RiskPredictor(d_model, n_layers, config).to(device).float()
    print(f"Risk Predictor params: {sum(p.numel() for p in risk_predictor.parameters()):,}")
    
    print("Loading data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex['text'] for ex in ds if len(ex['text']) > 50]
    random.shuffle(texts)
    print(f"Loaded {len(texts)} samples")
    
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': config.lr_lora},
        {'params': risk_predictor.parameters(), 'lr': config.lr_predictor}
    ], weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps, eta_min=1e-6
    )
    
    print("\n" + "=" * 70)
    print("TRAINING (with class weighting)")
    print("=" * 70)
    
    model.train()
    risk_predictor.train()
    
    step = 0
    data_idx = 0
    acc_loss, acc_lm, acc_risk_loss = 0, 0, 0
    acc_precision, acc_recall, acc_f1 = 0, 0, 0
    start_time = time.time()
    
    while step < config.max_steps:
        batch = [texts[(data_idx + i) % len(texts)] for i in range(config.batch_size)]
        data_idx += config.batch_size
        
        enc = tokenizer(batch, truncation=True, max_length=config.max_length,
                       padding='max_length', return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True
        )
        
        lm_loss = outputs.loss
        
        # Get logits from risk predictor
        risk_logits = risk_predictor(outputs.hidden_states[1:])
        
        # Compute labels
        rep_labels = compute_repetition_labels_fast(input_ids, config.rep_window)
        
        # CLASS-WEIGHTED LOSS
        # Compute class weights dynamically based on batch
        mask = attention_mask.float()
        n_pos = (rep_labels * mask).sum().clamp(min=1)
        n_neg = ((1 - rep_labels) * mask).sum().clamp(min=1)
        pos_weight = n_neg / n_pos  # Weight positives by ratio of negatives to positives
        pos_weight = pos_weight.clamp(max=10.0)  # Cap at 10x
        
        # BCEWithLogitsLoss with pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(
            risk_logits, rep_labels, 
            pos_weight=torch.ones_like(rep_labels) * pos_weight,
            reduction='none'
        )
        risk_loss = (bce_loss * mask).sum() / mask.sum()
        
        # Total loss
        loss = lm_loss + risk_loss
        
        (loss / config.grad_accum).backward()
        
        # Metrics (apply sigmoid for evaluation)
        with torch.no_grad():
            risk_pred = torch.sigmoid(risk_logits)
            pred_binary = (risk_pred > 0.5).float()
            tp = ((pred_binary == 1) & (rep_labels == 1) & (mask == 1)).sum()
            fp = ((pred_binary == 1) & (rep_labels == 0) & (mask == 1)).sum()
            fn = ((pred_binary == 0) & (rep_labels == 1) & (mask == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        acc_loss += loss.item()
        acc_lm += lm_loss.item()
        acc_risk_loss += risk_loss.item()
        acc_precision += precision.item()
        acc_recall += recall.item()
        acc_f1 += f1.item()
        
        step += 1
        
        if step % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                list(lora_params) + list(risk_predictor.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if step % config.log_every == 0:
            eta = (config.max_steps - step) / (step / (time.time() - start_time)) / 3600
            n = config.log_every
            
            print(
                f"Step {step:5d} | "
                f"Loss: {acc_loss/n:.4f} | "
                f"LM: {acc_lm/n:.4f} | "
                f"Risk: {acc_risk_loss/n:.4f} | "
                f"P: {acc_precision/n:.3f} | "
                f"R: {acc_recall/n:.3f} | "
                f"F1: {acc_f1/n:.3f} | "
                f"ETA: {eta:.1f}h"
            )
            
            acc_loss, acc_lm, acc_risk_loss = 0, 0, 0
            acc_precision, acc_recall, acc_f1 = 0, 0, 0
        
        if step % config.save_every == 0:
            ckpt = os.path.join(config.output_dir, f"ckpt_{step}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            torch.save({
                'risk_predictor': risk_predictor.state_dict(),
                'step': step
            }, os.path.join(ckpt, "risk_predictor.pt"))
            print(f">>> Saved: {ckpt}")
        
        if step % config.eval_every == 0:
            model.eval()
            risk_predictor.eval()
            
            print("\n--- Evaluation ---")
            
            prompt = "The will to power, as described by Nietzsche, is"
            inp = tokenizer(prompt, return_tensors='pt')
            input_ids = inp['input_ids'].to(device)
            
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=60,
                    do_sample=True, temperature=0.8, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
                
                gen_outputs = model(out, output_hidden_states=True)
                gen_logits = risk_predictor(gen_outputs.hidden_states[1:])
                gen_risk = torch.sigmoid(gen_logits)
                
                risk_vals = gen_risk[0].cpu().numpy()
                
            print(f"  Generated: {generated_text[:200]}...")
            print(f"  Risk (first 10): {[f'{r:.2f}' for r in risk_vals[:10]]}")
            print(f"  Risk (last 10): {[f'{r:.2f}' for r in risk_vals[-10:]]}")
            print(f"  Mean: {risk_vals.mean():.3f}, Max: {risk_vals.max():.3f}, Min: {risk_vals.min():.3f}")
            
            # Check correlation
            gen_ids = out[0].cpu().numpy()
            actual_reps = []
            for t in range(1, len(gen_ids)):
                start = max(0, t - config.rep_window)
                is_rep = gen_ids[t] in gen_ids[start:t]
                actual_reps.append(1 if is_rep else 0)
            
            # Correlation between risk and actual repeats
            if len(actual_reps) > 1:
                risk_at_reps = [risk_vals[i+1] for i, r in enumerate(actual_reps) if r == 1 and i+1 < len(risk_vals)]
                risk_at_nonreps = [risk_vals[i+1] for i, r in enumerate(actual_reps) if r == 0 and i+1 < len(risk_vals)]
                
                if risk_at_reps and risk_at_nonreps:
                    print(f"  Avg risk at REPEATS: {sum(risk_at_reps)/len(risk_at_reps):.3f}")
                    print(f"  Avg risk at NON-REPS: {sum(risk_at_nonreps)/len(risk_at_nonreps):.3f}")
            
            print("--- End Eval ---\n")
            
            model.train()
            risk_predictor.train()
    
    final = os.path.join(config.output_dir, "final")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    torch.save({
        'risk_predictor': risk_predictor.state_dict(),
        'step': step
    }, os.path.join(final, "risk_predictor.pt"))
    
    print(f"\nDONE! Saved to {final}")


if __name__ == "__main__":
    main()
