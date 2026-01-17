#!/usr/bin/env python3
"""CF-HoT CORRECT: Log-space attention modulation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import math, os

@dataclass
class CFAdapterConfig:
    d_model: int = 4096
    n_layers: int = 32
    d_fiber: int = 16
    d_control: int = 64
    momentum: float = 0.9
    lambda_init: float = 0.1
    lambda_hol_loss: float = 1e-4

class CFAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fiber_proj = nn.Linear(config.d_model, config.d_fiber, bias=False)
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model + config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1),
            nn.Softplus()
        )
        nn.init.zeros_(self.predictor[-2].bias)
        nn.init.normal_(self.predictor[-2].weight, std=0.01)
        self.lambda_gate = nn.Parameter(torch.tensor(config.lambda_init))
    
    def forward(self, hidden, prev_field=None):
        batch, seq_len, _ = hidden.shape
        orig_dtype = hidden.dtype
        h = hidden.float()
        
        fiber = self.fiber_proj(h)
        combined = torch.cat([h, fiber], dim=-1)
        risk = self.predictor(combined).squeeze(-1)
        
        if prev_field is None:
            field = (1 - self.config.momentum) * risk
        else:
            if prev_field.shape[-1] < seq_len:
                prev_field = torch.cat([prev_field, torch.zeros(batch, seq_len - prev_field.shape[-1], device=prev_field.device)], dim=-1)
            elif prev_field.shape[-1] > seq_len:
                prev_field = prev_field[..., :seq_len]
            field = self.config.momentum * prev_field.float() + (1 - self.config.momentum) * risk
        
        gate = torch.sigmoid(-self.lambda_gate * field)
        return gate.to(orig_dtype), field.to(orig_dtype), risk.to(orig_dtype)

class CFHoTLlamaCorrect(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.config = config
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.cf_adapters = nn.ModuleList([CFAdapter(config) for _ in range(config.n_layers)])
        self.control_field = None
        self.total_risk = 0.0
        self.gates = []
        self._patch_attention()
        print(f"[CFHoT-Correct] Adapter params: {sum(p.numel() for p in self.cf_adapters.parameters()):,}")
    
    def _patch_attention(self):
        layers = self.base_model.model.layers
        for idx, layer in enumerate(layers):
            layer.self_attn = GatedAttentionWrapper(layer.self_attn, self, idx)
    
    def _reset_state(self):
        self.control_field = None
        self.total_risk = 0.0
        self.gates = []
    
    def get_gate_for_layer(self, hidden_states, layer_idx):
        gate, field, risk = self.cf_adapters[layer_idx](hidden_states, self.control_field)
        self.control_field = field
        self.total_risk = self.total_risk + risk.sum()
        self.gates.append(gate.mean().item())
        return gate
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        self._reset_state()
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        result = {'logits': outputs.logits, 'total_risk': self.total_risk, 
                  'mean_gate': sum(self.gates)/len(self.gates) if self.gates else 1.0}
        if labels is not None:
            lm_loss = outputs.loss
            norm = input_ids.shape[0] * input_ids.shape[1] * self.config.n_layers
            risk_reg = self.config.lambda_hol_loss * self.total_risk / norm
            result.update({'lm_loss': lm_loss, 'risk_reg': risk_reg, 'loss': lm_loss + risk_reg})
        return result
    
    def get_adapter_params(self):
        return self.cf_adapters.parameters()

class GatedAttentionWrapper(nn.Module):
    def __init__(self, original_attn, cfhot, layer_idx):
        super().__init__()
        self.attn = original_attn
        self.cfhot = cfhot
        self.layer_idx = layer_idx
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        gate = self.cfhot.get_gate_for_layer(hidden_states, self.layer_idx)
        bsz, q_len, _ = hidden_states.shape
        cfg = self.cfhot.base_model.config
        num_heads, num_kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, self.attn.head_dim
        
        q = self.attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k = self.attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = self.attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._rotary(q, k, cos, sin)
        
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None
        
        if num_kv_heads != num_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # CF-HoT: log-space attention modulation
        kv_len = k.shape[2]
        if gate.shape[-1] < kv_len:
            gate = torch.cat([torch.ones(bsz, kv_len - gate.shape[-1], device=gate.device, dtype=gate.dtype), gate], dim=-1)
        elif gate.shape[-1] > kv_len:
            gate = gate[..., -kv_len:]
        scores = scores + torch.log(gate + 1e-8).unsqueeze(1).unsqueeze(2)
        
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(bsz, q_len, -1)
        return self.attn.o_proj(out), present
    
    def _rotary(self, q, k, cos, sin):
        def rotate(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
        return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)

def train():
    tokenizer = AutoTokenizer.from_pretrained('/mnt/nvme2/ubermesnchetien4/models/merged-final-v5')
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading model...")
    base = AutoModelForCausalLM.from_pretrained('/mnt/nvme2/ubermesnchetien4/models/merged-final-v5',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map='auto', torch_dtype=torch.float16)
    
    config = CFAdapterConfig(d_model=base.config.hidden_size, n_layers=base.config.num_hidden_layers)
    model = CFHoTLlamaCorrect(base, config)
    model.cf_adapters = model.cf_adapters.to('cuda').float()
    
    print("Loading data...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    ds = ds.filter(lambda x: len(x['text']) > 100)
    ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, max_length=512, padding='max_length'), batched=True, remove_columns=['text'])
    ds.set_format('torch')
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    
    opt = torch.optim.AdamW(model.get_adapter_params(), lr=1e-4)
    print("="*60 + "\nTRAINING CF-HoT CORRECT (Log-space)\n" + "="*60)
    
    for step, batch in enumerate(loader):
        if step >= 100: break
        opt.zero_grad()
        out = model(input_ids=batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda(), labels=batch['input_ids'].cuda())
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.cf_adapters.parameters(), 1.0)
        opt.step()
        if (step+1) % 10 == 0:
            print(f"Step {step+1:3d} | Loss: {out['lm_loss'].item():.4f} | Risk: {out['total_risk'].item():.1f} | Gate: {out['mean_gate']:.3f}")
    
    os.makedirs('results/phase_b_correct', exist_ok=True)
    torch.save({'adapter_state_dict': model.cf_adapters.state_dict(), 'config': config}, 'results/phase_b_correct/cf_adapter_final.pt')
    print("="*60 + "\nDONE! Testing...\n" + "="*60)
    
    model.cf_adapters.eval()
    model._reset_state()
    with torch.no_grad():
        out = base.generate(tokenizer("The will to power, as described by Nietzsche, is", return_tensors='pt').input_ids.cuda(),
                           max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print(f"\nMean gate: {sum(model.gates)/len(model.gates):.4f}")

if __name__ == "__main__":
    train()
