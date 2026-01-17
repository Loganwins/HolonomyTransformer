#!/usr/bin/env python3
"""
CF-HoT v3 - CORRECT CAUSAL ACCUMULATION
========================================
Fixed: EMA accumulates across POSITIONS (causal), not layers.
h_t = α·h_{t-1} + (1-α)·Δh_t  (within each layer, across positions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os, time, random, math
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class CFHoTConfig:
    model_path: str = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
    output_dir: str = "./results/cfhot_v3"
    d_fiber: int = 16
    d_control: int = 64
    ema_momentum: float = 0.9  # Back to 0.9 - half-life ~6.6 tokens
    gate_scale: float = 1.0
    max_steps: int = 5000
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 256
    lr_lora: float = 2e-5
    lr_cfhot: float = 1e-4
    weight_decay: float = 0.01
    lambda_hol: float = 0.001
    log_every: int = 10
    save_every: int = 200
    eval_every: int = 100


class CFAdapter(nn.Module):
    """
    CORRECT implementation: causal EMA within each layer.
    """
    def __init__(self, d_model: int, config: CFHoTConfig):
        super().__init__()
        self.config = config
        self.d_model = d_model
        
        # Fiber projection
        self.fiber_proj = nn.Linear(d_model, config.d_fiber, bias=False)
        
        # Holonomy predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model + config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1),
            nn.Softplus()
        )
        
        # Gate parameters
        self.lambda_gate = nn.Parameter(torch.tensor(1.0))
        
        # Small init
        nn.init.normal_(self.fiber_proj.weight, std=0.02)
        nn.init.zeros_(self.predictor[-2].bias)
        nn.init.normal_(self.predictor[-2].weight, std=0.01)
        
        # For tracking accumulated field across generation steps
        self.cached_field = None
    
    def causal_ema(self, delta_h: torch.Tensor, prev_final: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Causal EMA across sequence positions.
        h_t = α·h_{t-1} + (1-α)·Δh_t
        
        For efficiency, implemented as cumulative sum with decay.
        """
        B, S = delta_h.shape
        alpha = self.config.ema_momentum
        device = delta_h.device
        dtype = delta_h.dtype
        
        # Create decay weights: [α^(S-1), α^(S-2), ..., α, 1]
        positions = torch.arange(S, device=device, dtype=dtype)
        
        # For each position t, h_t = (1-α) * Σ_{i=0}^{t} α^{t-i} * Δh_i
        # Plus contribution from previous context if any
        
        # Simple iterative version (can vectorize later if needed)
        h = torch.zeros_like(delta_h)
        
        if prev_final is not None:
            # Continue from previous field value
            h_prev = prev_final
        else:
            h_prev = torch.zeros(B, device=device, dtype=dtype)
        
        for t in range(S):
            h[:, t] = alpha * h_prev + (1 - alpha) * delta_h[:, t]
            h_prev = h[:, t]
        
        return h
    
    def forward(
        self, 
        hidden: torch.Tensor,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: gate, field, risk
        """
        B, S, D = hidden.shape
        orig_dtype = hidden.dtype
        h = hidden.float()
        
        # Fiber projection
        fiber = self.fiber_proj(h)
        
        # Risk prediction (Δh)
        combined = torch.cat([h, fiber], dim=-1)
        delta_h = self.predictor(combined).squeeze(-1)  # [B, S]
        
        # Causal EMA accumulation
        if use_cache and self.cached_field is not None:
            # Continue from cached field
            field = self.causal_ema(delta_h, self.cached_field[:, -1])
        else:
            field = self.causal_ema(delta_h)
        
        # Update cache with final field value for next generation step
        if use_cache:
            if self.cached_field is None:
                self.cached_field = field
            else:
                self.cached_field = torch.cat([self.cached_field, field], dim=1)
        
        # Gate: σ(-λ·h)
        gate = torch.sigmoid(-self.lambda_gate * field)
        
        return gate.to(orig_dtype), field.to(orig_dtype), delta_h.mean()
    
    def reset_cache(self):
        self.cached_field = None


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
        
        self.gate_history = []
        self.total_risk = 0.0
        self.use_cache = False  # Set True during generation
        
        self._patch_attention()
        
        param_count = sum(p.numel() for p in self.adapters.parameters())
        print(f"[CF-HoT v3] Params: {param_count:,}")
        print(f"[CF-HoT v3] Causal EMA with α={config.ema_momentum}")
    
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
            # Compute gate with causal EMA
            gate, field, risk = wrapper.adapters[layer_idx](
                hidden_states,
                use_cache=wrapper.use_cache
            )
            
            wrapper.total_risk = wrapper.total_risk + risk
            wrapper.gate_history.append(gate.mean().item())
            
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
            # CF-HoT INTERVENTION
            # Get the full gate for all KV positions
            # =================================================================
            
            if wrapper.use_cache and wrapper.adapters[layer_idx].cached_field is not None:
                # During generation: use cached field for all positions
                cached = wrapper.adapters[layer_idx].cached_field
                full_gate = torch.sigmoid(-wrapper.adapters[layer_idx].lambda_gate * cached)
                
                # Ensure it matches kv_len
                if full_gate.shape[1] < kv_len:
                    # Pad (shouldn't happen if cache is managed right)
                    pad = torch.ones(bsz, kv_len - full_gate.shape[1], 
                                     device=full_gate.device, dtype=full_gate.dtype)
                    full_gate = torch.cat([pad, full_gate], dim=1)
                elif full_gate.shape[1] > kv_len:
                    full_gate = full_gate[:, -kv_len:]
            else:
                # During training: gate is for current positions
                if gate.shape[1] < kv_len:
                    pad = torch.ones(bsz, kv_len - gate.shape[1],
                                     device=gate.device, dtype=gate.dtype)
                    full_gate = torch.cat([pad, gate], dim=1)
                else:
                    full_gate = gate[:, :kv_len]
            
            # Apply log-space gating
            eps = 1e-8
            log_gate = torch.log(full_gate + eps)
            log_gate = log_gate.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, kv_len]
            
            scores = scores + log_gate
            # =================================================================
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            output = torch.matmul(attn_weights, v)
            
            output = output.transpose(1, 2).reshape(bsz, q_len, -1)
            output = attn.o_proj(output)
            
            return output, present
        
        attn.forward = patched_forward
    
    def reset_state(self):
        self.gate_history = []
        self.total_risk = 0.0
        for adapter in self.adapters:
            adapter.reset_cache()
    
    def set_generation_mode(self, enabled: bool):
        self.use_cache = enabled
        if not enabled:
            for adapter in self.adapters:
                adapter.reset_cache()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
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
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def main():
    config = CFHoTConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("CF-HoT v3 - CORRECT CAUSAL EMA")
    print("=" * 70)
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
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
    
    # LoRA
    print("Adding LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))
    model.print_trainable_parameters()
    
    # CF-HoT
    print("Adding CF-HoT v3...")
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
    cfhot.set_generation_mode(False)
    
    step = 0
    data_idx = 0
    acc_loss, acc_lm, acc_risk = 0, 0, 0
    start_time = time.time()
    
    while step < config.max_steps:
        batch = [texts[(data_idx + i) % len(texts)] for i in range(config.batch_size)]
        data_idx += config.batch_size
        
        enc = tokenizer(batch, truncation=True, max_length=config.max_length,
                        padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        
        outputs, risk, gates = cfhot(input_ids=ids, attention_mask=mask, labels=ids)
        lm_loss = outputs.loss
        
        loss = lm_loss + config.lambda_hol * risk
        (loss / config.grad_accum).backward()
        
        acc_loss += loss.item()
        acc_lm += lm_loss.item()
        acc_risk += risk.item() if isinstance(risk, torch.Tensor) else risk
        
        step += 1
        
        if step % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(list(lora_params) + list(cfhot.adapters.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if step % config.log_every == 0:
            eta = (config.max_steps - step) / (step / (time.time() - start_time)) / 3600
            mean_gate = sum(gates) / len(gates) if gates else 0.5
            print(f"Step {step:5d} | Loss: {acc_loss/config.log_every:.4f} | "
                  f"LM: {acc_lm/config.log_every:.4f} | Risk: {acc_risk/config.log_every:.1f} | "
                  f"Gate: {mean_gate:.3f} | ETA: {eta:.1f}h")
            acc_loss, acc_lm, acc_risk = 0, 0, 0
        
        if step % config.save_every == 0:
            ckpt = os.path.join(config.output_dir, f"ckpt_{step}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            torch.save({'cfhot': cfhot.adapters.state_dict(), 'step': step}, 
                       os.path.join(ckpt, "cfhot.pt"))
            print(f">>> Saved: {ckpt}")
        
        if step % config.eval_every == 0:
            model.eval()
            cfhot.adapters.eval()
            cfhot.set_generation_mode(True)
            
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
                print(f"  {tokenizer.decode(out[0], skip_special_tokens=True)[:200]}...")
            
            if cfhot.gate_history:
                print(f"  Gates: {[f'{g:.3f}' for g in cfhot.gate_history[-10:]]}")
            print("--- End Eval ---\n")
            
            cfhot.set_generation_mode(False)
            model.train()
            cfhot.adapters.train()
    
    # Final save
    final = os.path.join(config.output_dir, "final")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    torch.save({'cfhot': cfhot.adapters.state_dict()}, os.path.join(final, "cfhot.pt"))
    print(f"\nDONE! Saved to {final}")


if __name__ == "__main__":
    main()
