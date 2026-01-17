#!/usr/bin/env python3
"""
CF-HoT Phase 2: Decode-Time Guided Generation
==============================================
Uses the trained risk predictor to guide generation away from repetition.

The base model is UNCHANGED. We only intervene in the sampling step:
- High risk → suppress tokens that would be repeats
- Low risk → minimal intervention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    model_path: str = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
    checkpoint_path: str = "./results/cfhot_risk_v2/final"
    d_fiber: int = 16
    d_control: int = 64
    rep_window: int = 32


# =============================================================================
# RISK PREDICTOR (same architecture as training)
# =============================================================================

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
        
        self.predictor = nn.Sequential(
            nn.Linear(config.d_fiber, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, config.d_control),
            nn.GELU(),
            nn.Linear(config.d_control, 1)
        )
    
    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        fibers = []
        for i, (proj, h) in enumerate(zip(self.fiber_projs, hidden_states)):
            if i < len(hidden_states):
                fiber = proj(h.float())
                fibers.append(fiber)
        
        weights = F.softmax(self.layer_weights[:len(fibers)], dim=0)
        aggregated = sum(w * f for w, f in zip(weights, fibers))
        
        logits = self.predictor(aggregated).squeeze(-1)
        return torch.sigmoid(logits)


# =============================================================================
# GUIDED GENERATION
# =============================================================================

class CFHoTGenerator:
    """
    Wraps a model + risk predictor for guided generation.
    """
    def __init__(
        self, 
        model: nn.Module, 
        risk_predictor: RiskPredictor,
        tokenizer,
        device: torch.device
    ):
        self.model = model
        self.risk_predictor = risk_predictor
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        penalty_scale: float = 5.0,  # How much to penalize predicted repeats
        window: int = 32,            # Look-back window for repetition
        verbose: bool = False
    ) -> str:
        """
        Generate text with CF-HoT risk-guided sampling.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            penalty_scale: Multiplier for risk-based penalty
            window: How far back to look for potential repeats
            verbose: Print debug info
        
        Returns:
            Generated text
        """
        self.model.eval()
        self.risk_predictor.eval()
        
        # Encode prompt
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        
        generated_tokens = []
        risk_history = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.model(input_ids, output_hidden_states=True)
                logits = outputs.logits[:, -1, :].float()  # [1, vocab_size]
                
                # Get risk prediction for current position
                risk = self.risk_predictor(outputs.hidden_states[1:])[:, -1]  # [1]
                risk_val = risk.item()
                risk_history.append(risk_val)
                
                # Apply temperature
                logits = logits / temperature
                
                # Get recent tokens (potential repeats)
                recent_tokens = input_ids[0, -window:].tolist()
                
                # Apply risk-based penalty to recent tokens
                # Higher risk → stronger penalty on tokens that would be repeats
                if risk_val > 0.1:  # Only intervene if risk is non-trivial
                    penalty = risk_val * penalty_scale
                    for token_id in set(recent_tokens):
                        logits[0, token_id] -= penalty
                
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if verbose and step % 10 == 0:
                    token_str = self.tokenizer.decode([next_token.item()])
                    print(f"  Step {step}: risk={risk_val:.3f}, token='{token_str}'")
        
        # Decode
        full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if verbose:
            print(f"\nRisk stats: mean={sum(risk_history)/len(risk_history):.3f}, "
                  f"max={max(risk_history):.3f}, min={min(risk_history):.3f}")
        
        return full_text
    
    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        Generate WITHOUT CF-HoT intervention (for comparison).
        """
        self.model.eval()
        
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def count_repetitions(text: str, tokenizer, window: int = 32) -> dict:
    """Count repetition statistics in generated text."""
    tokens = tokenizer.encode(text)
    
    total = 0
    repeats = 0
    
    for t in range(1, len(tokens)):
        start = max(0, t - window)
        if tokens[t] in tokens[start:t]:
            repeats += 1
        total += 1
    
    return {
        'total_tokens': total,
        'repeat_tokens': repeats,
        'repeat_rate': repeats / total if total > 0 else 0,
        'unique_tokens': len(set(tokens)),
        'unique_rate': len(set(tokens)) / len(tokens) if tokens else 0
    }


def compute_distinct_ngrams(text: str, tokenizer, n: int = 2) -> float:
    """Compute distinct-n metric."""
    tokens = tokenizer.encode(text)
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    
    print("=" * 70)
    print("CF-HoT PHASE 2: GUIDED GENERATION")
    print("=" * 70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    # Load LoRA
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, config.checkpoint_path)
    
    device = next(model.parameters()).device
    print(f"Device: {device}")
    
    # Load risk predictor
    print("Loading risk predictor...")
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    
    risk_predictor = RiskPredictor(d_model, n_layers, config).to(device).float()
    
    ckpt = torch.load(
        os.path.join(config.checkpoint_path, "risk_predictor.pt"),
        weights_only=False
    )
    risk_predictor.load_state_dict(ckpt['risk_predictor'])
    print(f"Loaded risk predictor from step {ckpt.get('step', 'unknown')}")
    
    # Create generator
    generator = CFHoTGenerator(model, risk_predictor, tokenizer, device)
    
    # Test prompts
    test_prompts = [
        "The will to power, as described by Nietzsche, is",
        "In the beginning, there was",
        "The fundamental nature of consciousness is",
        "To achieve true happiness, one must",
        "The relationship between mind and body",
    ]
    
    print("\n" + "=" * 70)
    print("COMPARISON: BASELINE vs CF-HoT GUIDED")
    print("=" * 70)
    
    results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print("=" * 70)
        
        # Baseline generation
        print("\n--- BASELINE (no intervention) ---")
        baseline_text = generator.generate_baseline(
            prompt, max_new_tokens=100, temperature=0.8, top_p=0.9
        )
        print(baseline_text[:300] + "..." if len(baseline_text) > 300 else baseline_text)
        baseline_stats = count_repetitions(baseline_text, tokenizer)
        print(f"\nStats: {baseline_stats['repeat_rate']:.1%} repetition, "
              f"{baseline_stats['unique_rate']:.1%} unique")
        
        # CF-HoT guided generation
        print("\n--- CF-HoT GUIDED (penalty_scale=5.0) ---")
        guided_text = generator.generate(
            prompt, max_new_tokens=100, temperature=0.8, top_p=0.9,
            penalty_scale=5.0, verbose=False
        )
        print(guided_text[:300] + "..." if len(guided_text) > 300 else guided_text)
        guided_stats = count_repetitions(guided_text, tokenizer)
        print(f"\nStats: {guided_stats['repeat_rate']:.1%} repetition, "
              f"{guided_stats['unique_rate']:.1%} unique")
        
        # Distinct-2 metric
        baseline_d2 = compute_distinct_ngrams(baseline_text, tokenizer, 2)
        guided_d2 = compute_distinct_ngrams(guided_text, tokenizer, 2)
        print(f"\nDistinct-2: Baseline={baseline_d2:.3f}, Guided={guided_d2:.3f}")
        
        results.append({
            'prompt': prompt,
            'baseline_rep_rate': baseline_stats['repeat_rate'],
            'guided_rep_rate': guided_stats['repeat_rate'],
            'baseline_d2': baseline_d2,
            'guided_d2': guided_d2,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_baseline_rep = sum(r['baseline_rep_rate'] for r in results) / len(results)
    avg_guided_rep = sum(r['guided_rep_rate'] for r in results) / len(results)
    avg_baseline_d2 = sum(r['baseline_d2'] for r in results) / len(results)
    avg_guided_d2 = sum(r['guided_d2'] for r in results) / len(results)
    
    print(f"\nAverage Repetition Rate:")
    print(f"  Baseline: {avg_baseline_rep:.1%}")
    print(f"  CF-HoT:   {avg_guided_rep:.1%}")
    print(f"  Change:   {(avg_guided_rep - avg_baseline_rep) / avg_baseline_rep * 100:+.1f}%")
    
    print(f"\nAverage Distinct-2:")
    print(f"  Baseline: {avg_baseline_d2:.3f}")
    print(f"  CF-HoT:   {avg_guided_d2:.3f}")
    print(f"  Change:   {(avg_guided_d2 - avg_baseline_d2) / avg_baseline_d2 * 100:+.1f}%")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("Type a prompt and see baseline vs CF-HoT guided generation.")
    print("Commands: 'quit' to exit, 'penalty X' to set penalty scale")
    print("=" * 70)
    
    penalty_scale = 5.0
    
    while True:
        try:
            user_input = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower().startswith('penalty '):
            try:
                penalty_scale = float(user_input.split()[1])
                print(f"Penalty scale set to {penalty_scale}")
                continue
            except:
                print("Usage: penalty <number>")
                continue
        
        print("\n--- BASELINE ---")
        baseline = generator.generate_baseline(user_input, max_new_tokens=80)
        print(baseline)
        
        print(f"\n--- CF-HoT (penalty={penalty_scale}) ---")
        guided = generator.generate(user_input, max_new_tokens=80, penalty_scale=penalty_scale)
        print(guided)
        
        # Quick stats
        b_stats = count_repetitions(baseline, tokenizer)
        g_stats = count_repetitions(guided, tokenizer)
        print(f"\nRepetition: Baseline={b_stats['repeat_rate']:.1%}, "
              f"Guided={g_stats['repeat_rate']:.1%}")


if __name__ == "__main__":
    main()
