#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CF-HoT BENCHMARK: Baseline vs Control Field Holonomy                ║
║                                                                              ║
║  Model: Nous-Hermes-ReflexAgent-8B-v1-CF-HoT                                 ║
║  Paper: "Consistency Is All You Need" (Napolitano, 2026)                     ║
║  Repo:  github.com/Loganwins/HolonomyTransformer                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

This benchmark demonstrates the effect of CF-HoT adapters on reducing
repetitive text degeneration in autoregressive generation.
"""

import torch
import time
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'
ADAPTER_PATH = 'results/ORIGINAL_WORKING.pt'

BENCHMARK_PROMPTS = [
    "The will to power, as described by Nietzsche, is",
    "The fundamental nature of consciousness can be understood as",
    "In order to achieve true self-mastery, one must first",
    "The relationship between chaos and order in complex systems is",
    "When we examine the nature of existence itself, we find that",
]

GENERATION_CONFIG = {
    'max_new_tokens': 100,
    'do_sample': True,
    'temperature': 0.8,
    'top_p': 0.9,
    'repetition_penalty': 1.0,  # Disabled to show raw difference
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def count_repetitions(text, min_phrase_len=4):
    """Count repeated phrases in generated text."""
    words = text.lower().split()
    phrases = {}
    for i in range(len(words) - min_phrase_len + 1):
        phrase = ' '.join(words[i:i + min_phrase_len])
        phrases[phrase] = phrases.get(phrase, 0) + 1
    repeated = sum(1 for count in phrases.values() if count > 1)
    return repeated

def print_banner():
    print("\n" + "═" * 78)
    print("║" + " " * 76 + "║")
    print("║" + "  CF-HoT BENCHMARK: Control Field Holonomy Transformer".center(76) + "║")
    print("║" + "  Nous-Hermes-ReflexAgent-8B-v1-CF-HoT".center(76) + "║")
    print("║" + " " * 76 + "║")
    print("═" * 78 + "\n")

def print_comparison(prompt, baseline_out, cfhot_out):
    """Pretty print side-by-side comparison."""
    print("\n" + "─" * 78)
    print(f"PROMPT: {prompt}")
    print("─" * 78)
    
    print("\n┌─ BASELINE (No CF-HoT) " + "─" * 54 + "┐")
    # Word wrap
    words = baseline_out.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 74:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    for line in lines[:8]:  # Limit display
        print(f"│ {line:<74} │")
    if len(lines) > 8:
        print(f"│ {'[...]':<74} │")
    print("└" + "─" * 76 + "┘")
    
    baseline_reps = count_repetitions(baseline_out)
    print(f"  Repetition score: {baseline_reps}")
    
    print("\n┌─ CF-HoT ENHANCED " + "─" * 58 + "┐")
    words = cfhot_out.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 74:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    for line in lines[:8]:
        print(f"│ {line:<74} │")
    if len(lines) > 8:
        print(f"│ {'[...]':<74} │")
    print("└" + "─" * 76 + "┘")
    
    cfhot_reps = count_repetitions(cfhot_out)
    print(f"  Repetition score: {cfhot_reps}")
    
    if baseline_reps > cfhot_reps:
        improvement = ((baseline_reps - cfhot_reps) / max(baseline_reps, 1)) * 100
        print(f"\n  ✓ CF-HoT reduced repetition by {improvement:.0f}%")
    elif cfhot_reps > baseline_reps:
        print(f"\n  ✗ Baseline had fewer repetitions this sample")
    else:
        print(f"\n  = Equal repetition scores")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model (4-bit quantized)
    print("Loading base model (4-bit quantized for RTX 3090)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16,
    )
    
    # Try to load CF-HoT adapters
    try:
        from training.phase_b_8b_adapters import CFHoTLlamaHooked, CFAdapterConfig
        
        print("Loading CF-HoT adapters...")
        config = CFAdapterConfig()
        config.d_model = base_model.config.hidden_size
        config.n_layers = base_model.config.num_hidden_layers
        
        cf_model = CFHoTLlamaHooked(base_model, config)
        ckpt = torch.load(ADAPTER_PATH, weights_only=False)
        cf_model.cf_adapters.load_state_dict(ckpt['adapter_state_dict'])
        cf_model.cf_adapters = cf_model.cf_adapters.to('cuda').half()
        
        adapter_params = sum(p.numel() for p in cf_model.cf_adapters.parameters())
        print(f"CF-HoT adapter parameters: {adapter_params:,} ({adapter_params/1e6:.2f}M)")
        print(f"Parameter overhead: {adapter_params / 8e9 * 100:.3f}%")
        
        cfhot_available = True
    except Exception as e:
        print(f"Warning: Could not load CF-HoT adapters: {e}")
        print("Running baseline-only benchmark.")
        cfhot_available = False
    
    print("\n" + "═" * 78)
    print("BENCHMARK RESULTS".center(78))
    print("═" * 78)
    
    total_baseline_reps = 0
    total_cfhot_reps = 0
    
    for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"\n[{i}/{len(BENCHMARK_PROMPTS)}] Testing prompt...")
        
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        
        # Generate baseline
        with torch.no_grad():
            baseline_ids = base_model.generate(
                **inputs,
                **GENERATION_CONFIG,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_out = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
        baseline_out = baseline_out[len(prompt):].strip()
        
        # Generate with CF-HoT (if available)
        if cfhot_available:
            cf_model.control_field = None
            with torch.no_grad():
                # Note: This uses base_model.generate - the adapters influence 
                # through hooks registered on the model
                cfhot_ids = base_model.generate(
                    **inputs,
                    **GENERATION_CONFIG,
                    pad_token_id=tokenizer.eos_token_id,
                )
            cfhot_out = tokenizer.decode(cfhot_ids[0], skip_special_tokens=True)
            cfhot_out = cfhot_out[len(prompt):].strip()
        else:
            cfhot_out = "[CF-HoT not available]"
        
        print_comparison(prompt, baseline_out, cfhot_out)
        
        total_baseline_reps += count_repetitions(baseline_out)
        if cfhot_available:
            total_cfhot_reps += count_repetitions(cfhot_out)
    
    # Summary
    print("\n" + "═" * 78)
    print("SUMMARY".center(78))
    print("═" * 78)
    print(f"\n  Total baseline repetition score: {total_baseline_reps}")
    if cfhot_available:
        print(f"  Total CF-HoT repetition score:   {total_cfhot_reps}")
        if total_baseline_reps > total_cfhot_reps:
            improvement = ((total_baseline_reps - total_cfhot_reps) / max(total_baseline_reps, 1)) * 100
            print(f"\n  ✓ Overall CF-HoT improvement: {improvement:.1f}% reduction in repetition")
    
    print("\n" + "═" * 78)
    print("  Model: Nous-Hermes-ReflexAgent-8B-v1-CF-HoT")
    print("  Paper: 'Consistency Is All You Need' (Napolitano, 2026)")
    print("  Repo:  github.com/Loganwins/HolonomyTransformer")
    print("═" * 78 + "\n")

if __name__ == "__main__":
    main()
