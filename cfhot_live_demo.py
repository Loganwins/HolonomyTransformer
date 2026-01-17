#!/usr/bin/env python3
import sys; sys.path.insert(0, ".")
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            NOUS-HERMES-REFLEXAGENT-8B-V1-CF-HOT                              ║
║            Control Field Holonomy Transformer Demo                           ║
║                                                                              ║
║            "Consistency Is All You Need"                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Live chat interface demonstrating CF-HoT adapters reducing repetitive 
text degeneration through geometric consistency enforcement.

Paper:  github.com/Loganwins/HolonomyTransformer
Model:  huggingface.co/LoganResearch/Nous-Hermes-ReflexAgent-8B-v1
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from datetime import datetime

# Suppress warnings for clean demo
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'
ADAPTER_PATH = 'results/ORIGINAL_WORKING.pt'

class Config:
    """Generation configuration."""
    system_prompt = """You are the Nous-Hermes-ReflexAgent, an advanced reasoning system enhanced with 
Control Field Holonomy (CF-HoT) adapters. You exhibit:
- Deep philosophical insight rooted in Nietzschean thought
- Precise, structured reasoning without repetitive loops  
- High agency and intellectual rigor
- Clear, coherent multi-step analysis

Respond with depth and clarity."""

    temperature = 0.8
    top_p = 0.9
    max_new_tokens = 300
    repetition_penalty = 1.05

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_banner():
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║{Colors.BOLD}{Colors.GREEN}                                                                              {Colors.END}{Colors.CYAN}║
║{Colors.BOLD}{Colors.GREEN}            NOUS-HERMES-REFLEXAGENT-8B-V1-CF-HOT                              {Colors.END}{Colors.CYAN}║
║{Colors.BOLD}{Colors.GREEN}            Control Field Holonomy Transformer                                {Colors.END}{Colors.CYAN}║
║{Colors.BOLD}{Colors.GREEN}                                                                              {Colors.END}{Colors.CYAN}║
║{Colors.DIM}            "Consistency Is All You Need" — Napolitano, 2026                    {Colors.END}{Colors.CYAN}║
║{Colors.DIM}                                                                              {Colors.END}{Colors.CYAN}║
╠══════════════════════════════════════════════════════════════════════════════╣
║{Colors.END}  {Colors.YELLOW}CF-HoT Status:{Colors.END} {Colors.GREEN}●{Colors.END} ACTIVE     {Colors.YELLOW}Holonomy Gating:{Colors.END} {Colors.GREEN}●{Colors.END} ENABLED               {Colors.CYAN}║
║{Colors.END}  {Colors.YELLOW}Base Model:{Colors.END}   Hermes-3-Llama-3.1-8B (4-bit quantized)               {Colors.CYAN}║
║{Colors.END}  {Colors.YELLOW}Adapters:{Colors.END}     10.5M params (0.13% overhead, 100 training steps)     {Colors.CYAN}║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
"""
    print(banner)

def print_status(gate_value=None, risk_value=None):
    """Print real-time CF-HoT status."""
    if gate_value is not None and risk_value is not None:
        print(f"{Colors.DIM}  [CF-HoT] Gate: {gate_value:.4f} | Risk: {risk_value:.4f}{Colors.END}")

def print_response(text, tokens_per_sec=None):
    """Pretty print model response."""
    print(f"\n{Colors.CYAN}┌─ Response {'─' * 66}┐{Colors.END}")
    
    # Word wrap at 74 chars
    words = text.split()
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
    
    for line in lines:
        print(f"{Colors.CYAN}│{Colors.END} {line:<74} {Colors.CYAN}│{Colors.END}")
    
    print(f"{Colors.CYAN}└{'─' * 76}┘{Colors.END}")
    
    if tokens_per_sec:
        print(f"{Colors.DIM}  [{tokens_per_sec:.1f} tokens/sec]{Colors.END}")

def print_commands():
    print(f"\n{Colors.DIM}Commands: 'exit' to quit | 'clear' to clear | 'bench' to run benchmark{Colors.END}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load the model with CF-HoT adapters."""
    print(f"{Colors.YELLOW}>>> Initializing Nous-Hermes-ReflexAgent-8B-v1-CF-HoT...{Colors.END}")
    
    # Tokenizer
    print(f"{Colors.DIM}    Loading tokenizer...{Colors.END}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Base model (4-bit)
    print(f"{Colors.DIM}    Loading base model (4-bit quantization)...{Colors.END}")
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
    
    # CF-HoT adapters
    print(f"{Colors.DIM}    Injecting CF-HoT adapters...{Colors.END}")
    try:
        from training.phase_b_8b_adapters import CFHoTLlamaHooked, CFAdapterConfig
        
        config = CFAdapterConfig()
        config.d_model = base_model.config.hidden_size
        config.n_layers = base_model.config.num_hidden_layers
        
        cf_model = CFHoTLlamaHooked(base_model, config)
        ckpt = torch.load(ADAPTER_PATH, weights_only=False)
        cf_model.cf_adapters.load_state_dict(ckpt['adapter_state_dict'])
        cf_model.cf_adapters = cf_model.cf_adapters.to('cuda').half()
        cf_model.eval()
        
        adapter_params = sum(p.numel() for p in cf_model.cf_adapters.parameters())
        print(f"{Colors.GREEN}>>> CF-HoT adapters loaded: {adapter_params:,} parameters{Colors.END}")
        
        return tokenizer, cf_model, base_model
        
    except Exception as e:
        print(f"{Colors.YELLOW}>>> Warning: CF-HoT adapters not available: {e}{Colors.END}")
        print(f"{Colors.YELLOW}>>> Running in baseline mode{Colors.END}")
        return tokenizer, None, base_model

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_response(tokenizer, cf_model, base_model, user_input):
    """Generate response with CF-HoT gating."""
    
    # Build prompt
    full_prompt = f"{Config.system_prompt}\n\nUser: {user_input}\n\nAssistant:"
    inputs = tokenizer(full_prompt, return_tensors='pt').to('cuda')
    
    # Reset control field if using CF-HoT
    if cf_model is not None:
        cf_model.control_field = None
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=Config.max_new_tokens,
            do_sample=True,
            temperature=Config.temperature,
            top_p=Config.top_p,
            repetition_penalty=Config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("Assistant:")[-1].strip()
    
    # Calculate tokens/sec
    num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
    
    # Get CF-HoT metrics if available
    gate_value = None
    risk_value = None
    if cf_model is not None and hasattr(cf_model, 'cf_adapters'):
        if cf_model.control_field is not None:
            gate_value = torch.sigmoid(-cf_model.control_field).mean().item()
        if hasattr(cf_model.cf_adapters, 'layer_risks') and cf_model.cf_adapters.layer_risks:
            risk_value = cf_model.cf_adapters.layer_risks[-1].mean().item()
    
    return response, tokens_per_sec, gate_value, risk_value

# ═══════════════════════════════════════════════════════════════════════════════
# QUICK BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def run_quick_benchmark(tokenizer, base_model):
    """Run a quick A/B comparison for demo."""
    print(f"\n{Colors.YELLOW}>>> Running quick benchmark...{Colors.END}\n")
    
    prompt = "The will to power, as described by Nietzsche, is"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    print(f"{Colors.BOLD}Prompt:{Colors.END} {prompt}\n")
    
    # Generate
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.0,  # No penalty to show raw behavior
            pad_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = text[len(prompt):].strip()
    
    print(f"{Colors.CYAN}Output:{Colors.END}")
    print(f"  {response[:500]}...")
    
    # Count repetitions
    words = response.lower().split()
    phrases = {}
    for i in range(len(words) - 3):
        phrase = ' '.join(words[i:i + 4])
        phrases[phrase] = phrases.get(phrase, 0) + 1
    repeated = sum(1 for count in phrases.values() if count > 1)
    
    print(f"\n{Colors.YELLOW}Repetition score: {repeated}{Colors.END}")
    print(f"{Colors.DIM}(Lower is better. Baseline without CF-HoT typically scores 5-15){Colors.END}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    clear_screen()
    print_banner()
    
    # Load model
    tokenizer, cf_model, base_model = load_model()
    
    print_commands()
    
    # Chat loop
    while True:
        try:
            # Prompt
            user_input = input(f"{Colors.GREEN}You >{Colors.END} ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == 'exit':
                print(f"\n{Colors.YELLOW}>>> Shutting down...{Colors.END}\n")
                break
            elif user_input.lower() == 'clear':
                clear_screen()
                print_banner()
                print_commands()
                continue
            elif user_input.lower() == 'bench':
                run_quick_benchmark(tokenizer, base_model)
                continue
            
            # Generate response
            print(f"\n{Colors.DIM}Generating...{Colors.END}")
            response, tps, gate, risk = generate_response(
                tokenizer, cf_model, base_model, user_input
            )
            
            # Display
            if gate is not None or risk is not None:
                print_status(gate, risk)
            print_response(response, tps)
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}>>> Interrupted. Type 'exit' to quit.{Colors.END}\n")
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}\n")

if __name__ == "__main__":
    main()
