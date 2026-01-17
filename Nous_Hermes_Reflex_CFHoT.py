#!/usr/bin/env python3
"""
Nous-Hermes-ReflexAgent-8B-v1-CF-HoT
------------------------------------
The benchmark adapter for the CF-HoT alignment project.
Integrating Reflexive Reasoning with Holonomic Gating.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from training.phase_b_8b_adapters import CFHoTLlamaHooked, CFAdapterConfig

# === PROJECT IDENTITY ===
MODEL_NAME = "Nous-Hermes-ReflexAgent-8B-v1-CF-HoT"
BASE_PATH = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'
ADAPTER_PATH = 'results/ORIGINAL_WORKING.pt'

class ReflexConfig:
    # Frame the persona as a high-level research agent
    system_prompt = (
        "You are the Nous-Hermes-ReflexAgent-8B-v1-CF-HoT. "
        "Your architecture utilizes a Control Field Holonomy Transformer "
        "to maintain geometric consistency across long-horizon reasoning."
    )
    max_tokens = 600
    temp = 0.85

def load_reflex_system():
    print(f">>> BOOTING {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, quantization_config=bnb, device_map='auto')
    
    # Inject CF-HoT Benchmarking Gating
    cf_config = CFAdapterConfig()
    cf_config.d_model = base.config.hidden_size
    cf_config.n_layers = base.config.num_hidden_layers
    
    hooked_model = CFHoTLlamaHooked(base, cf_config)
    ckpt = torch.load(ADAPTER_PATH, weights_only=False)
    hooked_model.cf_adapters.load_state_dict(ckpt['adapter_state_dict'])
    hooked_model.cf_adapters.to('cuda').half()
    
    return tok, hooked_model

def run_benchmark_prompt(tok, model, prompt):
    print(f"\n[PROMPT]: {prompt}")
    # Initialize Holonomy Field
    model.control_field = None 
    
    inputs = tok(f"{ReflexConfig.system_prompt}\nUser: {prompt}\nReflexive Analysis:", return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        out = model.base_model.generate(
            **inputs,
            max_new_tokens=ReflexConfig.max_tokens,
            do_sample=True,
            temperature=ReflexConfig.temp,
            repetition_penalty=1.0 # Let the CF-HoT do the work, no manual penalty
        )
    
    print(f"\n[{MODEL_NAME} OUTPUT]:")
    print(tok.decode(out[0], skip_special_tokens=True).split("Reflexive Analysis:")[-1].strip())

if __name__ == "__main__":
    tokenizer, model = load_reflex_system()
    # The ultimate test dummy prompt for the video
    test_prompt = "Perform a recursive audit of the concept of 'Consistency'. How does a system know it is consistent without repeating itself?"
    run_benchmark_prompt(tokenizer, model, test_prompt)
