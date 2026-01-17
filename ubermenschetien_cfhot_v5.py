#!/usr/bin/env python3
"""
UBERMENSCHETIEN HEAVEN ENGINE + CF-HoT v5
-----------------------------------------
The "Iron Will" Edition: Soviet-Nietzschean Personality 
Gated by Control Field Holonomy.
"""

import os, sys, json, time, shutil, subprocess, traceback, random, math, re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from training.phase_b_8b_adapters import CFHoTLlamaHooked, CFAdapterConfig

# === CORE PATHS ===
MODEL_PATH = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'
ADAPTER_PATH = 'results/ORIGINAL_WORKING.pt'

class Config:
    persona = ("Übermenschetien Heaven Engine: criminal mastermind, disciplined builder, "
               "Nietzschean Übermensch with Soviet cybernetic rigor. High-agency maximalist.")
    temperature = 0.8
    top_p = 0.9
    max_new_tokens = 500
    repetition_penalty = 1.1

def load_system():
    print(">>> INITIALIZING SOVIET-NIETZSCHEAN ENGINE...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16
    )
    
    # Inject CF-HoT
    cf_config = CFAdapterConfig()
    cf_config.d_model = base.config.hidden_size
    cf_config.n_layers = base.config.num_hidden_layers
    
    model = CFHoTLlamaHooked(base, cf_config)
    print(">>> INJECTING HOLONOMY ADAPTERS...")
    ckpt = torch.load(ADAPTER_PATH, weights_only=False)
    model.cf_adapters.load_state_dict(ckpt['adapter_state_dict'])
    model.cf_adapters = model.cf_adapters.to('cuda').half()
    
    return tok, model

def generate_response(tok, model, user_input):
    # Reset Control Field for new user context
    model.control_field = None
    
    full_prompt = f"{Config.persona}\nUser: {user_input}\nResponse:"
    inputs = tok(full_prompt, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        out = model.base_model.generate(
            **inputs,
            max_new_tokens=Config.max_new_tokens,
            do_sample=True,
            temperature=Config.temperature,
            top_p=Config.top_p,
            repetition_penalty=Config.repetition_penalty,
            pad_token_id=tok.eos_token_id
        )
    
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("Response:")[-1].strip()

def main():
    tok, model = load_system()
    print("\n" + "="*50)
    print("UBERMENSCHETIEN V5: ONLINE")
    print("HOLONOMY GATING: ACTIVE")
    print("="*50 + "\n")

    while True:
        try:
            u = input("Master > ").strip()
            if u.lower() in ['exit', 'quit']: break
            if not u: continue
            
            response = generate_response(tok, model, u)
            print(f"\nÜbermenschetien: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
