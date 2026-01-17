#!/usr/bin/env python3
"""
CF-HoT Interactive Chat
=======================
Persistent chatbot using validated 100-step CF-HoT adapters.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from training.phase_b_8b_adapters import CFHoTLlamaHooked, CFAdapterConfig

def main():
    print("=" * 60)
    print("CF-HoT INTERACTIVE CHAT")
    print("=" * 60)
    
    # Load model
    print("\n[Loading model...]")
    model_path = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16
    )
    
    # Load CF-HoT adapters
    print("[Loading CF-HoT adapters...]")
    config = CFAdapterConfig()
    config.d_model = base_model.config.hidden_size
    config.n_layers = base_model.config.num_hidden_layers
    
    cf_model = CFHoTLlamaHooked(base_model, config)
    ckpt = torch.load('results/ORIGINAL_WORKING.pt', weights_only=False)
    cf_model.cf_adapters.load_state_dict(ckpt['adapter_state_dict'])
    cf_model.cf_adapters = cf_model.cf_adapters.to('cuda').half()
    
    print("\n[Ready! Type 'quit' to exit, 'clear' to reset context]\n")
    print("-" * 60)
    
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        if user_input.lower() == 'clear':
            history = []
            cf_model.control_field = None
            print("[Context cleared]")
            continue
        
        # Build prompt
        cf_model.control_field = None
        
        # Simple completion mode
        prompt = user_input
        
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = base_model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response if it's there
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        print(f"\nCF-HoT: {response}")

if __name__ == '__main__':
    main()
