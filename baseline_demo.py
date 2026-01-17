#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            BASELINE DEMO (NO CF-HOT)                                         ║
║            Showing repetitive text degeneration                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script demonstrates the BASELINE model behavior WITHOUT CF-HoT adapters.
Use this to show the repetitive loops that CF-HoT is designed to prevent.

For video: Run this FIRST, then run cfhot_live_demo.py to show the improvement.
"""

import os
import torch
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'

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

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"""
{Colors.RED}╔══════════════════════════════════════════════════════════════════════════════╗
║{Colors.BOLD}                                                                              {Colors.END}{Colors.RED}║
║{Colors.BOLD}            BASELINE MODEL (NO CF-HOT ADAPTERS)                               {Colors.END}{Colors.RED}║
║{Colors.BOLD}            Demonstrating Repetitive Text Degeneration                        {Colors.END}{Colors.RED}║
║{Colors.BOLD}                                                                              {Colors.END}{Colors.RED}║
╠══════════════════════════════════════════════════════════════════════════════╣
║{Colors.END}  {Colors.RED}CF-HoT Status:{Colors.END} {Colors.RED}●{Colors.END} DISABLED   {Colors.RED}Holonomy Gating:{Colors.END} {Colors.RED}●{Colors.END} OFF                   {Colors.RED}║
║{Colors.END}  {Colors.YELLOW}Base Model:{Colors.END}   Hermes-3-Llama-3.1-8B (4-bit quantized)               {Colors.RED}║
║{Colors.END}  {Colors.YELLOW}Adapters:{Colors.END}     NONE - Raw baseline behavior                          {Colors.RED}║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
""")

    # Load model
    print(f"{Colors.YELLOW}>>> Loading baseline model (no adapters)...{Colors.END}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16,
    )
    
    print(f"{Colors.GREEN}>>> Baseline model loaded (NO CF-HoT){Colors.END}")
    print(f"\n{Colors.DIM}Commands: 'exit' to quit | 'demo' for standard test prompt{Colors.END}\n")
    
    # Chat loop
    while True:
        try:
            user_input = input(f"{Colors.RED}Baseline >{Colors.END} ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                break
            
            if user_input.lower() == 'demo':
                user_input = "The will to power, as described by Nietzsche, is"
                print(f"{Colors.DIM}Using standard test prompt: {user_input}{Colors.END}")
            
            # Generate with NO repetition penalty to show raw behavior
            inputs = tokenizer(user_input, return_tensors='pt').to('cuda')
            
            print(f"\n{Colors.DIM}Generating (no CF-HoT, no repetition penalty)...{Colors.END}")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.0,  # NO penalty - raw behavior
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = text[len(user_input):].strip()
            
            # Display with repetition highlighting
            print(f"\n{Colors.RED}┌─ BASELINE OUTPUT (watch for repetition) {'─' * 35}┐{Colors.END}")
            
            words = response.split()
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
            
            for line in lines[:10]:
                print(f"{Colors.RED}│{Colors.END} {line:<74} {Colors.RED}│{Colors.END}")
            if len(lines) > 10:
                print(f"{Colors.RED}│{Colors.END} {'[...]':<74} {Colors.RED}│{Colors.END}")
            
            print(f"{Colors.RED}└{'─' * 76}┘{Colors.END}")
            
            # Count and highlight repetitions
            words_lower = response.lower().split()
            phrases = {}
            for i in range(len(words_lower) - 3):
                phrase = ' '.join(words_lower[i:i + 4])
                phrases[phrase] = phrases.get(phrase, 0) + 1
            
            repeated_phrases = [(p, c) for p, c in phrases.items() if c > 1]
            repeated_phrases.sort(key=lambda x: -x[1])
            
            if repeated_phrases:
                print(f"\n{Colors.RED}⚠ DETECTED REPETITIONS:{Colors.END}")
                for phrase, count in repeated_phrases[:5]:
                    print(f"  {Colors.YELLOW}'{phrase}'{Colors.END} × {count}")
            
            rep_score = len(repeated_phrases)
            print(f"\n{Colors.RED}Repetition score: {rep_score}{Colors.END}")
            print(f"{Colors.DIM}(This is what CF-HoT is designed to prevent){Colors.END}\n")
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}>>> Interrupted{Colors.END}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}\n")

if __name__ == "__main__":
    main()
