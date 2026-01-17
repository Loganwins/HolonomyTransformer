#!/usr/bin/env python3
"""Resume CF-HoT from checkpoint 1500"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
import os, time, random, math

MODEL_PATH = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
CHECKPOINT = "./results/cfhot_v3/ckpt_1400"  # Last saved checkpoint
OUTPUT_DIR = "./results/cfhot_v3"
START_STEP = 1400
MAX_STEPS = 5000

# Import the classes from cfhot_v3
exec(open('training/cfhot_v3.py').read().split('def main():')[0])

def main():
    config = CFHoTConfig()
    config.max_steps = MAX_STEPS
    
    print("=" * 70)
    print(f"RESUMING CF-HoT FROM STEP {START_STEP}")
    print("=" * 70)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model with LoRA from checkpoint
    print("Loading model + LoRA from checkpoint...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, CHECKPOINT)
    model = model.merge_and_unload()  # Merge then re-add trainable LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))
    
    device = next(model.parameters()).device
    print(f"Device: {device}")
    
    # CF-HoT
    print("Loading CF-HoT adapters from checkpoint...")
    cfhot = CFHoTWrapper(model, config)
    cfhot.adapters = cfhot.adapters.to(device).float()
    
    # Load CF-HoT weights
    ckpt = torch.load(os.path.join(CHECKPOINT, "cfhot.pt"), weights_only=False)
    cfhot.adapters.load_state_dict(ckpt['cfhot'])
    print(f"Loaded CF-HoT from step {ckpt.get('step', 'unknown')}")
    
    # Data
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
        optimizer, T_max=MAX_STEPS - START_STEP, eta_min=1e-6
    )
    
    # Training
    print(f"\nResuming from step {START_STEP} to {MAX_STEPS}")
    
    model.train()
    cfhot.adapters.train()
    cfhot.set_generation_mode(False)
    
    step = START_STEP
    data_idx = 0
    acc_loss, acc_lm, acc_risk = 0, 0, 0
    start_time = time.time()
    
    while step < MAX_STEPS:
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
            eta = (MAX_STEPS - step) / ((step - START_STEP) / (time.time() - start_time)) / 3600 if step > START_STEP else 0
            mean_gate = sum(gates) / len(gates) if gates else 0.5
            print(f"Step {step:5d} | Loss: {acc_loss/config.log_every:.4f} | "
                  f"LM: {acc_lm/config.log_every:.4f} | Risk: {acc_risk/config.log_every:.1f} | "
                  f"Gate: {mean_gate:.3f} | ETA: {eta:.1f}h")
            acc_loss, acc_lm, acc_risk = 0, 0, 0
        
        if step % config.save_every == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"ckpt_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            torch.save({'cfhot': cfhot.adapters.state_dict(), 'step': step}, 
                       os.path.join(ckpt_dir, "cfhot.pt"))
            print(f">>> Saved: {ckpt_dir}")
        
        if step % config.eval_every == 0:
            model.eval()
            cfhot.adapters.eval()
            cfhot.set_generation_mode(True)
            
            print("\n--- Evaluation ---")
            for prompt in ["The will to power, as described by Nietzsche, is",
                          "In the beginning, there was"]:
                cfhot.reset_state()
                with torch.no_grad():
                    inp = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                    out = model.generate(inp, max_new_tokens=80, do_sample=True, 
                                        temperature=0.8, top_p=0.9,
                                        pad_token_id=tokenizer.eos_token_id)
                print(f"  {tokenizer.decode(out[0], skip_special_tokens=True)[:200]}...")
            
            if cfhot.gate_history:
                print(f"  Gates: {[f'{g:.3f}' for g in cfhot.gate_history[-10:]]}")
            print("--- End Eval ---\n")
            
            cfhot.set_generation_mode(False)
            model.train()
            cfhot.adapters.train()
    
    # Final save
    final = os.path.join(OUTPUT_DIR, "final_5000")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    torch.save({'cfhot': cfhot.adapters.state_dict(), 'step': step}, os.path.join(final, "cfhot.pt"))
    print(f"\nDONE! Saved to {final}")


if __name__ == "__main__":
    main()
