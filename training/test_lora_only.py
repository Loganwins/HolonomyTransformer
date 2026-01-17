import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = "/mnt/nvme2/ubermesnchetien4/models/merged-final-v5"
CKPT = "./results/true_cfhot_v2/checkpoint_600"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map='auto')
model = PeftModel.from_pretrained(model, CKPT)

prompts = [
    "The will to power, as described by Nietzsche, is",
    "In the beginning, there was",
]

for p in prompts:
    inp = tok(p, return_tensors='pt').input_ids.to(model.device)
    out = model.generate(inp, max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
    print(f"\n{tok.decode(out[0], skip_special_tokens=True)}\n{'='*60}")
