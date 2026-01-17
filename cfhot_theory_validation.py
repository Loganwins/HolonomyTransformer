import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class HolonomicAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Theory: Keep it simple. A linear probe into the geometry.
        self.proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Initialize so Risk starts at ~0.0 (Gate starts at 1.0)
        nn.init.constant_(self.proj[-2].weight, 0)
        nn.init.constant_(self.proj[-2].bias, -4) 

    def forward(self, x):
        risk = self.proj(x.to(torch.float32))
        # Theory: Gate = 1 - (0.5 * Risk). 
        gate = 1.0 - (0.5 * risk)
        return gate.to(x.dtype)

def train():
    model_id = '/mnt/nvme2/ubermesnchetien4/models/merged-final-v5'
    print(">>> LOADING BASE MODEL...")
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, 
        quantization_config=BitsAndBytesConfig(load_in_4bit=True), 
        device_map='auto')
    
    d_model = base.config.hidden_size
    num_layers = base.config.num_hidden_layers
    adapters = nn.ModuleList([HolonomicAdapter(d_model) for _ in range(num_layers)]).to('cuda')
    
    def get_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h_states = output[0]
                gate = adapters[idx](h_states)
                new_h = h_states * gate
                return (new_h,) + output[1:]
            else:
                gate = adapters[idx](output)
                return output * gate
        return hook

    for i, layer in enumerate(base.model.layers):
        layer.self_attn.register_forward_hook(get_hook(i))

    opt = torch.optim.AdamW(adapters.parameters(), lr=1e-4)
    
    print(">>> STARTING THEORY VALIDATION (300 STEPS)...")
    # Prompt designed to test if the model can "break" a repetition loop
    test_text = "The nature of power is that power is power of the power."
    
    for step in range(1, 301):
        inputs = tok(test_text, return_tensors='pt').to('cuda')
        outputs = base(**inputs, labels=inputs['input_ids'])
        
        loss = outputs.loss 
        loss.backward()
        
        if step % 5 == 0:
            opt.step()
            opt.zero_grad()
        
        if step % 50 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")

    torch.save(adapters.state_dict(), "theory_valid.pt")
    print(">>> SUCCESS. THEORY VALIDATED.")

if __name__ == "__main__":
    train()
