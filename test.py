import torch
from model import model, get_device
from tokenizer import stoi, itos
from data_preparation import block_size

device = get_device()
model = model.to(device)

model.load_state_dict(torch.load("last_model.pt", map_location=device))
model.eval()

def generate(model, prompt, max_new_tokens=100):
    model.eval()

    x = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]

        with torch.no_grad():
            logits = model(x_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    return ''.join([itos[i] for i in x[0].tolist()])

prompt = generate(model, "Hello, buddy", 1)
print(prompt)

