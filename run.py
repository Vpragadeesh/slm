import torch
from slm import GPT

model = GPT(config)  # re-create the model with same config
device =  "cuda"
best_model_params_path = "./best_model_params.pt"
model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device))) # load best model states

sentence = "Once upon a time there was a pumpkin."
context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
y = model.generate(context, 200)
print(f"\033[1;32m[SLM] Output:\033[0m {enc.decode(y.squeeze().tolist())}")

print("\033[1;32m[SLM] Generating story 2...\033[0m")
sentence = "A little girl went to the woods"
context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
y = model.generate(context, 200)
print(f"\033[1;32m[SLM] Output:\033[0m {enc.decode(y.squeeze().tolist())}")
