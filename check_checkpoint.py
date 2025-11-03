import torch

checkpoint = torch.load("models/saved/custom_best.pt", map_location='cpu')
print("Checkpoint keys:", checkpoint.keys())
