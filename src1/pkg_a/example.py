import torch
import torch.nn as nn
import torch.optim as optim

from pkg_a import helter_utils

torch.manual_seed(42)

# Distances in miles for recent bike deliveries
distances = torch.tensor([[1.0],[2.0],[3.0][4.0]], dtype=torch.float32)

# Coresponding delivery times in minuts
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

# Create a model with one input (distance) and one output (time)
model = nn.sequential(nn.linear(1, 1))