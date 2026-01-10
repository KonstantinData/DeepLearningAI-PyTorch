import torch

x = torch.tensor([
    [12.5, 3.2, 7.0],
    [ 8.0, 1.5, 2.0],
    [20.1, 4.8, 9.0],
    [ 5.4, 0.9, 1.0],
])

v = x[0]          # first sample
print(v)
print(v.shape)   # torch.Size([3])

