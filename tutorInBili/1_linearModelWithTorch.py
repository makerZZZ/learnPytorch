import torch
import torch.nn as nn

x = torch.tensor([1.], [2.], [3.])
y = torch.tensor([2.], [4.], [6.])

w = torch.tensor([1.])

def forward(x):
    y_pred = x * w
    