import  torch
import torch.nn as nn
import torch.autograd as autograd

input = autograd.Variable(torch.randn(3,5))
m = nn.Conv1d(5, 6, 1)
output = m(input)
d = m.weight.data

print ''