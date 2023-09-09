import torch
a=torch.tensor([[1,2],[3,4]])
#print((a>2).nonzero())
b=(a>2).nonzero()
print(b.shape[0])
c=0.7*8
print(int(c))