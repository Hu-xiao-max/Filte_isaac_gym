import torch
a=torch.tensor([[1,1],[1,1]])
if torch.all(a<1):
    print('1')
    
print(a.view(-1,2))
print(torch.norm(a.float(),dim=-1))