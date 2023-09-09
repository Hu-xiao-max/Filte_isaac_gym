import torch
# input = torch.randn(10, 3, 4)
# mat2 = torch.randn(10, 4, 5)
# res = torch.bmm(input, mat2)
# res.size()
# torch.Size([10, 3, 5])

input=torch.randn(10, 3)
mat2=torch.randn(3,1)
res=input[:,:2]
print(res)