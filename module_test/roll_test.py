import numpy as np
arr=np.load('/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/011_banana.npy')
tmp=np.zeros((arr.shape[0],3))
for i in range(arr.shape[0]):
    tmp[i,0]=arr[i,0]
    tmp[i,1]=arr[i,1]
    tmp[i,2]=arr[i,2]
res=tmp[1,0:3]
R=np.array([[1,0,0],[0,0,-1],[0,1,0 ]])
res=np.array([1,1,1])
print(R.shape)
c=res.dot(R)
print(c)
