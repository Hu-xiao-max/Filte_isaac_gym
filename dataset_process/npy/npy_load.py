import numpy as np
# arr=np.load('/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/011_banana.npy')
# print(arr.shape[0])
# tmp=np.zeros((arr.shape[0],3))
# for i in range(arr.shape[0]):
#     tmp[i,0]=arr[i,0]
#     tmp[i,1]=arr[i,1]
#     tmp[i,2]=arr[i,2]

# print(tmp.shape)
arr=np.load('/home/pxing/code/PointNetGPD/PointNetGPD/data/ycb_grasps/train/002_lions.npy')
print(arr.shape[0])
tmp=np.zeros((arr.shape[0],3))
for i in range(arr.shape[0]):
    tmp[i,0]=arr[i,0]
    tmp[i,1]=arr[i,1]
    tmp[i,2]=arr[i,2]

print(tmp.shape)

