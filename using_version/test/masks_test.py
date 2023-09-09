import numpy as np 
masks=[True, True, True, False, False, True, True, False, True, False, True, True, True, False, True, True, True, False, True, True, True, False, False, True, True, True, True, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True]

mask = [not tbl for tbl in masks]
print(len(mask))
# x = np.arange(100) 
# x.reshape([100,1])
# print(x.shape[0])
# print(x[mask])
arr=np.load('/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/14kb/003_cracker_box.npy')
arr=arr[:100,:]
print(arr[mask].shape)
arr=arr[mask]
np.save("using_version/data/arr.npy", arr)

print(np.load('/home/tencent_go/Music/codes/multi_feature_get/using_version/data/arr.npy'))





