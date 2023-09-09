
import os
import numpy as np


# filePath = '/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/14kb'
# npy_list=os.listdir(filePath)
# for i in npy_list:
#     name=i.split('.')[0]
#     print(name)

#     arr=np.load('npy_path')
#     np.save("using_version/data/npy_select/"+name+'.npy', arr)

arr=np.load('/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/14kb/003_cracker_box.npy')
print(arr.shape)