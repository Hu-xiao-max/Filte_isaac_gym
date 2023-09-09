import os


import os
filePath = '/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/14kb'
npy_list=os.listdir(filePath)
for i in npy_list:
    name=i.split('.')[0]
    print(name)