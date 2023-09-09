import numpy as np

datapath='/home/tencent_go/Music/codes/multi_feature_get/utils/npztest/005_labels.npz'
label = np.load(datapath)
points = label['points']#(1895, 3)
offsets = label['offsets']#(1895, 300, 12, 4, 3)
scores = label['scores']#(1895, 300, 12, 4)
collision = label['collision']#(1895, 300, 12, 4)
