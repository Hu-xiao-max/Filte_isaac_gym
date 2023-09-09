from itertools import count
from pickle import TRUE
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_trans
torch.set_default_dtype(torch.float32)

def rm2eula(Rm):
    r = R_trans.from_matrix(Rm)
    eulas=r.as_euler('xyz', degrees=False)
    eulas=np.flip(eulas,axis=1)
    return eulas

def qua2rm(Rqs):
    rs = R_trans.from_quat(Rqs)
    Rms = rs.as_matrix()
    return Rms

def rm2qua(Rms):
    rs = R_trans.from_matrix(Rms)
    qua = rs.as_quat()
    return qua

def viewpoint_params_to_matrix(towards, angle):
    '''
    **Input:**

    - towards: numpy array towards vector with shape (3,).

    - angle: float of in-plane rotation.

    **Output:**

    - numpy array of the rotation matrix with shape (3, 3).
    '''
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix.astype(np.float32)



def get_model_grasps(datapath,point_file_path):
    ''' 
    Load grasp labels from .npz files.
    '''
    label = np.load(datapath)
    arr=np.load(point_file_path)
    print(arr.shape[0])
    tmp=np.zeros((arr.shape[0],3))
    for i in range(arr.shape[0]):
        tmp[i,0]=arr[i,0]
        tmp[i,1]=arr[i,1]
        tmp[i,2]=arr[i,2]
    print(tmp.shape)
    points = tmp
    offsets = label['offsets']
    offsets=offsets[:arr.shape[0],]
    scores = label['scores']
    scores=scores[:arr.shape[0],]
    collision = label['collision']
    collision=collision[:arr.shape[0],]
    return points, offsets, scores, collision

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    ''' 
    View sampling on a sphere using Febonacci lattices.

    **Input:**

    - N: int, number of viewpoints.

    - phi: float, constant angle to sample views, usually 0.618.

    - center: numpy array of (3,), sphere center.

    - R: float, sphere radius.

    **Output:**

    - numpy array of (N, 3), coordinates of viewpoints.
    '''
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views
    


dataset_root='/home/tencent_go/Music/codes/multi_feature_get/utils/npztest'
obj_idx = 5
point_file_path='/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/003_cracker_box.npy'
sampled_points, offsets, scores, collision = get_model_grasps('%s/%03d_labels.npz'%(dataset_root, obj_idx),point_file_path)
#上例的路径为/home/mo/Downloads/grasp_label/005_labels.npz
flag = False
th=0.4#设定的分值的阈值
max_width=0.08
# np.random.shuffle(point_inds)
num_views = 300
views = generate_views(num_views)
point_inds = np.arange(sampled_points.shape[0])#看sampled_points有几行，就是有多少个点
Rs=[]
target_points=[]
depths=[]
grasp_indice=[]
new_label={}


# ======================= get  grasp poses =======================
for point_ind in point_inds:

    target_point = sampled_points[point_ind]
    offset = offsets[point_ind]
    score = scores[point_ind]
    view_inds = np.arange(300)#(1,300)
    for v in view_inds:

        view = views[v]
        angle_inds = np.arange(12)#(1,12)
        for a in angle_inds:

            depth_inds = np.arange(4)#(1,4)
            for d in depth_inds:

                angle, depth, width = offset[v, a, d]
                #从offset最后一个维度（1，3）分别拿出角度、深度、宽度
                if score[v, a, d] > th or score[v, a, d] < 0 or width > max_width:
                    continue
                #如果（1.分值大于阈值2.分数小于03.宽度大于最大宽度）就结束本次循环
                #后面就是添加差不多的抓取配置来用issacgym测试

                R = viewpoint_params_to_matrix(-view, angle)
                axis_y_90=np.array([[0,0,1],[0,1,0],[-1,0,0]])
                R = R.dot(axis_y_90)
                #旋转夹爪姿态与issac配置时一致
                t = target_point
                Rs.append(R)
                target_points.append(t)
                depths.append(depth)
                grasp_indice.append([point_ind,v,a,d])
                #输出可抓取配置的数组

print('+++++++++++++++++++++++++++++++++')
grasp_ind=[]
grasp_indice=np.asarray(grasp_indice)
for i in range(grasp_indice.shape[0]):
    grasp_ind.append(grasp_indice[i,0])
grasp_ind=np.asarray(grasp_ind)
print(grasp_ind.shape)#(10765,)  input：(150, 3)