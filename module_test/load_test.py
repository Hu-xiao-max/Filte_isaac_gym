import numpy as np

def npy_load(point_file_path):
    arr=np.load(point_file_path)
    # tmp=np.zeros((arr.shape[0],3))
    # for i in range(arr.shape[0]):
    #     tmp[i,0]=arr[i,0]
    #     tmp[i,1]=arr[i,1]
    #     tmp[i,2]=arr[i,2]

    #print(tmp.shape)
    return arr

def r_matrix_get(grasp):
    center = grasp[0:3]# 抓取中心点
    axis = grasp[3:6] # binormal抓取点方向向量
    width = grasp[6]
    angle = grasp[7]#夹爪旋转度数

    axis = axis/np.linalg.norm(axis)#转换为单位向量
    binormal = axis
    # cal approach
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]
    #按照列合并数组，表示绕y轴旋转angle度，R1（3，3）
    axis_y = axis
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    #x轴与y轴正交
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)#转换为单位向量
    axis_y = axis_y / np.linalg.norm(axis_y)#转换为单位向量
    axis_z = np.cross(axis_x, axis_y)#叉乘得到z向量
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    #按列合并，R2为[x,y,z]
    approach = R2.dot(R1)[:, 0]
    #R2点乘R1的第一列
    approach = approach / np.linalg.norm(approach)
    #approach转换为单位向量
    minor_normal = np.cross(axis, approach)
    #叉乘axis和approach
    #print(minor_normal)#z
    #print(binormal)#y
    #print(approach)#x
    R=np.c_[approach,binormal,minor_normal]
    '''
    这个接口实际上就是：
    得到夹爪的旋转矩阵
    
    '''
    return R

def scripts():
    point_file_path='/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/003_cracker_box.npy'
    arr=npy_load(point_file_path)#arr(n,12)
    point_inds = np.arange(arr.shape[0])
    max_width=0.08
    Rs=[]
    target_points=[]
    for point_ind in point_inds:
        target_point = arr[point_ind,0:3]
        grasp=arr[point_ind]
        R=r_matrix_get(grasp)
        width = grasp[6]
        if  width > max_width:
            continue
        R = r_matrix_get(grasp)
        axis_y_90=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        R = R.dot(axis_y_90)
        t = target_point
        Rs.append(R)
        target_points.append(t)
        
        return Rs,target_points


        
        
        
        
if __name__=='__main__':
    scripts()