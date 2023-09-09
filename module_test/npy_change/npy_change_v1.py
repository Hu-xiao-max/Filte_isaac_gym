import numpy as np


class npy_change():
    def __init__(self,path) -> None:
        self.path=path
        pass

    def npy_load(self):
        path=self.path
        arr=np.load(path)
        #print(arr.shape[0])
        tmp=np.zeros((arr.shape[0],3))
        for i in range(arr.shape[0]):
            tmp[i,0]=arr[i,0]
            tmp[i,1]=arr[i,1]
            tmp[i,2]=arr[i,2]

        #print(tmp.shape)
        return tmp,arr
    
    
    def grasp_single(self,tmp):
        #print(arr[0])
        return arr[0]
        

    def r_matrix_get(self,grasp):
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
        
    
    
    
if __name__=='__main__':
    path='/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/003_cracker_box.npy'
    npy=npy_change(path)
    tmp,arr=npy.npy_load()
    grasp=npy.grasp_single(tmp)
    npy.r_matrix_get(grasp)
    R=npy.r_matrix_get(grasp)
    print(R.shape)
