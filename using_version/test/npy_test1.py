import numpy as np


def npy_load(path):
    arr=np.load(path)
    #print(arr.shape[0])
    tmp=np.zeros((arr.shape[0],3))
    for i in range(arr.shape[0]):
        tmp[i,0]=arr[i,0]
        tmp[i,1]=arr[i,1]
        tmp[i,2]=arr[i,2]
    grasp=arr[0]
    #return tmp,arr
    return grasp
 
def main():
    path='/home/tencent_go/Music/codes/multi_feature_get/dataset_process/npy/003_cracker_box.npy'
    grasp=npy_load(path)
    print(grasp[7])


if __name__=='__main__':
    main()
    print('---------')
    x=np.array([1,0,0])
    y=np.array([0,1,0])
    z=np.cross(x,y)
    r=np.c_[x,y,z]
    print(r.dot(r)[:,0])
    



