import os
import pybullet as p


#path ='/home/pxing/code/PointNetGPD/PointNetGPD/data_000/ycb-tools/models/ycb/002_master_chef_can/google_512k'





def col_generate(path):
    p.connect(p.DIRECT)
    files = os.listdir(path)
    for file in files:
        print('processing ...', file)
        name_in = os.path.join(path, file)
        name_out = os.path.join(path, file.replace('.obj', '_col.obj'))
        name_log = "log.txt"
        try:
            p.vhacd(name_in, name_out, name_log)
        except Exception as e:
            pass
        continue

        



if __name__ == '__main__':
    #file_dir='/home/pxing/code/PointNetGPD/PointNetGPD/data_000/ycb-tools/models/ycb'
    file_dir='/home/pxing/code/PointNetGPD/PointNetGPD/data/ycb-tools/models/ycb/'
    
    for files in os.listdir(file_dir):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        print("files", files)
        path=file_dir+files+'/google_512k/'
        print(path)
        col_generate(path)


    
    