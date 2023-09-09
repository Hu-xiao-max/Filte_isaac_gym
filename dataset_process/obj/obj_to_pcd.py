import os 
import numpy as np
import sys
import shutil
sys.path.append('/home/tencent_go/Music/codes/multi_feature_get/dataset_process/obj/')

filedir = os.path.dirname(sys.argv[0])
os.chdir(filedir)
wdir = os.getcwd()
print('当前工作目录为：{}\n'.format(wdir))


for parent,dirs,files in os.walk(wdir):

	print(dirs)
	if 'data' in parent:

		os.chdir('data')
	#os.mkdir('pcd_files')
	for file in files:

		prefix = file.split('.')[0]
		#f = open('0_pred.obj','rb')
		new_name = prefix + '.' + 'pcd'
		print(new_name)
		f = open(new_name,'w')
		file='/home/tencent_go/Music/codes/multi_feature_get/dataset_process/obj/obj_to_pcd.py'



		num_lines = sum(1 for line in open(file))
		print(num_lines)  
		#pcd的数据格式 https://blog.csdn.net/BaiYu_King/article/details/81782789  

		f.write('# .PCD v0.7 - Point Cloud Data file format \nVERSION 0.7 \nFIELDS x y z rgba \nSIZE 4 4 4 4 \nTYPE F F F U \nCOUNT 1 1 1 1 \n' )
		f.write('WIDTH {} \nHEIGHT 1 \nVIEWPOINT 0 0 0 1 0 0 0 \n'.format(num_lines))
		f.write('POINTS {} \nDATA ascii\n'.format(num_lines))
		f1 = open(file,'rb')
		#f2 = open('new_book.pcd','w')

		lines = f1.readlines()
		a = []
		for line in lines:
			line1 = line.decode()

			new_line = line1.split(' ')[1] + ' ' + line1.split(' ')[2] + ' ' + line1.split(' ')[3] + ' ' + line1.split(' ')[4] + ' ' + line1.split(' ')[5] + ' ' + line1.split(' ')[6] 
			#new_line = line.split(' ')[1]

			#f2.write(new_line)
			f.write(new_line)
		f.close()

		shutil.move(new_name,'pcd_files')
