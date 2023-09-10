import numpy as np

# 读取文件内容
with open("/home/tencent_go/paper/Filte_isaac_gym/run_txt/test.txt", "r") as file:
    lines = file.readlines()

# 定义一个函数来提取[]中的数据并转换为numpy.array
def extract_bracket_data(line, index):
    bracket_data = line.split("[")[index].split("]")[0]
    return np.fromstring(bracket_data, sep=' ')

# 遍历每一行
for line in lines:
    arrays = []
    # 提取四个[]中的数据
    for i in range(1, 5):
        array_data = extract_bracket_data(line, i)
        arrays.append(array_data)

    # 输出转换后的numpy.array
    print(arrays[0])
    print(arrays[1:4])
 