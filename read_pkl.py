import pickle

# 打开文件并读取内容
with open('blosum_dict.pkl', 'rb') as f:
    data = pickle.load(f)

# 处理数据
print('data:',data)