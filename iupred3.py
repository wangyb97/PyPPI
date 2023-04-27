import os

data_num = '2'

data_path = './split_dataset/seq_'+data_num
data_list = os.listdir(data_path)
print(data_list)

for i in data_list:
    a = "python './iupred3/iupred3.py' -a " + '\' '+data_path + '/' + i+'\' ' + 'long'
    os.system(a)