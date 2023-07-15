import numpy as np
import math

def label_one_hot(label_list):
    label = []
    for i in label_list:
        if i=='0':
            label.append([1,0])
        else:
            label.append([0,1])
    return label

def data_split(vec_xl_1, vec_bio_1, emsemble_num): # 根据提供的标签将数据分为正样本和负样本
  label1 = np.load('label_72.npy')
  label2 = np.load('label_164.npy')
  label = np.concatenate([label1,label2],axis=0)
  negative_list = []
  positive_list = []
  for i in range(len(label)):
    if label[i] == '0':
      negative_list.append(i)
    else:
      positive_list.append(i)
  random.shuffle(negative_list)
  split_num = emsemble_num #round(list(label).count('0')/list(label).count('1'))-3
  sample_num = list(label).count('0')//split_num  # 将 sample_num 计算为 label 中 '0' 标签的数量除以 split_num 的整数部分。
  print(len(positive_list))

  sub_list_xl = []
  sub_list_bio = []
  positive_list_xl = vec_xl_1[positive_list]
  positive_list_bio = vec_bio_1[positive_list]
  for i in range(split_num):
    start = i*sample_num
    end = (i+1)*sample_num
    if i == split_num-1:
      end = len(negative_list)
    sub_list_xl.append(vec_xl_1[negative_list[start:end]])
    sub_list_bio.append(vec_bio_1[negative_list[start:end]])
  return positive_list_xl, positive_list_bio, sub_list_xl, sub_list_bio


def label_sum(pre,now):
    c = []
    for i in range(len(now)):
        c.append(np.sum((pre[i], now[i]), axis=0))
    return c


data_split()