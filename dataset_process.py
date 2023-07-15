import math
import numpy as np
from model import createmodel3
from utils import evaluate, callbacks, data_split, label_sum, label_one_hot
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import (EarlyStopping, LearningRateScheduler)
import os

kmer = 25
os.environ["CUDA_VISIBLE_DEVICES"]="1"

vec_xl_1 = np.load('/home/houzilong/Main/EDLMPPI/xl_model/vec_843.npy').reshape(-1,1,1024)
vec_xl_2 = np.load('./xl_model/vec_448.npy').reshape(-1,1,1024)
vec_xl_3 = np.load('./xl_model/vec_72.npy').reshape(-1,1,1024)
vec_xl_4 = np.load('./xl_model/vec_164.npy').reshape(-1,1,1024)
vec_xl_5 = np.load('./xl_model/vec_186.npy').reshape(-1,1,1024)
vec_bio_1 = np.load('./bio_features/wind_' + str(kmer) +'/843.npy')
vec_bio_2 = np.load('./bio_features/wind_' + str(kmer) +'/448.npy')
vec_bio_3 = np.load('./bio_features/wind_' + str(kmer) +'/72.npy')
vec_bio_4 = np.load('./bio_features/wind_' + str(kmer) +'/164.npy')
vec_bio_5 = np.load('./bio_features/wind_' + str(kmer) +'/186.npy')
label1 = np.array(label_one_hot(np.load('./xl_model/label_843.npy')))
label2 = np.array(label_one_hot(np.load('./xl_model/label_448.npy')))
label3 = np.array(label_one_hot(np.load('./xl_model/label_72.npy')))
label4 = np.array(label_one_hot(np.load('./xl_model/label_164.npy')))
label5 = np.array(label_one_hot(np.load('./xl_model/label_186.npy')))

vec_xl_1 = np.concatenate([vec_xl_1,vec_xl_5], axis=0)
vec_bio_1 = np.concatenate([vec_bio_1,vec_bio_5],axis=0)
label1 = np.concatenate([label1,label5],axis=0)

print(vec_xl_1.shape)

def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 7.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate


callbacks = [EarlyStopping(monitor='val_loss', patience=6),LearningRateScheduler(step_decay)]

positive_list_xl, positive_list_bio, sub_list_xl, sub_list_bio = data_split(vec_xl_1, vec_bio_1, 6)
batchSize = 1024
maxEpochs = 30
pred_result1 = [[0, 0]] * len(label2)
pred_result2 = [[0, 0]] * len(label3)
pred_result3 = [[0, 0]] * len(label4)
pred_result4 = [[0, 0]] * len(label5)
print(len(sub_list_xl))
for i in range(len(sub_list_xl)):
    train_xl = np.array(np.concatenate((sub_list_xl[i], positive_list_xl), axis=0))
    train_bio = np.array(np.concatenate((sub_list_bio[i], positive_list_bio), axis=0))
    label = np.concatenate((np.zeros(len(sub_list_xl[i]), dtype=int), np.ones(len(positive_list_xl), dtype=int)))
    label = [str(i) for i in label]
    train_label = np.array(label_one_hot(label))

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)
    for train_index, val_index in split.split(train_xl, train_label):
        train_X_xl = train_xl[train_index]
        train_X_bio = train_bio[train_index]
        val_X_xl = train_xl[val_index]
        val_X_bio = train_bio[val_index]
        train_y = train_label[train_index]
        val_y = train_label[val_index]

        model = createmodel3(kmer)

        model.fit([train_X_xl, train_X_bio], train_y,
                  epochs=maxEpochs,
                  batch_size=batchSize,
                  callbacks=callbacks,
                  verbose=1,
                  validation_data=([val_X_xl, val_X_bio], val_y),
                  shuffle=True)
        model.save('./predicted_model/' + str(i) + '.h5')

    pred_result1 = label_sum(pred_result1, model.predict([vec_xl_2, vec_bio_2]))
    pred_result2 = label_sum(pred_result2, model.predict([vec_xl_3, vec_bio_3]))
    pred_result3 = label_sum(pred_result3, model.predict([vec_xl_4, vec_bio_4]))
    pred_result4 = label_sum(pred_result4, model.predict([vec_xl_5, vec_bio_5]))
    print("****************" + str((i + 1)) + "*****************")

print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ('Index', 'Sens', 'Spec', 'Pre', 'ACC', 'F1', 'MCC', 'AUROC', 'AUPRC'))
evaluate(label2, pred_result1)
evaluate(label3, pred_result2)
evaluate(label4, pred_result3)
evaluate(label5, pred_result4)
