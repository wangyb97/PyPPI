# import tensorflow.python.keras.engine.functional
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
import numpy as np
from tensorflow.keras.layers import Activation,\
    Concatenate, AveragePooling1D, Dropout, GRU, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Convolution1D, Dense, BatchNormalization, MaxPool1D, Flatten, Attention
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import keras.backend as K
from keras.layers import Layer
import tensorflow as tf
import math

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'



clf_names = ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'BaggingClassifier', 'RandomForestClassifier',
             'AdaBoostClassifier', 'GradientBoostingClassifier', 'SVM', 'LinearDiscriminantAnalysis', 'ExtraTreesClassifier']

ML_Classifiers = [
    LogisticRegression(max_iter=10000),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    BaggingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    SVC(probability=True),
    LinearDiscriminantAnalysis(),
    ExtraTreesClassifier()
]
# callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min', restore_best_weights=True)]
def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 7.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate


callbacks = [EarlyStopping(monitor='val_loss', patience=6),LearningRateScheduler(step_decay)]

def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)
    input_at = Activation('relu')(input_bn)
    input_dp = Dropout(0.4)(input_at)
    return input_dp

def ConvolutionBlock(input, f, k):
    A1 = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    A1 = bn_activation_dropout(A1)
    return A1

def MultiScale(input):
    A = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 3)
    D = ConvolutionBlock(input, 64, 1)
    D = ConvolutionBlock(D, 64, 5)
    D = ConvolutionBlock(D, 64, 5)
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('relu')(result)
    return result


def createMLP(hidden_sizes):
    return MLPClassifier(hidden_layer_sizes=hidden_sizes, max_iter=300, learning_rate='adaptive', early_stopping=True,
                         verbose=True, n_iter_no_change=5)


def createCNN(shape1, shape2):
    CNN_input = Input(shape=(shape1, shape2))
    x = Convolution1D(filters=32, kernel_size=3, activation='relu', padding='same')(CNN_input)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(padding='same')(x)
    x = Convolution1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(padding='same')(x)
    x = Flatten()(x)
    x = Dense(units=768, activation='relu')(x)
    output = Dense(units=2, activation='sigmoid')(x)

    model = Model(inputs=CNN_input, outputs=output)

    return model


def createLSTM(shape1, shape2):
    RNN_input = Input(shape=(shape1, shape2))
    x = Bidirectional(LSTM(units=16, return_sequences=False))(RNN_input)
    x = BatchNormalization()(x)
    x = Attention()([x, x])
    x = BatchNormalization()(x)
    x = Dense(units=768, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(units=2, activation='sigmoid')(x)
    output = BatchNormalization()(output)
    model = Model(inputs=RNN_input, outputs=output)

    return model


def createGRU(shape1, shape2):
    RNN_input = Input(shape=(shape1, shape2))
    x = Bidirectional(GRU(units=16, return_sequences=False))(RNN_input)
    x = BatchNormalization()(x)
    x = Attention()([x, x])
    x = BatchNormalization()(x)
    x = Dense(units=768, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(units=2, activation='sigmoid')(x)
    output = BatchNormalization()(output)
    model = Model(inputs=RNN_input, outputs=output)

    return model


def createResNet(shape1, shape2):
    sequence_input = Input(shape=(shape1, shape2), name='sequence_input')
    sequence = Convolution1D(filters=128, kernel_size=3, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('relu')(sequence)
    overallResult = MultiScale(sequence)
    overallResult = AveragePooling1D(pool_size=5, padding='same')(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = Bidirectional(GRU(120, return_sequences=True))(overallResult)
    overallResult = Flatten()(overallResult)
    overallResult = Dense(768, activation='relu')(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)

    return Model(inputs=[sequence_input], outputs=[ss_output])


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x


#define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings,
            'share_weights': self.share_weights,
            'activation': self.activation,
        })
        return config

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def createCapsuleNet(shape1,shape2):
    input1 = Input(shape=(shape1, shape2), dtype="float", name='input1')

    xl_0 = BatchNormalization(axis=-1)(input1)
    xl_1 = Bidirectional(GRU(128, return_sequences=True), name='BiGRU1')(xl_0)
    drop1 = Dropout(0.5)(xl_1)
    xl_2 = Capsule(
        num_capsule=64, dim_capsule=10,
        routings=3, share_weights=True, name='Capsule1')(drop1)
    xl_3 = Flatten()(xl_2)
    # xl_4 = BatchNormalization(axis=-1)(xl_3)

    norm = BatchNormalization(axis=-1, name='cont')(xl_3)
    # drop = Dropout(0.5)(concat)
    outputs = Dense(2, activation='softmax', name='outputs')(norm)

    model = Model(inputs=[input1], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.summary()
    return model


def evaluateMLclassifers(features, labels, file_path='', shuffle=True, folds=5):
    scale = StandardScaler()
    features = scale.fit_transform(features)

    cv = KFold(n_splits=folds, shuffle=shuffle)

    results_file = open(file_path + 'ML_evalution_metrics.csv', 'w')

    results_file.write('clf_name,metrics,metric_name\n')

    print('Starting runnning machine learning classifiers using ' + str(folds) + '-fold cross-validation, please be patient...')
    for clf_name, clf in zip(clf_names, ML_Classifiers):
        ACCs = []
        F1_Scores = []
        AUCs = []
        MCCs = []
        Recalls = []
        print('running ' + clf_name + '...')
        for train_index, test_index in cv.split(labels):
            train_features = features[train_index]
            train_labels = labels[train_index]

            test_features = features[test_index]
            test_labels = labels[test_index]
            clf.fit(train_features, train_labels)

            pre_proba = clf.predict_proba(test_features)[:, 1]
            pre_labels = clf.predict(test_features)


            auc = roc_auc_score(y_true=test_labels, y_score=pre_proba)
            acc = accuracy_score(y_pred=pre_labels, y_true=test_labels)
            f1 = f1_score(y_true=test_labels, y_pred=pre_labels)
            mcc = matthews_corrcoef(y_true=test_labels, y_pred=pre_labels)
            recall = recall_score(y_true=test_labels, y_pred=pre_labels)

            AUCs.append(auc)
            ACCs.append(acc)
            MCCs.append(mcc)
            Recalls.append(recall)
            F1_Scores.append(f1)
        print('finish')

        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(AUCs)) + ',' + 'AUC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(ACCs)) + ',' + 'ACC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(MCCs)) + ',' + 'MCC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(Recalls)) + ',' + 'Recall\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(F1_Scores)) + ',F1_Scores\n')

    print('All classifiers have finished running, the result file are locate in ' + file_path)

    results_file.close()




def evaluateDLclassifers(features, labels, file_path='', shuffle=True, folds=5):
    CNN = createCNN(features.shape[1], features.shape[2])
    LSTM = createLSTM(features.shape[1], features.shape[2])
    GRU=createGRU(features.shape[1], features.shape[2])
    ResNet = createResNet(features.shape[1], features.shape[2])
    MLP = createMLP((5, 5, 5, 2, 2))
    CapsuleNet=createCapsuleNet(features.shape[1],features.shape[2])
    DL_classifiers = {'CNN':CNN, 'LSTM':LSTM, 'GRU':GRU, 'ResNet-1D':ResNet, 'MLP':MLP,'CapsuleNet':CapsuleNet}
    # DL_classifiers = {'CNN': CNN}

    cv = KFold(n_splits=folds, shuffle=shuffle)
    results_file = open(file_path + 'DL_evalution_metrics.csv', 'w')

    results_file.write('clf_name,metrics,metric_name\n')

    labels_2D = to_categorical(labels)
    print('Starting runnning deep learning models using ' + str(folds) + '-fold cross-validation, please be patient...')
    for clf_name in DL_classifiers:
        print('running ' + clf_name + '...')
        ACCs = []
        F1_Scores = []
        AUCs = []
        MCCs = []
        Recalls = []
        for train_index, test_index in cv.split(labels):
            train_features = features[train_index]
            train_labels_2D = labels_2D[train_index]
            train_labels = labels[train_index]

            test_features = features[test_index]
            test_labels_2D = labels_2D[test_index]
            test_labels = labels[test_index]
            if clf_name == 'MLP':
                train_features = train_features.reshape(train_features.shape[0], -1)
                test_features = test_features.reshape(test_features.shape[0], -1)
                DL_classifiers[clf_name].fit(train_features, train_labels)
                pre_proba = DL_classifiers[clf_name].predict_proba(test_features)[:, 1]
                pre_labels = DL_classifiers[clf_name].predict(test_features)
                auc = roc_auc_score(y_true=test_labels, y_score=pre_proba)
                acc = accuracy_score(y_pred=pre_labels, y_true=test_labels)
                f1 = f1_score(y_true=test_labels, y_pred=pre_labels)
                mcc = matthews_corrcoef(y_true=test_labels, y_pred=pre_labels)
                recall = recall_score(y_true=test_labels, y_pred=pre_labels)

            else:
                DL_classifiers[clf_name].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                train_X, val_X, train_Y, val_Y = train_test_split(train_features, train_labels_2D, test_size=0.10, stratify=train_labels_2D)
                DL_classifiers[clf_name].fit(x=train_X, y=train_Y, epochs=30, batch_size=64, verbose=0, shuffle=True, callbacks=callbacks,
                          validation_data=(val_X, val_Y))
                pre_proba = DL_classifiers[clf_name].predict(test_features)
                pre_labels = np.argmax(pre_proba, axis=-1)
                pre_proba = pre_proba[:, 1]
                auc = roc_auc_score(y_true=test_labels, y_score=np.array(pre_proba))
                acc = accuracy_score(y_pred=np.array(pre_labels), y_true=np.array(test_labels))
                f1 = f1_score(y_true=np.array(test_labels), y_pred=np.array(pre_labels))
                mcc = matthews_corrcoef(y_true=np.array(test_labels), y_pred=np.array(pre_labels))
                recall = recall_score(y_true=np.array(test_labels), y_pred=np.array(pre_labels))

            AUCs.append(auc)
            ACCs.append(acc)
            MCCs.append(mcc)
            Recalls.append(recall)
            F1_Scores.append(f1)
        print('finish')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(AUCs)) + ',' + 'AUC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(ACCs)) + ',' + 'ACC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(MCCs)) + ',' + 'MCC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(Recalls)) + ',' + 'Recall\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(F1_Scores)) + ',F1_Scores\n')
    print('All models have finished running, the result file are locate in ' + file_path)

    results_file.close()


