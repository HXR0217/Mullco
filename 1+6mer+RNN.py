# 姓名：黄
# 日期：2022.04.16
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import itertools
from Bio import SeqIO
import tensorflow as tf
import pickle as pkl
import os
import sys
import re
from sklearn.model_selection import KFold
from datetime import datetime as dt
import pandas as pd
import numpy as np
from keras.layers.core import Lambda
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.problem_transform import LabelPowerset
from TrainingMetrics import evalu_kfold
import joblib
from keras.layers import Dense, Dropout,Input
from keras import layers
from keras.models import Model
from skmultilearn.ext import Keras

from keras.layers import  Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU


from keras.models import Sequential




my_clusters= [[2, 3, 5], [3, 5, 6], [2, 6, 8], [2, 6, 7], [2, 5, 6], [2, 3, 8], [2, 3, 7], [2, 3, 6], [1, 4, 6],
             [1, 3, 6], [1, 3, 5], [1, 2, 6], [1, 2, 3], [0, 3, 6], [0, 2, 6], [0, 2, 3], [0, 1, 6], [0, 1, 4]]
name='RNN+1+6mer'
method='5_fold_RNN'
train_x,train_y,test_x,test_y=np.loadtxt('train_fea_1+6mer.csv'),np.loadtxt('train_label_1+6mer.csv'),np.loadtxt('test_fea_1+6mer.csv'),np.loadtxt('test_label_1+6mer.csv')


def create_model_multiclass(input_dim, output_dim):
    # create model
    inputdata=Input(shape=(input_dim,1))
    x1 = GRU(64, activation='selu')(inputdata)
    x2 = Dense(64, activation='selu')(x1)
    x = Dropout(0.1)(x2)
    layer_out = Dense(output_dim, activation='softmax', name='layer_out')(x)
    model = Model(inputs=inputdata, outputs=layer_out)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model





KERAS_PARAMS = dict(epochs=50, batch_size=128, verbose=0)
cc = Keras(create_model_multiclass, True, KERAS_PARAMS)


def perform_test(clf,name):
    predic= clf.predict(test_x).toarray()
    ac,one_error, cov, rank_loss, ham_loss, ap = evalu_kfold(test_y, predic)
    filename_test = open('testoutput_'+str(method)+'.txt', "a")
    print(str(name)+':', file=filename_test)
    print('independent test ACC', ac, file=filename_test)
    print('independent test One-error', one_error,file=filename_test)
    print('independent test Coverage', cov,file=filename_test)
    print('independent test Ranking Loss', rank_loss,file=filename_test)
    print('independent test Hamming Loss', ham_loss,file=filename_test)
    print('independent test AveragePrecision', ap,file=filename_test)
    #print(predic)
    # calculate single-label performance
    from sklearn.metrics import accuracy_score
    localization_set = {0:'Exosome', 1:'Nucleus', 2:'Nucleoplasm',
                        3:'Chromatin', 4:'Cytoplasm', 5:'Nucleolus',
                        6:'Cytosol', 7:'Membrane', 8:'Ribosome'}
    for i in range(9):
        acc = accuracy_score(test_y[:,i],predic[:,i])
        print(localization_set[i],'Accuracy is',acc,file=filename_test)





def perform_model(train_x, train_y, val_x, val_y,cc,name):
    model = MajorityVotingClassifier(
        classifier=LabelPowerset(
            classifier=cc,
            require_dense=[True, False]
        ),
        clusterer=FixedLabelSpaceClusterer(clusters=my_clusters),
        require_dense=[False, False]
    )
    clf = model.fit(train_x,train_y)
    clf.fit(train_x, train_y)
    predic = clf.predict(val_x).toarray()
    joblib.dump(clf,str(name))
    perform_test(clf,name)
    acc, one_error, cov, rank_loss, ham_loss, ap = evalu_kfold(val_y, predic)


    return acc, one_error, cov, rank_loss, ham_loss, ap





def fold_5_cross_valication(x, y,name,cc):
    ACC = []
    ONE_E = []
    COVER = []
    RANKLOSS = []
    HAMLOSS = []
    AP = []


    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    for train_index, test_index in kf.split(x):
        train_x, train_y = x[train_index], y[train_index]
        val_x, val_y = x[test_index], y[test_index]
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(val_x)
        test_y = np.array(val_y)

        ac, one, cov, rankloss, hamloss, ap = perform_model(train_x, train_y, test_x, test_y,cc,name)

        ACC.append(ac)
        ONE_E.append(one)
        COVER.append(cov)
        RANKLOSS.append(rankloss)
        HAMLOSS.append(hamloss)
        AP.append(ap)

        print('5-fold ACC', ACC)
        print('5-fold One-error', ONE_E)
        print('5-fold Coverage', COVER)
        print('5-fold RankingLoss', RANKLOSS)
        print('5-fold HammingLoss', HAMLOSS)
        print('5-fold AveragePrecision', AP)


    filename = open('output_' +str(method)+ '.txt', "a")
    print(str(name))
    print('5-fold ACC', np.mean(ACC),file=filename)
    print('5-fold One-error', np.mean(ONE_E),file=filename)
    print('5-fold Coverage', np.mean(COVER),file=filename)
    print('5-fold RankingLoss', np.mean(RANKLOSS),file=filename)
    print('5-fold HammingLoss', np.mean(HAMLOSS),file=filename)
    print('5-fold AveragePrecision', np.mean(AP),file=filename)
    print()

fold_5_cross_valication(train_x,train_y,name,cc)