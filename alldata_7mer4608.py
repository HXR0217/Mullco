# 姓名：黄
# 日期：2022.04.16
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime as dt
import numpy as np
from TrainingMetrics import evalu_kfold
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import hamming_loss
from xgboost import XGBClassifier
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.problem_transform import LabelPowerset
import pandas as pd
import pickle
import math
from sklearn import svm


my_clusters= [[2, 3, 5], [3, 5, 6], [2, 6, 8], [2, 6, 7], [2, 5, 6], [2, 3, 8], [2, 3, 7], [2, 3, 6], [1, 4, 6],
             [1, 3, 6], [1, 3, 5], [1, 2, 6], [1, 2, 3], [0, 3, 6], [0, 2, 6], [0, 2, 3], [0, 1, 6], [0, 1, 4]]

x,y=np.loadtxt('train_x_xg_7mer_selectfrommodel2'),np.loadtxt('train_label_1+7mer.csv')
test_x,test_y = np.loadtxt('test_x_xg_7mer_selectfrommodel2'),np.loadtxt('test_label_1+7mer.csv')
print(x.shape)
print(test_x.shape)
def perform_test(clf,name):
    predic= clf.predict(test_x).toarray()
    ac,one_error, cov, rank_loss, ham_loss, ap = evalu_kfold(test_y, predic)
    filename_test = open('testoutput_alldata_notiaocanxgboost' +'.txt', "a")
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




l_rate = 0.3
m_depth =8
estimators=1600

name = str(l_rate) + '_' + str(m_depth) + '_' + str(estimators)

xgb= XGBClassifier(tree_method='auto', use_label_encoder=False, objective='multi:softmax',
                                        max_depth=m_depth,
                                        learning_rate=l_rate, n_estimators=estimators,
                                        eval_metric='mlogloss',n_jobs=-1)

model = MajorityVotingClassifier(
        classifier=LabelPowerset(
            classifier=xgb,
            require_dense=[True, False]
        ),
        clusterer=FixedLabelSpaceClusterer(clusters=my_clusters),
        require_dense=[False, False]
    )

clf = model.fit(x,y)
joblib.dump(clf,str(name))
perform_test(clf,name)