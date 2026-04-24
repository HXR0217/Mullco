from sklearn.metrics import f1_score
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
clf=joblib.load('all_data')
test_x,test_y = np.loadtxt('test_x_xg_7mer_selectfrommodel2'),np.loadtxt('test_label_1+7mer.csv')
predic= clf.predict(test_x).toarray()
# 计算F1-score
f1 = f1_score(test_y, predic)

print("F1-score:", f1)