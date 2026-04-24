# 姓名：黄
# 日期：2022.04.16
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime as dt
import numpy as np
from TrainingMetrics import evalu_kfold
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.problem_transform import LabelPowerset
import pandas as pd
from sklearn.feature_selection import SelectFromModel

method='7mer_selectfrommodel1'
train_x,train_y=np.loadtxt('train_x_selectfrommodel1'),np.loadtxt('train_label_1+7mer_all.csv')
print(train_x.shape)
test_x= np.loadtxt('elseraw_independen_selectfea1')

print(test_x.shape)
#(33274, 21844)



name=('extra_selectfrommodel')



def selectfrommodel():
    selector = SelectFromModel(estimator=XGBClassifier() , threshold="mean")
    selected_train_x = selector.fit_transform(train_x, train_y)
    joblib.dump(selector,'selector2')

    selected_test_x= selector.transform(test_x)
    df = pd.DataFrame(selected_test_x)
    df.to_csv('elseraw_independen_selectfea2', index=False, header=False, sep=" ")


selectfrommodel()


