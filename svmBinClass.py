import pandas as pd
from pandas import read_csv
import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv("/home/miles/data/processed.cleveland.train.csv")
df.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
df = df.dropna()
print(df)

df['14'] = df['14'].apply(lambda x: 1 if x == 0 else -1)
print(df)

y = df['14']
df.drop('14', axis = 1)
X = df

clf = svm.SVC()
clf.fit(X,y)
dfTest = pd.read_csv("/home/miles/data/processed.cleveland.test.csv")
dfTest.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
dfTest['14'] = dfTest['14'].apply(lambda x: 1 if x == 0 else -1)
dfTest = dfTest.dropna()
TrueVals = dfTest['14']
dfTest.drop('14', axis = 1)
predictedVals = clf.predict(dfTest)
print(predictedVals)
print(roc_auc_score(TrueVals, clf.predict(dfTest)))
print (f1_score(TrueVals, predictedVals))

lr = LogisticRegression(max_iter=10000)
lr.fit(X, y)
print(lr.score(dfTest, TrueVals))
def getScore(model, X_train , X_test, y_Train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
Kf = KFold(n_splits=10)
dfFull = pd.read_csv("/home/miles/data/processed_cleveland_train.csv")
dfFull = dfFull.dropna()
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in kf.split(dfFull):
    X_train, X_test, y_train, y_test = dfFull.data[train_index], dfFull.data[test_index], dfFull.target[train_index], dfFull.target[test_index] 