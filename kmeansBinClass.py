import pandas as pd
from pandas import read_csv
import sklearn as sk
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import model_selection
from sklearn.model_selection import KFold

#load in and process data
dfFull = pd.read_csv("/home/miles/data/processed_cleveland_train.csv")
dfFull.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
dfFull['14'] = dfFull['14'].apply(lambda x: 1 if x == 0 else -1)
dfFull = dfFull.dropna()
#split data into main data and target/label data
dfFullData = dfFull.drop('14', axis = 1)
dfFullData = dfFullData.dropna()
dfFullTarget = dfFull['14']

#create a K fold object
kf = KFold(n_splits = 10, random_state=None, shuffle=False)

#perform cross validation on different index ranges of the data
for train_index, test_index in kf.split(dfFullData):
    X_train, X_test = dfFullData.iloc[train_index], dfFullData.iloc[test_index]
    y_train, y_test = dfFullTarget.iloc[train_index], dfFullTarget.iloc[test_index]
    model = svm.SVC(kernel='sigmoid', degree=4)
    model.fit(X_train, y_train)
    #display scores for each model in the cross validation
    print("for test index:" , test_index[0] , "to" , test_index[len(test_index)-1] )
    print()
    print("model score:" , model.score(X_test, y_test))
    print("roc score:" , roc_auc_score(y_test, model.predict(X_test)))
    print ("f1 score:" , f1_score(y_test, model.predict(X_test)))
    print()
    print("----------------------------")