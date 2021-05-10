# %%
from sgp4.api import Satrec
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve

# %%
random_state = np.random.RandomState(42)
testrun = True
PATHUP = '../'
# TLE_FILES = ['kristall.txt', 'kvant-1.txt', 'kvant-2.txt', 'mir.txt', 'priroda.txt', 
#             'salyut-7.txt', 'spektr.txt', 'zarya.txt']
TLE_FILES = ['spektr.txt', 'zarya.txt']
# %%
if testrun:
    for i, txt in enumerate(TLE_FILES):
        TLE_FILES[i] = os.path.join(PATHUP, txt)


# %%
# Read the first 1000 TLEs from files
columns = ['rx','ry', 'rz', 'vx', 'vy', 'vz', 'error', 'sat_name']
dflist = []
for sat_name in TLE_FILES:
    TLEs = open(sat_name, 'r')
    sats = np.zeros([1000, 7])
    for i in range(1000):
        line1 = TLEs.readline()
        line2 = TLEs.readline()
        satellite = Satrec.twoline2rv(line1, line2)
        e, r, v = satellite.sgp4(satellite.jdsatepoch, satellite.jdsatepochF)
        sats[i, 0:3] = r
        sats[i, 3:6] = v
        sats[i, 6] = e
    if testrun:
        name_column = sat_name[3:-4]
    else:
        name_column = sat_name[:-4]
    df_temp = pd.DataFrame(data=sats, columns=columns[:-1])
    df_temp['sat_name'] = name_column
    dflist.append(df_temp)

df = pd.concat(dflist)

TLEs.close()

# %%
df.head()
df.describe()
# %%
df = df.dropna()

# %%
# df.groupby('sat_name').PetalWidth.plot(kind='kde')

# %%
# pd.plotting.scatter_matrix(df, figsize=[15,15])
# %%
X_train, X_test, y_train, y_test = train_test_split(df.drop(['sat_name', 'error'], axis='columns'),
                                                    df['sat_name'], 
                                                    random_state=random_state)
# %%
mmscaler = MinMaxScaler()
le = LabelEncoder()
X_train_t = mmscaler.fit_transform(X_train)
y_train_t = le.fit_transform(y_train)
X_test_t = mmscaler.transform(X_test)
y_test_t = le.transform(y_test)
# %%
model = LogisticRegression()
model.fit(X=X_train_t, y=y_train_t)
score = model.score(X_test_t, y_test_t)
print('LR score: ', score)
# %%
model2 = LinearDiscriminantAnalysis()
model2.fit(X=X_train_t, y=y_train_t)
score2 = model2.score(X_test_t, y_test_t)
print('LDA score: ', score2)
# %%
model3 = QuadraticDiscriminantAnalysis()
model3.fit(X=X_train_t, y=y_train_t)
score3 = model3.score(X_test_t, y_test_t)
print('QDA score: ', score3)
# %%
model4 = SVC()
model4.fit(X=X_train_t, y=y_train_t)
score4 = model4.score(X_test_t, y_test_t)
print('SVC score: ', score4)
# %%
model5 = NuSVC()
model5.fit(X=X_train_t, y=y_train_t)
score5 = model5.score(X_test_t, y_test_t)
print('NuSVC score: ', score5)
# %%
model6 = SGDClassifier()
model6.fit(X=X_train_t, y=y_train_t)
score6 = model6.score(X_test_t, y_test_t)
print('SGCDclass score: ', score6)
# %%
