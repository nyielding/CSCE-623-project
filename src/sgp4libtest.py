# %%
from sgp4.api import Satrec
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve

# %% GLOBALS
random_state = np.random.RandomState(42)
use_jupyter = True
PATHUP = '../'
# TLE_FILES = ['kristall.txt', 'kvant-1.txt', 'kvant-2.txt', 'mir.txt', 'priroda.txt',
#             'salyut-7.txt', 'spektr.txt', 'zarya.txt']
TLE_FILES = ['spektr.txt', 'zarya.txt']

# %%
if use_jupyter:
    for i, txt in enumerate(TLE_FILES):
        TLE_FILES[i] = os.path.join(PATHUP, txt)


# %%
def quick_model(models_dict, Xtr, Xte, ytr, yte):
    score_list = []
    for key in models_dict:
        models_dict[key].fit(X=Xtr, y=ytr)
        score = models_dict[key].score(Xte, yte)
        print(' '.join([key, 'score: ']), score)
        score_list.append(score)
    return score_list


# %%
# Read the first 1000 TLEs from files
columns = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'error', 'sat_name']
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
    if use_jupyter:
        name_column = sat_name[3:-4]
    else:
        name_column = sat_name[:-4]
    df_temp = pd.DataFrame(data=sats, columns=columns[:-1])
    df_temp['sat_name'] = name_column
    dflist.append(df_temp)

df_all = pd.concat(dflist)

TLEs.close()

# %%
df_all.head()
df_all.describe()
df_all = df_all.dropna()

# %%
columns = list(df_all.columns)
train, test = train_test_split(df_all,
                             random_state=random_state, 
                             test_size=0.2, 
                             stratify=df_all[['sat_name']])

df_test = pd.DataFrame(test, columns=columns)
df = pd.DataFrame(train, columns=columns)

# %%
pd.plotting.scatter_matrix(df.drop('error', axis='columns'), figsize=[15, 15])

# %%
X_train, X_test, y_train, y_test = train_test_split(df.drop(['sat_name', 'error'], axis='columns'),
                                                    df['sat_name'],
                                                    random_state=random_state,
                                                    test_size=0.2,
                                                    stratify=df[['sat_name']])
# %%
mm_scaler = StandardScaler()
le = LabelEncoder()
X_train_t = mm_scaler.fit_transform(X_train)
y_train_t = le.fit_transform(y_train)
X_test_t = mm_scaler.transform(X_test)
y_test_t = le.transform(y_test)

df_mag = df.drop(['error'], axis='columns')
df_mag['r_mag'] = np.sqrt(df.rx**2 + df.ry**2 + df.rz**2)
df_mag['v_mag'] = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)

# %% Try the magnitudes of the vectors instead.
X_mag_train, X_mag_test, y_mag_train, y_mag_test = train_test_split(df_mag[['r_mag', 'v_mag']],
                                                                    df_mag['sat_name'],
                                                                    random_state=random_state,
                                                                    test_size=0.2,
                                                                    stratify=df_mag[['sat_name']])

mm_mag_scaler = StandardScaler()

X_mag_train_t = mm_mag_scaler.fit_transform(X_mag_train)
X_mag_test_t = mm_mag_scaler.transform(X_mag_test)
y_mag_train_t = le.transform(y_mag_train)
y_mag_test_t = le.transform(y_mag_test)

# %% Plots
pd.plotting.scatter_matrix(
    df_mag[['r_mag', 'v_mag', 'sat_name']], figsize=[15, 15])
sns.pairplot(df_mag[['r_mag', 'v_mag', 'sat_name']], hue='sat_name')

# %%
models = {
    'LogisticRegression': LogisticRegression(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'SVC': SVC(),
    'NuSVC': NuSVC(),
    # 'LinearSVC': LinearSVC(),
    'SGCDClass': SGDClassifier()
}

# %%
print('\nScores only scaling: ')
scores = quick_model(models, X_train_t, X_test_t, y_train_t, y_test_t)

print('\nScores of vector magnitude: ')
scores_mag = quick_model(models, X_mag_train_t,
                         X_mag_test_t, y_mag_train_t, y_mag_test_t)
# %%
