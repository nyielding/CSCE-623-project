# %%
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
from sgp4.api import Satrec
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve

from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
# from ray.tune.sklearn import TuneGridSearchCV

# %% GLOBALS
random_state = np.random.RandomState(42)
TLE_FILES = ['kristall.txt', 'kvant-1.txt', 'kvant-2.txt', 'mir.txt', 'priroda.txt',
            'salyut-7.txt', 'spektr.txt', 'zarya.txt']
# TLE_FILES = ['spektr.txt', 'zarya.txt']

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
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          savefig=False):
    """
    this function is from https://sklearn.org/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savefig:
        figname = os.path.join('./plots/', title + '.png')
        plt.savefig(figname)


# %%
def run_pairplot(data, scalar, le):
    data.reset_index(inplace=True)
    labels = data['sat_name']
    train_forplot = data.drop(['error', 'sat_name', 'index'], axis='columns')
    plot_columns = list(train_forplot.columns)
    print(plot_columns)
    train_forplot_t = scaler.transform(train_forplot)
    train_forplot = pd.DataFrame(train_forplot_t, columns=plot_columns)
    train_forplot['sat_name'] = labels

    sns.set(font_scale=2)
    sns.pairplot(train_forplot, hue='sat_name')
    plt.savefig('./plots/scaled_pairs_multi.png')


def myGridSearch(estimator=None, tuned_parameters=None,X_train=None, X_test=None, y_train=None, y_test=None):
    start = time.time()
    scores = ['accuracy']
    #for other scorers, see https://scikit-learn.org/stable/modules/model_evaluation.html 


    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            estimator, tuned_parameters, scoring=score,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found during crossval:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
    end = time.time()
    print('myGridsearch Time: ', end-start)
    return clf


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
    name_column = sat_name[:-4]
    df_temp = pd.DataFrame(data=sats, columns=columns[:-1])
    df_temp['sat_name'] = name_column
    dflist.append(df_temp)

df = pd.concat(dflist)
df['r_mag'] = np.sqrt(df.rx**2 + df.ry**2 + df.rz**2)
df['v_mag'] = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)
TLEs.close()

# %%
df = df.dropna()
df.head()
df.describe()


# %% Sequester test set
columns = list(df.columns)
train, test = train_test_split(df,
                             random_state=random_state, 
                             test_size=0.2, 
                             stratify=df[['sat_name']])

test = pd.DataFrame(test, columns=columns)
train = pd.DataFrame(train, columns=columns)

# %% Split train into train/val
X_train, X_val, y_train, y_val = train_test_split(train.drop(['sat_name', 'error'], axis='columns'),
                                                    train['sat_name'],
                                                    random_state=random_state,
                                                    test_size=0.2,
                                                    stratify=train[['sat_name']])

# %%
scaler = StandardScaler()
le = LabelEncoder()
X_train_t = scaler.fit_transform(X_train)
y_train_t = le.fit_transform(y_train)
X_test_t = scaler.transform(X_val)
y_test_t = le.transform(y_val)

# %%
models = {
    # 'LogisticRegression': LogisticRegression(),
    # 'LDA': LinearDiscriminantAnalysis(),
    # 'QDA': QuadraticDiscriminantAnalysis(),
    'SVC': SVC(),
    # 'NuSVC': NuSVC(),
    # 'LinearSVC': LinearSVC(),
    # 'SGCDClass': SGDClassifier()
    # 'DecisionTree': DecisionTreeClassifier(max_depth=10),
    # 'RandomForest': RandomForestClassifier(max_depth=10),
    # 'BoostedTree': GradientBoostingClassifier()
}

# %% Quick score test of basic models
# print('\nScores only scaling: ')
# scores = quick_model(models, X_train_t, X_test_t, y_train_t, y_test_t)

# %% Do a grid search
tree_params = {'max_depth': [5, 10, 15, 20, 25, 30],}
boost_params = {'max_depth': [3, 5, 8, 10, 12, 15]}
svc_params = {'C': [1, 5, 10, 15], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [3, 4, 5]}
nu_params = {'nu': [0.5], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [3, 4, 5]}
estimator1 = DecisionTreeClassifier(random_state=random_state)
estimator2 = RandomForestClassifier(random_state=random_state)
estimator3 = GradientBoostingClassifier(random_state=random_state)
estimator4 = SVC(random_state=random_state)
estimator5 = NuSVC(random_state=random_state)

# print('\nDecision Tree: \n')
# tree_grid = myGridSearch(estimator=estimator1, 
#                         tuned_parameters=tree_params, 
#                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)

# print('\nRandom Forest: \n')
# rf_grid = myGridSearch(estimator=estimator2, 
#                         tuned_parameters=tree_params, 
#                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)

# print('\nGradient Boost Tree: \n')
# boost_grid = myGridSearch(estimator=estimator3, 
#                         tuned_parameters=boost_params, 
#                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)

print('\nState Vector Machine: \n')
svc_grid = myGridSearch(estimator=estimator4, 
                        tuned_parameters=svc_params, 
                        X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)

print('\nNuSVC: \n')
nu_svc_grid = myGridSearch(estimator=estimator5, 
                        tuned_parameters=nu_params, 
                        X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
# %% Plot confusion matrices
class_names = le.classes_

# model = tree_grid.best_estimator_
# y_pred = model.predict(X_test_t)
# cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
# title = 'DecisionTreeCM'
# plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = rf_grid.best_estimator_
# y_pred = model.predict(X_test_t)
# cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
# title = 'RandomForestCM'
# plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = boost_grid.best_estimator_
# y_pred = model.predict(X_test_t)
# cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
# title = 'BoostedTreeCM'
# plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

model = svc_grid.best_estimator_
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'StateVectorMachineCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

model = nu_svc_grid.best_estimator_
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'NuSVC_CM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
# %%
