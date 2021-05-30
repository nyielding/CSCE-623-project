# %%
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
from sgp4.api import Satrec
import pandas as pd
import numpy as np
import os
import time
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

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
# TLE_FILES = ['kristall.txt', 'kvant-1.txt', 'kvant-2.txt', 'mir.txt', 'priroda.txt',
#             'salyut-7.txt', 'spektr.txt', 'zarya.txt']
TLE_FILES = ['met3-01.txt', 'met3-02.txt', 'met3-03.txt', 'met3-04.txt', 'met3-05.txt', 'met3-06.txt']
train_models = True

# %%
def quick_model(models_dict, Xtr, Xte, ytr, yte):
    score_list = []
    for key in models_dict:
        models_dict[key].fit(X=Xtr, y=ytr)
        score = models_dict[key].score(Xte, yte)
        print(' '.join([key, 'score: ']), score)
        score_list.append(score)
    return score_list


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
        figname = os.path.join('./plots/', 'coe' + title + '.png')
        plt.savefig(figname)


def run_pairplot(data, scalar, le, title):
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
    plt.savefig('./plots/' + title + '.png')


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


def saveModel(model=None, name='model'):
    filename = os.path.join('./models/','coe' + name + '.joblib')
    dump(model, filename=filename)


def loadModel(name=None):
    filename = os.path.join('./models/', name + '.joblib')
    clf = load(filename)
    return clf


# %%
# Read the first 1000 TLEs from files
columns = ['mean_motion','inclo', 'RAAN', 'ecc', 'argpo', 'm_anomaly', 'sat_name']
dflist = []
for sat_name in TLE_FILES:
    TLEs = open(sat_name, 'r')
    sats = np.zeros([1000, 6])
    for i in range(1000):
        line1 = TLEs.readline()
        line2 = TLEs.readline()
        satellite = Satrec.twoline2rv(line1, line2)
        # e, r, v = satellite.sgp4(satellite.jdsatepoch, satellite.jdsatepochF)
        sats[i, 0] = satellite.no_kozai
        sats[i, 1] = satellite.inclo
        sats[i, 2] = satellite.nodeo
        sats[i, 3] = satellite.ecco
        sats[i, 4] = satellite.argpo
        sats[i, 5] = satellite.mo
        # sats[i, 3:6] = v
        # sats[i, 6] = e
    name_column = sat_name[:-4]
    df_temp = pd.DataFrame(data=sats, columns=columns[:-1])
    df_temp['sat_name'] = name_column
    dflist.append(df_temp)

df = pd.concat(dflist)
# df['r_mag'] = np.sqrt(df.rx**2 + df.ry**2 + df.rz**2)
# df['v_mag'] = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)
TLEs.close()

# %%
df = df.dropna()
df = df[(df[['mean_motion']] != 0).any(axis=1)]
# df.head()
print(df.describe())


# %% Sequester test set
columns = list(df.columns)
train, test = train_test_split(df,
                             random_state=random_state, 
                             test_size=0.2, 
                             stratify=df[['sat_name']])

test = pd.DataFrame(test, columns=columns)
train = pd.DataFrame(train, columns=columns)

# %% Split train into train/val
X_train, X_val, y_train, y_val = train_test_split(train.drop(['sat_name'], axis='columns'),
                                                    train['sat_name'],
                                                    random_state=random_state,
                                                    test_size=0.2,
                                                    stratify=train[['sat_name']])

# %%
# scaler = StandardScaler()
scaler = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=random_state)
le = LabelEncoder()
X_train_t = scaler.fit_transform(X_train)
# X_train_t = X_train
y_train_t = le.fit_transform(y_train)
X_test_t = scaler.transform(X_val)
# X_test_t = X_val
y_test_t = le.transform(y_val)

# %%
models = {
    'LogisticRegression': LogisticRegression(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'SVC': SVC(),
    'NuSVC': NuSVC(),
    'LinearSVC': LinearSVC(),
    'SGCDClass': SGDClassifier(),
    'DecisionTree': DecisionTreeClassifier(max_depth=10),
    'RandomForest': RandomForestClassifier(max_depth=10),
    # 'BoostedTree': GradientBoostingClassifier()
}

# %% Quick score test of basic models
print('\nScores only scaling: ')
scores = quick_model(models, X_train_t, X_test_t, y_train_t, y_test_t)

# %% Do a grid search
if train_models:
    log_params = {'C': [0.1, 1, 100, 1000, 10000, 100000, 1000000],
        # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 10000],
        'n_jobs': [-1],
        'penalty': ['l1', 'l2', 'elasticnet']}
    lda_params = {'solver': ['svd', 'lsqr', 'eigen'],
        }
    tree_params = {'max_depth': [15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30],
        'criterion': ['gini', 'entropy'],
        # 'min_samples_split': [2, 10, 50, 100]
        }
    rf_params = {'max_depth': [18, 19, 20, 21, 22, 23],
        'criterion': ['gini', 'entropy'],
        # 'min_samples_split': [2, 10, 50, 100],
        # 'n_estimators': [100, 200], # this seems to have no effect on the score
        'n_jobs': [-1]
        }
    boost_params = {'max_depth': [10, 11, 12, 13, 15],
        'n_estimators': [200, 300, 400],
        # 'learning_rate': [0.05, 0.1, 0.15]} # not significant
    }
    svc_params = {'C': [1, 100, 1000, 10000, 100000, 1000000], 
        'kernel': ['rbf'],
        # 'gamma': ['scale', 'auto'] # not significant
        }
    ens_params = {'RF__max_depth': [20],
        'RF__criterion': ['gini'],
        'GB__max_depth': [11],
        'GB__n_estimators': [200],
        'SVC__C': [1000000],
        'weights': [[1, 1, 1], [1, 1, 2], [1, 1, 1.9]]
        }

    estimator0 = LogisticRegression(random_state=random_state)
    estimator01 = LinearDiscriminantAnalysis()
    estimator1 = DecisionTreeClassifier(random_state=random_state)
    estimator2 = RandomForestClassifier(random_state=random_state)
    estimator3 = GradientBoostingClassifier(random_state=random_state)
    estimator4 = SVC(random_state=random_state)
    ensemble = VotingClassifier(estimators=[('RF', estimator2), ('GB', estimator3), ('SVC', estimator4)],
        n_jobs=-1)

    # print('\nLogistic Regression: \n')
    # log_grid = myGridSearch(estimator=estimator0, 
    #                         tuned_parameters=log_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    # saveModel(model=log_grid.best_estimator_, name='LogisticRegModel')

    # print('\nLDA: \n')
    # lda_grid = myGridSearch(estimator=estimator01, 
    #                         tuned_parameters=lda_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    # saveModel(model=lda_grid.best_estimator_, name='LDAModel')

    print('\nDecision Tree: \n')
    tree_grid = myGridSearch(estimator=estimator1, 
                            tuned_parameters=tree_params, 
                            X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    saveModel(model=tree_grid.best_estimator_, name='DecisionTreeModel')
    
    print('\nRandom Forest: \n')
    rf_grid = myGridSearch(estimator=estimator2, 
                            tuned_parameters=rf_params, 
                            X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    saveModel(model=rf_grid.best_estimator_, name='RandomForestModel')   
    
    print('\nGradient Boost Tree: \n')
    boost_grid = myGridSearch(estimator=estimator3, 
                            tuned_parameters=boost_params, 
                            X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    saveModel(model=boost_grid.best_estimator_, name='GradientBoostedModel')   
    
    print('\nState Vector Machine: \n')
    svc_grid = myGridSearch(estimator=estimator4, 
                            tuned_parameters=svc_params, 
                            X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    saveModel(model=svc_grid.best_estimator_, name='SupportVectorModel')

    # print('\nEnsemble Method: \n')
    # ens_grid = myGridSearch(estimator=ensemble, 
    #                         tuned_parameters=ens_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    # saveModel(model=ens_grid.best_estimator_, name='EnsembleMethods')

# %% Plot confusion matrices
class_names = le.classes_

# model = loadModel('coeLDAModel')
# y_pred = model.predict(X_test_t)
# cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
# title = 'coeLDA_CM'
# plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = loadModel('coeLogisticRegModel')
# y_pred = model.predict(X_test_t)
# cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
# title = 'coeLogisticRegCM'
# plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = tree_grid.best_estimator_
model = loadModel('coeDecisionTreeModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'coeDecisionTreeCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = rf_grid.best_estimator_
model = loadModel('coeRandomForestModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'coeRandomForestCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = boost_grid.best_estimator_
model = loadModel('coeGradientBoostedModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'coeBoostedTreeCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = svc_grid.best_estimator_
model = loadModel('coeSupportVectorModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'coeSupportVectorMachineCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)

# model = loadModel('EnsembleMethods')
# y_pred = model.predict(X_test_t)
# cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
# title = 'EnsembleMethodsCM'
# plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
