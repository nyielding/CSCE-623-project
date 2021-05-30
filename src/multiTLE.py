# %%
import itertools
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import axes, axis
from sgp4.api import Satrec
import pandas as pd
import numpy as np
import os
import time
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
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

from sklearn.model_selection import GridSearchCV
# from ray.tune.sklearn import TuneGridSearchCV

# %% GLOBALS
random_state = np.random.RandomState(42)
# TLE_FILES = ['kristall.txt', 'kvant-1.txt', 'kvant-2.txt', 'mir.txt', 'priroda.txt',
#             'salyut-7.txt', 'spektr.txt', 'zarya.txt']
TLE_FILES = ['met3-01.txt', 'met3-02.txt', 'met3-03.txt', 'met3-04.txt', 'met3-05.txt', 'met3-06.txt']
train_models = False
pca_analysis = False
run_test_outputs = True

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
        figname = os.path.join('./plots/', title + '.png')
        plt.savefig(figname)


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


def run_PCA(X_train):
    pca = PCA(X_train.shape[1])  #the no-loss PCA transform
    pca.fit_transform(X_train)
    evr = pca.explained_variance_ratio_
    fig = plt.figure(figsize = (16,9))
    ax = fig.add_subplot(1,1,1)
    percent_ticks = np.linspace(0, 1.00, 21)
    component_index = np.arange(1,X_train.shape[1]+1,1)
    plt.plot(component_index,np.cumsum(evr))
    ax.set_yticks(percent_ticks)
    component_ticks=[1,3,10,30,100,300]
    #ax.set_xticks(component_ticks)
    ax.set_xlim(0,9)
    plt.xlabel('N Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.savefig('./plots/pca.png')
    return pca.components_, pca.explained_variance_, pca.explained_variance_ratio_

def myGridSearch(estimator=None, tuned_parameters=None,X_train=None, X_test=None, y_train=None, y_test=None, cv=None, verbose=0):
    start = time.time()
    scores = ['accuracy']
    #for other scorers, see https://scikit-learn.org/stable/modules/model_evaluation.html 


    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            estimator, tuned_parameters, scoring=score,
            n_jobs=-1, return_train_score=True, verbose=verbose, cv=cv
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found during crossval:")
        print()
        print(clf.best_params_)
        print()
        print("With %s: %0.3f" % (score, clf.best_score_))
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        meant = clf.cv_results_['mean_train_score']
        stdst = clf.cv_results_['std_train_score']
        for mean, std, mt, st, params in zip(means, stds, meant, stdst, clf.cv_results_['params']):
            print("Test: %0.3f (+/-%0.03f) Train: %0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, mt, st *2, params))
    end = time.time()
    print('myGridsearch Time: ', end-start)
    return clf


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, 
    filename='grid_search', title='Grid Search Scores', logscale=False):
    '''
    From: https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    Example call: 
    plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')
    '''
    # Get Test Scores Mean and std for each grid search
    scores_mean_list = cv_results['mean_test_score']
    scores_mean = np.zeros([len(grid_param_2), len(grid_param_1)])
    # scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    for i in range(len(grid_param_2)):
        scores_mean[i,:] = scores_mean_list[i::len(grid_param_2)]

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    if logscale:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')
    plt.savefig('./plots/' + filename + '.png')


def saveModel(model=None, name='model'):
    filename = os.path.join('./models/', name + '.joblib')
    dump(model, filename=filename)


def loadModel(name=None):
    filename = os.path.join('./models/', name + '.joblib')
    clf = load(filename)
    return clf


def saveResults(cv_results_=None, filename=None):
    with open('./grid_results/'+filename+'.pickle','wb') as f:
        pickle.dump(cv_results_, f)
    

def loadResults(filename=None):
    with open('./grid_results/'+filename+'.pickle', 'rb') as f:
        data = pickle.load(f)
    return data


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
X_train, X_val, y_train, y_val = train_test_split(train.drop(['sat_name', 'error'], axis='columns'),
                                                    train['sat_name'],
                                                    random_state=random_state,
                                                    test_size=0.2,
                                                    stratify=train[['sat_name']])

# %%
scaler = StandardScaler()
# qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=random_state)
# ss = StandardScaler()
# scaler = ColumnTransformer([('quantile', qt, [0, 1, 3, 4]), ('standard', ss, [2, 5, 6, 7])])
le = LabelEncoder()
X_train_t = scaler.fit_transform(X_train)
y_train_t = le.fit_transform(y_train)
X_test_t = scaler.transform(X_val)
y_test_t = le.transform(y_val)

# %% Get a pairplot of the scaled training data
# run_pairplot(train, scaler, le)

# # %%
# models = {
#     'LogisticRegression': LogisticRegression(),
#     'LDA': LinearDiscriminantAnalysis(),
#     'QDA': QuadraticDiscriminantAnalysis(),
#     'SVC': SVC(),
#     'NuSVC': NuSVC(),
#     'LinearSVC': LinearSVC(),
#     'SGCDClass': SGDClassifier(),
#     'DecisionTree': DecisionTreeClassifier(max_depth=10),
#     'RandomForest': RandomForestClassifier(max_depth=10),
#     'BoostedTree': GradientBoostingClassifier()
# }

# # %% Quick score test of basic models
# print('\nScores only scaling: ')
# scores = quick_model(models, X_train_t, X_test_t, y_train_t, y_test_t)
if pca_analysis:
    run_PCA(X_train_t)

# %% Do a grid search
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
svc_C = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
svc_params = {'C': svc_C, 
    'kernel': ['sigmoid', 'rbf'],
    # 'gamma': ['scale', 'auto'] # not significant
    }
svc_p_C = [1, 10, 100, 1000]
svc_degrees = [3, 4, 5]
svc_poly_params = {'C': svc_p_C,
    'kernel': ['poly'],
    'degree': svc_degrees,
    }
# svc_gamma = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
start = 0.1
step = 0.01
stop = 0.35
svc_gamma = np.arange(start, stop+step, step)
svc_gamma_params = {'C': [10000000],
    'kernel': ['rbf'],
    'gamma': svc_gamma,
    }
# ens_params = {'RF__max_depth': [20],
#     'RF__criterion': ['gini'],
#     'GB__max_depth': [11],
#     'GB__n_estimators': [200],
#     'SVC__C': [1000000],
#     'weights': [[1, 1, 1], [1, 1, 2], [1, 1, 1.9]]
#     }
if train_models:
    estimator1 = DecisionTreeClassifier(random_state=random_state)
    estimator2 = RandomForestClassifier(random_state=random_state)
    estimator3 = GradientBoostingClassifier(random_state=random_state)
    estimator4 = SVC(random_state=random_state, probability=True)
    estimator4pf = SVC(random_state=random_state, probability=False)
    # ensemble = VotingClassifier(estimators=[('RF', estimator2), ('GB', estimator3), ('SVC', estimator4)],
    #     n_jobs=-1)

    # print('\nDecision Tree: \n')
    # tree_grid = myGridSearch(estimator=estimator1, 
    #                         tuned_parameters=tree_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    # saveResults(tree_grid.cv_results_, 'dt_results')
    # saveModel(model=tree_grid.best_estimator_, name='DecisionTreeModel')
    
    # print('\nRandom Forest: \n')
    # rf_grid = myGridSearch(estimator=estimator2, 
    #                         tuned_parameters=rf_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    # saveResults(rf_grid.cv_results_, 'rf_results')
    # saveModel(model=rf_grid.best_estimator_, name='RandomForestModel')   
    
    # print('\nGradient Boost Tree: \n')
    # boost_grid = myGridSearch(estimator=estimator3, 
    #                         tuned_parameters=boost_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    # saveResults(boost_grid.cv_results_, 'boost_results')
    # saveModel(model=boost_grid.best_estimator_, name='GradientBoostedModel')   
    
    print('\Support Vector Machine: \n')
    # svc_grid = myGridSearch(estimator=estimator4, 
    #                         tuned_parameters=svc_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t,
    #                         verbose=4,
    #                         cv=5)
    # saveResults(svc_grid.cv_results_, 'svc_results')
    # saveModel(model=svc_grid.best_estimator_, name='SupportVectorModel')

    # svc_poly_grid = myGridSearch(estimator=estimator4pf, 
    #                         tuned_parameters=svc_poly_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t,
    #                         verbose=4,
    #                         cv=5)
    # saveResults(svc_poly_grid.cv_results_, 'svc_poly_results')
    # saveModel(model=svc_poly_grid.best_estimator_, name='SupportVectorModelpoly')

    svc_gamma_grid = myGridSearch(estimator=estimator4pf, 
                            tuned_parameters=svc_gamma_params, 
                            X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t,
                            verbose=4,
                            cv=5)
    saveResults(svc_gamma_grid.cv_results_, 'svc_gamma_results')
    saveModel(model=svc_gamma_grid.best_estimator_, name='SupportVectorModelgamma')

    # print('\nEnsemble Method: \n')
    # ens_grid = myGridSearch(estimator=ensemble, 
    #                         tuned_parameters=ens_params, 
    #                         X_train=X_train_t, X_test=X_test_t, y_train=y_train_t, y_test=y_test_t)
    # saveModel(model=ens_grid.best_estimator_, name='EnsembleMethods')

# %% Plot confusion matrices
class_names = le.classes_

# model = tree_grid.best_estimator_
model = loadModel('DecisionTreeModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'DecisionTreeCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
plot_confusion_matrix(cm=cm, classes=class_names, title='nn'+title, normalize=False, savefig=True)
# y_pred_train = model.predict(X_train_t)
# cmt = confusion_matrix(y_true=y_train_t, y_pred=y_pred_train)
# plot_confusion_matrix(cm=cmt, classes=class_names, title='train '+title, normalize=True, savefig=True)

# model = rf_grid.best_estimator_
model = loadModel('RandomForestModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'RandomForestCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
plot_confusion_matrix(cm=cm, classes=class_names, title='nn'+title, normalize=False, savefig=True)
# y_pred_train = model.predict(X_train_t)
# cmt = confusion_matrix(y_true=y_train_t, y_pred=y_pred_train)
# plot_confusion_matrix(cm=cmt, classes=class_names, title='train '+title, normalize=True, savefig=True)


# model = boost_grid.best_estimator_
model = loadModel('GradientBoostedModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'BoostedTreeCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
plot_confusion_matrix(cm=cm, classes=class_names, title='nn'+title, normalize=False, savefig=True)
# y_pred_train = model.predict(X_train_t)
# cmt = confusion_matrix(y_true=y_train_t, y_pred=y_pred_train)
# plot_confusion_matrix(cm=cmt, classes=class_names, title='train '+title, normalize=True, savefig=True)

# model = svc_grid.best_estimator_
model = loadModel('SupportVectorModel')
y_pred = model.predict(X_test_t)
cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
title = 'SupportVectorMachineCM'
plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
plot_confusion_matrix(cm=cm, classes=class_names, title='nn'+title, normalize=False, savefig=True)

# C based sigmoid/rbf grid
svc_kernels = ['sigmoid', 'rbf']
cv_results_ = loadResults('svc_results')
plot_grid_search(cv_results_, svc_C, svc_kernels, 'C Value', 'Kernel', filename='svc_grid_plot', logscale=True)

# C based poly grid
cv_results_ = loadResults('svc_poly_results')
plot_grid_search(cv_results_, svc_p_C, svc_degrees, 'C Value', 'Degree', filename='svc_poly_grid_plot', title='Polynomial Grid Search', logscale=True)

# Gamma based sigmoid/rbf grid
svc_kernels = ['rbf']
cv_results_ = loadResults('svc_gamma_results')
plot_grid_search(cv_results_, svc_gamma, svc_kernels, 'Gamma', 'Kernel', filename='svc_gamma_grid_plot', title='Gamma Grid Search C=1e7')

# y_pred_train = model.predict(X_train_t)
# cmt = confusion_matrix(y_true=y_train_t, y_pred=y_pred_train)
# plot_confusion_matrix(cm=cmt, classes=class_names, title='train '+title, normalize=True, savefig=True)

# model = loadModel('EnsembleMethods')
# y_pred = model.predict(X_test_t)
# cm = confusion_matrix(y_true=y_test_t, y_pred=y_pred)
# title = 'EnsembleMethodsCM'
# plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
# %% 
if run_test_outputs:
    X_test = test.drop(['sat_name', 'error'], axis='columns')
    y_test = test['sat_name']
    X_test = scaler.transform(X_test)
    y_test = le.transform(y_test)
    best_model = SVC(C=10000000, gamma=0.27, random_state=random_state, probability=True)
    best_model.fit(X_train_t, y_train_t)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    title = 'SupportVectorTest'
    plot_confusion_matrix(cm=cm, classes=class_names, title=title, normalize=True, savefig=True)
    plot_confusion_matrix(cm=cm, classes=class_names, title='nn'+title, normalize=False, savefig=True)

# %% Error analysis nightmare
run_eplots = True
if run_eplots:
    # Run error analysis
    cl_1, cl_2 = 0, 1
    x_11 = X_test[(y_test == cl_1) & (y_pred == cl_1)]
    x_12 = X_test[(y_test == cl_1) & (y_pred == cl_2)]
    x_21 = X_test[(y_test == cl_2) & (y_pred == cl_1)]
    x_22 = X_test[(y_test == cl_2) & (y_pred == cl_2)]
    x_11 = x_11[::20]
    x_22 = x_22[::20]
    class_11 = ['01 correct'] * len(x_11)
    class_22 = ['02 correct'] * len(x_22)
    class_12 = ['01 pred as 02'] * len(x_12)
    class_21 = ['02 pred as 01'] * len(x_21)
    x_array = np.vstack((x_11, x_12, x_21, x_22))
    class_list = class_11 + class_12 + class_21 + class_22
    columns = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'r_mag', 'v_mag']
    dfe = pd.DataFrame(x_array, columns=columns)
    dfe['errors'] = class_list

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for s in dfe.errors.unique():
        ax.scatter(dfe.rx[dfe.errors==s],
            dfe.ry[dfe.errors==s],
            dfe.rz[dfe.errors==s],
            label=s)
    ax.legend()
    plt.savefig('./plots/pos_errors.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for s in dfe.errors.unique():
        ax.scatter(dfe.vx[dfe.errors==s],
            dfe.vy[dfe.errors==s],
            dfe.vz[dfe.errors==s],
            label=s)
    ax.legend()
    plt.savefig('./plots/vel_errors.png')


    fig = plt.figure()
    sns.scatterplot(data=dfe, x='r_mag', y='v_mag', hue='errors', markers=['o', 's', 'v', 'D'])
    plt.savefig('./plots/mag_errors.png')
    

    plt.figure()
    sns.histplot(dfe, x='r_mag', y='v_mag', hue='errors')
    plt.savefig('./plots/error_hist_mag.png')
    # # sns.histplot(df_errors[['ry', 'errors']], hue='errors', ax=axes[1])
    # sns.histplot(df_errors, x='ry', hue='errors', ax=axes[1])
    # sns.set(font_scale=1)

    # sns.histplot(df_errors, x='rz', hue='errors', ax=axes[2])
    # # sns.set(font_scale=2)
    # plt.savefig('./plots/error_hist_r.png')

    # fig, axes = plt.subplots(1, 3)
    # fig.suptitle('Velocity Vector Errors')
    # sns.histplot(df_errors, x='vx', hue='errors', ax=axes[0])
    # # sns.set(font_scale=2)

    # sns.histplot(df_errors, x='vy', hue='errors', ax=axes[1])
    # # sns.set(font_scale=2)

    # sns.histplot(df_errors, x='vz', hue='errors', ax=axes[2])
    # # sns.set(font_scale=2)
    # plt.savefig('./plots/error_hist_v.png')

    # fig, axes = plt.subplots(1, 2)
    # fig.suptitle('Magnitude Errors')
    # sns.histplot(df_errors, x='r_mag', hue='errors', ax=axes[0])
    # # sns.set(font_scale=2)

    # sns.histplot(df_errors, x='v_mag', hue='errors', ax=axes[1])
    # # sns.set(font_scale=2)
    # plt.savefig('./plots/error_hist_mag.png')

    # sns.pairplot(df_errors.drop(['vx', 'vy', 'vz', 'r_mag', 'v_mag'], axis='columns'), hue='errors', markers=['o', 's', 'v', 'D'])
    # plt.savefig('./plots/error_pairs_r.png')

    # sns.set(font_scale=2)
    # sns.pairplot(df_errors.drop(['rx', 'ry', 'rz', 'r_mag', 'v_mag'], axis='columns'), hue='errors', markers=['o', 's', 'v', 'D'])
    # plt.savefig('./plots/error_pairs_v.png')

    # sns.set(font_scale=2)
    # sns.pairplot(df_errors[['r_mag', 'v_mag', 'errors']], hue='errors', markers=['o', 's', 'v', 'D'])
    # plt.savefig('./plots/error_pairs_mag.png')

