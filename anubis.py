import time
import os
import argparse
import warnings

from functools import reduce

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from scipy.stats import boxcox

# Blending
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNetCV, OrthogonalMatchingPursuit
from sklearn.ensemble import RandomForestRegressor

warnings.simplefilter("ignore")

# Отсчет времени работы программы
start_time = time.time()
work_time = None


def debprint(func_name, *values, sep=' ', end='\n'):
    print(f'[{func_name} {time.asctime()[11:19]}] -', *values, sep=sep, end=end)


def data_prepairing():
    '''
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    LogisticRegression, LogisticRegression
    :return:
    '''
    # Загрузка данных
    raw_data = pd.read_csv(f'{PATH}/files/RawData.csv', sep=';')
    debprint('Loader', f'Data loaded, shape: {raw_data.shape}')

    # Удаление выбросов
    data = raw_data[raw_data['AGE'] >= 18]

    # Замена категориальных признаков на числовые
    for value in data['CLIENT_GROUP'].unique():
        data[str(value)] = (data['CLIENT_GROUP'] == value).values * 1
    data = data.drop('CLIENT_GROUP', axis=1)
    data = data.drop('nan', axis=1)

    # Замена NAN на подходящие значения
    for column in data.columns:
        if column != 'flg' and column != 'APPLICATION_ID':
            data[column] = data[column].fillna((data[column].max() - data[column].min()) / 2)

    # Масштабирование данных
    for column in data.columns:
        if column not in ['GB', 'flg', 'APPLICATION_ID']:
            data[column] = boxcox(data[column] + BOXCOX, lmbda=LAMBBA)
    debprint('BoxCox', f'BoxCox function applied, addition {BOXCOX}, lambda {LAMBBA}')

    # Удаление коррелирующих факторов
    corr = data.corr()
    ignore = ['GB', 'flg', 'APPLICATION_ID']
    to_delete = []

    for i in data.columns:
        for j in data.columns:
            if i in ignore or j in ignore:
                continue
            if abs(corr[i][j]) > (CORREL / 100) and i != j:
                if max(data[i]) - min(data[i]) > max(data[j]) - min(data[j]):
                    to_delete.append(i)
                else:
                    to_delete.append(j)

    data = data.drop(pd.unique(to_delete), axis=1)
    debprint('Correl', f'Сorrelation results: {len(to_delete)} factors were deleted')

    data = data.drop('flg', axis=1)
    data = data.sort_values('APPLICATION_ID')

    '''
    70% - 76.529
    75% - 76.656
    80% - 76.884
    95% - 77.783
    '''
    train = data[:int(data.shape[0] / 100 * SPLIT_PERCENT)]
    test = data[int(data.shape[0] / 100 * SPLIT_PERCENT):]
    debprint('Splitg', f'Splitting - {SPLIT_PERCENT}%: train {train.shape}, test - {test.shape}')

    # ### Деление данных на 2 части
    Xtrain = train.drop('GB', axis=1)
    Xtest = test.drop('GB', axis=1)
    Ytrain = train[['GB', 'APPLICATION_ID']].set_index('APPLICATION_ID')
    Ytest = test[['GB', 'APPLICATION_ID']].set_index('APPLICATION_ID')

    # Функция проверяющая монотонность
    def monotonic(x):
        dx = np.diff(x)
        return np.all(dx <= 0) or np.all(dx >= 0)

    # Функция поиска монотонных элементов
    def clear_data(train, min_samples_leaf):
        model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        to_delete = []
        for label in train.columns:
            if label == 'APPLICATION_ID':
                continue
            model.fit(train[label].values.reshape(-1, 1), Ytrain)
            bounds = model.tree_.threshold
            bounds = bounds[bounds != -2]

            if len(bounds) == 0:
                to_delete.append(label)
        return to_delete

    # Функция получения границ и весов для факторов
    def count_woe(data, samp):
        model = DecisionTreeClassifier(min_samples_leaf=samp)
        Xdata = data.drop('GB', axis=1)
        Ydata = data['GB']
        total_bad = Ydata.sum()
        total_good = Ydata.count() - total_bad
        table = pd.DataFrame()
        tmp = {}
        for label in Xdata.columns:
            if label == 'APPLICATION_ID':
                continue
            model.fit(Xdata[label].values.reshape(-1, 1), Ydata)
            bounds = model.tree_.threshold
            bounds = bounds[bounds != -2]

            if len(bounds) == 0:
                continue

            bounds.sort()
            bounds = np.append(bounds, np.inf)
            woe = []
            left = -np.inf
            for i in range(len(bounds)):
                check = (data[label] >= left) & (data[label] <= bounds[i])
                bad = Ydata[check].sum()
                good = Ydata[check].count() - bad
                if good == 0:
                    good += 0.5
                    bad += 0.5
                left = bounds[i]
                woe.append(np.log((bad / total_bad) / (good / total_good)))

            tmp['Factor'] = label
            tmp['Bounds'] = bounds
            tmp['WOE'] = woe
            tmp['MONO'] = monotonic(woe)
            table = table.append([tmp])

        return table

    # Функция поиска минимального кол-ва листьев для дерева
    def leafs_bin_search(data, left_border, right_border):
        if right_border - left_border < 2:
            return right_border

        pos = int((left_border + right_border) / 2)
        t = count_woe(data, pos)
        res = True
        if reduce((lambda x, y: x * y), t['MONO']) == False:
            res = leafs_bin_search(data, pos, right_border)
        else:
            res = leafs_bin_search(data, left_border, pos)
        return res

    # Функция перевода фаторов в их вес
    def factors_to_woe(data, woe_frame) -> pd.DataFrame:
        data = data.drop('GB', axis=1)
        data_copy = data.copy()
        for row in data.columns:
            if ((woe_frame['Factor'] == row).sum() == 0):
                continue
            bounds = woe_frame[woe_frame['Factor'] == row]['Bounds'][0]
            woe = woe_frame[woe_frame['Factor'] == row]['WOE'][0]

            left = -np.inf
            for i in range(len(bounds)):
                data_copy[row][(data_copy[row] >= left) & (data_copy[row] <= bounds[i])] = woe[i]
                left = bounds[i]
        return data_copy

    # Получение таблицы весов для каждого запроса
    if COUNT_LEAFS:
        leaf_train = leafs_bin_search(train, 0, len(train) - 1)
        to_delete = clear_data(Xtrain, leaf_train)
    else:
        to_delete = clear_data(Xtrain, LEAFS_TRAIN)

    train = train.drop(to_delete, axis=1)
    test = test.drop(to_delete, axis=1)

    if COUNT_LEAFS:
        leaf_test = leafs_bin_search(test, 0, len(test))
        func_test = count_woe(test, leaf_test)
        debprint('LeafsC', f'Min samples leaf (recalculated): train: {leaf_train}, test:  {leaf_test}')
    else:
        func_test = count_woe(train, LEAFS_TRAIN)
        debprint('LeafsC', f'Min samples leaf (used from globals): train: {LEAFS_TRAIN}, test:  {LEAFS_TEST}')

    # Создание таблиц с весами вместо факторов
    debprint('FacWoe', 'Creating WOE train frame...')
    woe_train = factors_to_woe(train, func_test).sort_values('APPLICATION_ID').set_index('APPLICATION_ID')
    debprint('FacWoe', 'Creating WOE test frame...')
    woe_test = factors_to_woe(test, func_test).sort_values('APPLICATION_ID').set_index('APPLICATION_ID')
    debprint('FacWoe', 'WOE frames created')

    # Сохранение Xtrain, Ytrain
    debprint('DataSv', 'Saving Xtrain to /files/Xtrain.csv')
    Xtrain.to_csv(f'{PATH}/files/Xtrain.csv')
    debprint('DataSv', 'Saving Ytrain to /files/Ytrain.csv')
    Ytrain.to_csv(f'{PATH}/files/Ytrain.csv')

    # Сохранение Xtest, Ytest
    debprint('DataSv', 'Saving Xtest to /files/Xtest.csv')
    Xtest.to_csv(f'{PATH}/files/Xtest.csv')
    debprint('DataSv', 'Saving Ytest to /files/Ytest.csv')
    Ytest.to_csv(f'{PATH}/files/Ytest.csv')

    # Сохранение Woe train и test
    debprint('WoeSav', 'Saving Woe woe train to /files/woe_train.csv')
    woe_train.to_csv(f'{PATH}/files/woe_train.csv')
    debprint('WoeSav', 'Saving Woe test frame to /files/woe_test.csv')
    woe_test.to_csv(f'{PATH}/files/woe_test.csv')
    debprint('WoeSav', 'Woe frames saved')

    # Построение модели логической регрессии
    debprint('Models', 'Creating and training models on woe frames')
    model_test = LogisticRegression()
    model_test.fit(woe_test, Ytest.values.ravel())
    model_train = LogisticRegression()
    model_train.fit(woe_train, Ytrain.values.ravel())
    debprint('Models', 'WOE models trained')

    joblib.dump(model_train, f'{PATH}/files/model_train.pkl')
    joblib.dump(model_test, f'{PATH}/files/model_test.pkl')
    debprint('Models', 'WOE models saved')

    return woe_train, woe_test, Xtrain, Ytrain, Xtest, Ytest, model_train, model_test


def woe_to_score(data, model):
    # Функция перевода весов в баллы
    data_copy = data.copy()
    n = len(data_copy.columns)
    index = 0
    for row in data_copy.columns:
        data_copy[row] = -(data_copy[row] * model.coef_[0][index] + model.intercept_[0] / n) * COEF_A + COEF_B / n
        index += 1
    return round(data_copy.sum(axis=1))


class Defaults:
    leafs_train = 100
    leafs_test = 100
    correl = 90
    percent = 75


# Парсер аргументов
parser = argparse.ArgumentParser()
parser.add_argument('--woe', action='store_true',
                    help='Prepare and save woe frames and models')
parser.add_argument('--score', action='store_true',
                    help='Recompile score frames')
parser.add_argument('--blending', action='store_true',
                    help='Train all blending models')
parser.add_argument('--leafs', action='store_true',
                    help='Recount min leafs or use defaults (100, 100)')
parser.add_argument('--correl', nargs=1, type=int, default=Defaults.correl,
                    help='[ 90] Correlation constant')
parser.add_argument('--percent', nargs=1, type=int, default=Defaults.percent,
                    help='[ 75] What percentage of the data is train group')

args = parser.parse_args()
debprint('ArgPrs', f'Parsed args: {args}')

# --- Для отлючения ошибки ---
pd.options.mode.chained_assignment = None
# ----------------------------

# --- Путь ---
PATH = os.getcwd()
# ------------

# --- Глобальные переменные ---
COUNT_LEAFS = args.leafs

LEAFS_TRAIN = Defaults.leafs_train
LEAFS_TEST = Defaults.leafs_test

CORREL = args.correl
SPLIT_PERCENT = args.percent

BOXCOX = 300
LAMBBA = 1

COEF_A = 69 / np.log(2)
COEF_B = 444 - COEF_A * 10
# -----------------------------

# ----- ----- ----- ----- -----
# Если параметры, влияющие на WOE frames не изменены, загрузить готовые таблицы
if not args.woe and (not COUNT_LEAFS and CORREL == Defaults.correl and SPLIT_PERCENT == Defaults.percent):
    debprint('DataLd', 'Loading Xtrain and Ytrain')
    Xtrain = pd.read_csv(f'{PATH}/files/Xtrain.csv')
    Ytrain = pd.read_csv(f'{PATH}/files/Ytrain.csv').set_index('APPLICATION_ID')
    debprint('DataLd', 'Loading Xtest and Ytest')
    Xtest = pd.read_csv(f'{PATH}/files/Xtest.csv')
    Ytest = pd.read_csv(f'{PATH}/files/Ytest.csv').set_index('APPLICATION_ID')
    debprint('DataLd', 'Loading Woe train')
    woe_train = pd.read_csv(f'{PATH}/files/woe_train.csv').set_index('APPLICATION_ID')
    debprint('DataLd', 'Loading Woe test')
    woe_test = pd.read_csv(f'{PATH}/files/woe_test.csv').set_index('APPLICATION_ID')

    model_train = joblib.load(f'{PATH}/files/model_train.pkl')
    model_test = joblib.load(f'{PATH}/files/model_test.pkl')
    debprint('Models', 'Data and models loaded')
else:
    debprint('DataPr', 'Converting RawData to woe frames')
    woe_train, woe_test, Xtrain, Ytrain, Xtest, Ytest, model_train, model_test = data_prepairing()

    model_train2 = joblib.load(f'{PATH}/files/model_train.pkl')
    model_test2 = joblib.load(f'{PATH}/files/model_test.pkl')
# -----------------------------

wte = woe_test
wtr = woe_train
score_train = Ytrain.copy()
score_test = Ytest.copy()

# ----- ----- ----- ----- -----
# Для перерасчета таблицы с score
if args.score:
    # Получение баллов для каждого запроса
    debprint('ScorFr', 'Converting woe train frame to score one')
    score_train['Score'] = woe_to_score(wtr, model_train)
    debprint('ScorFr', 'Converting woe test frame to score one')
    score_test['Score'] = woe_to_score(wte, model_test)
    debprint('ScorFr', 'Saving frames...')
    joblib.dump(score_train, f'{PATH}/files/score_train.pkl')
    joblib.dump(score_test, f'{PATH}/files/score_test.pkl')
    debprint('ScorFr', 'Score frames created and saved')
else:
    # debprint('ScorFr', 'Loading score frames...')
    score_train = joblib.load(f'{PATH}/files/score_train.pkl')
    score_test = joblib.load(f'{PATH}/files/score_test.pkl')
    debprint('ScorFr', 'Score frames loaded')
# -----------------------------

Xtest = score_test['Score'].values.reshape(-1, 1)
Ytest = score_test['GB']
Xtrain = score_train['Score'].values.reshape(-1, 1)
Ytrain = score_train['GB']

Xtrain1 = Xtrain[:len(Xtrain) * 3 // 4]
Ytrain1 = Ytrain[:len(Xtrain) * 3 // 4]
Xtrain2 = Xtrain[len(Xtrain) * 3 // 4:]
Ytrain2 = Ytrain[len(Xtrain) * 3 // 4:]

NXtrain = [0] * 6
NXtest = [0] * 6

# ----- ----- ----- ----- -----
# Модели Blending'а
all_models = [('KNeighborsRegressor', KNeighborsRegressor()),
              ('Ridge', Ridge()),
              ('Lasso', Lasso()),
              ('ElasticNetCV', ElasticNetCV()),
              ('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit()),
              ('RandomForestRegressor', RandomForestRegressor())]
# -----------------------------
if args.blending:
    debprint('Blendg', 'Starting train for blending models...')
    for index, model_info in enumerate(all_models):
        model_name, blend_model = model_info
        model = blend_model
        model.fit(Xtrain1, Ytrain1)
        joblib.dump(model, f'{PATH}/files/models/{model_name}.pkl')

        NXtrain[index] = model.predict(Xtrain2)
        NXtest[index] = model.predict(Xtest)

        if index + 1 != len(all_models):
            debprint('Blendg', f'{model_name} trained, saved and predicted, next - {all_models[index + 1][0]}')

    debprint('Blendg', f'All blending models trained')
else:
    debprint('Blendg', 'Loading blending models and get predictions')
    index = 0
    for model_name, _ in all_models:
        model = joblib.load(f'{PATH}/files/models/{model_name}.pkl')

        NXtrain[index] = model.predict(Xtrain2)
        NXtest[index] = model.predict(Xtest)

        if index + 1 != len(all_models):
            debprint('Blendg', f'{model_name} loaded and predicted, next - {all_models[index + 1][0]}')
            index += 1
    debprint('Blendg', f'All blending models were loaded and made predictions')

model = Ridge()
blending_results = np.array(NXtrain)
model.fit(blending_results.T, Ytrain2)
fpr, tpr, _ = roc_curve(Ytest, model.predict(np.array(NXtest).T))
roc_auc = auc(fpr, tpr)
debprint('Report', f'Prediction accuracy: {round(roc_auc * 100, 3)}%')

# Время работы программы
if work_time is None:
    work_time = time.time() - start_time
debprint('WorkTm', f'Work time: {round(work_time, 3)} seconds')