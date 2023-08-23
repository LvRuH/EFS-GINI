from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, mutual_info_regression
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score
from sklearn.metrics import precision_score, f1_score
from skrebate import ReliefF, SURF, SURFstar
from joblib import Parallel, delayed


def load_dataset(dataset_name, percent):
    data_path = "data\{0}.csv".format(dataset_name)
    df = pd.read_csv(data_path, header=None)
    X_origin = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    k = int(X_origin.shape[1] * percent)
    X = preprocess(X_origin)
    if X.shape[1]<k:
        X=X_origin
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.4, random_state=42)
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train).astype(float)
    X_test_scaled = std.fit_transform(X_test).astype(float)
    return X_train_scaled, X_test_scaled, y_train, y_test, k

def preprocess(data):
    data_copy = data.copy()
    corr_matrix = data_copy.corr('spearman')
    threshold = 0.9  # 预设阈值
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    data_copy.drop(labels=correlated_features, axis=1, inplace=True)
    return data_copy

def gini_selection(X, y, k):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    importance = clf.feature_importances_
    idxs = np.argsort(importance)[::-1]
    return list(idxs[:k])

# def calculate_gini_index(y):
#     """
#     计算Gini Index
#     """
#     n = len(y)
#     values = np.unique(y)
#     gini = 0
#     for value in values:
#         idy = y == value
#         p=sum(idy)/n
#         gini+=p**2
#     return 1-gini
#
# def calculate_gini_gain(X, y, feature_idx):
#     """
#     计算指定特征的Gini Gain
#     """
#     n = len(y)
#     values = np.unique(X[:, feature_idx])
#     gini_gain = 0
#     for value in values:
#         idx = X[:, feature_idx] == value
#         D1 = y[idx]
#         D2 = y[~idx]
#         gini_gain += len(D1) / n * calculate_gini_index(D1) + len(D2) / n * calculate_gini_index(D2)
#     return calculate_gini_index(y) - gini_gain
#
# def gini_index_feature_selection(X, y, k):
#     """
#     使用Gini Index进行特征选择，选出k个特征
#     """
#     n_features = X.shape[1]
#     gini_gains = np.zeros(n_features)
#     for i in range(n_features):
#         gini_gains[i] = calculate_gini_gain(X, y, i)
#     idxs = np.argsort(gini_gains)[::-1]
#     return list(idxs[:k])


def mi_filter(X_train_scaled, y_train, k):
    kbest = SelectKBest(mutual_info_classif, k=k)
    kbest.fit(X_train_scaled, y_train)
    subsets = list(kbest.get_support(indices=True))
    return subsets


def f_filter(X_train_scaled, y_train, k):
    f_filter = SelectKBest(f_classif, k=k)
    f_filter.fit(X_train_scaled, y_train)
    subsets = list(f_filter.get_support(indices=True))
    return subsets


def reliefF(X_train_scaled, y_train, k):
    re = ReliefF(n_features_to_select=k, n_neighbors=20)
    re.fit(X_train_scaled, y_train)
    feature_indexes = np.argsort(re.feature_importances_)[::-1][:k]
    feature_indexes = list(feature_indexes)
    return feature_indexes


def Surf(X_train_scaled, y_train, k):
    surf = SURF(n_features_to_select=k, verbose=True)
    surf.fit(X_train_scaled, y_train)
    feature_indexes = np.argsort(surf.feature_importances_)[::-1][:k]
    feature_indexes = list(feature_indexes)
    return feature_indexes


def Surfstar(X_train_scaled, y_train, k):
    surf = SURFstar(n_features_to_select=k, verbose=True)
    surf.fit(X_train_scaled, y_train)
    feature_indexes = np.argsort(surf.feature_importances_)[::-1][:k]
    feature_indexes = list(feature_indexes)
    return feature_indexes


# def combiner(X_train_scaled,y_train,k): #5
#     set1,set2,set3,set4,set5=Parallel(n_jobs=-1)(delayed(func)(X_train_scaled,y_train,k) for func in
#                                                     [mi_filter, f_filter,reliefF,Surf,Surfstar])
#     set1,set2,set3,set4,set5=set(set1),set(set2),set(set3),set(set4),set(set5)
#     set_union=set1.union(set2).union(set3).union(set4).union(set5)
#     set_intersection=list(set1.intersection(set2).intersection(set3).intersection(set4).intersection(set5))
#     set_diff=list(set_union.difference(set_intersection))
#     set_add_index=gini_selection(X_train_scaled[:,set_diff], y_train, k-len(set_intersection))
#     set_add=[set_diff[i] for i in set_add_index]
#     return set_intersection+set_add
def combiner(X_train_scaled, y_train, k):  # 1+4
    set1 = f_filter(X_train_scaled, y_train, 4 * k)
    set1 = list(set(set1))
    X_train_scaled = X_train_scaled[:, set1]
    set2, set3, set4, set5 = Parallel(n_jobs=-1)(delayed(func)(X_train_scaled, y_train, k) for func in
                                                 [mi_filter, reliefF, Surf, Surfstar])
    set2, set3, set4, set5 = set(set2), set(set3), set(set4), set(set5)
    set_union = set2.union(set3).union(set4).union(set5)
    set_intersection = list(set2.intersection(set3).intersection(set4).intersection(set5))
    set_diff = list(set_union.difference(set_intersection))
    set_add_index = gini_selection(X_train_scaled[:, set_diff], y_train, k - len(set_intersection))
    set_add = set_intersection + [set_diff[i] for i in set_add_index]
    set_select = [set1[i] for i in set_add]

    print(set_select)
    return set_select


def ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, feature_indexes):
    # 分类器
    classifiers = [
        ('dt', DecisionTreeClassifier()),
        ('nb', GaussianNB()),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier()),
        ('rf', RandomForestClassifier(n_estimators=100))
    ]
    acc = 0
    precision = 0
    recall = 0
    clf = VotingClassifier(classifiers, voting='soft')

    for i in range(1, k + 1):
        clf.fit(X_train_scaled[:, feature_indexes], y_train)
        y_pred = clf.predict(X_test_scaled[:, feature_indexes])
        acc += accuracy_score(y_test, y_pred)
        precision += precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall += recall_score(y_test, y_pred, average='macro')

    conf = confusion_matrix(y_test, y_pred)
    acc /= k
    precision /= k
    recall /= k
    f1_score = 2 * precision * recall / (precision + recall)
    result = [round(acc, 4), round(precision, 4), round(recall, 4), round(f1_score, 4)]
    return result, conf



# def run_EFS(name):
#     X_train_scaled, X_test_scaled, y_train, y_test, k = load_dataset(name)
#     print("---------------数据集{0}上的计算结果如下---------------".format(name))
#
#     fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

# subset = mi_filter(X_train_scaled, y_train, k)
# result,f1_score= ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
# print('mi_filter ', 'acc', acc, 'precision', precision, 'recall', recall,'f1score',f1_score )
# draw_conf(fig, ax, conf, 0, 0, 'mi_filter')
#
# subset = f_filter(X_train_scaled, y_train, k)
# acc, precision, recall, conf,f1_score = ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
# print('f_filter ', 'acc', acc, 'precision', precision, 'recall', recall,'f1score',f1_score )
# draw_conf(fig, ax, conf, 0, 1, 'f_filter')
#
# subset = reliefF(X_train_scaled, y_train, k)
# acc, precision, recall, conf ,f1_score = ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
# print('reliefF ', 'acc', acc, 'precision', precision, 'recall', recall,'f1score',f1_score)
# draw_conf(fig, ax, conf, 0, 2, 'reliefF')
#
# subset = Surf(X_train_scaled, y_train, k)
# acc, precision, recall, conf ,f1_score = ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
# print('Surf ', 'acc', acc, 'precision', precision, 'recall', recall,'f1score',f1_score)
# draw_conf(fig, ax, conf, 1, 0, 'Surf')
#
# subset = Surfstar(X_train_scaled, y_train, k)
# acc, precision, recall, conf ,f1_score = ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
# print('Surfstar ', 'acc', acc, 'precision', precision, 'recall', recall,'f1score',f1_score )
# draw_conf(fig, ax, conf, 1, 1, 'Surfstar')
#
# subset = combiner(X_train_scaled, y_train, k)
# acc, precision, recall, conf ,f1_score = ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
# print('EFS-GINI ', 'acc', acc, 'precision', precision, 'recall', recall,'f1score',f1_score )
# draw_conf(fig, ax, conf, 1, 2, 'EFS-GINI')

# plt.show()
