import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns


def impute_missing_values(data, column_to_impute, missing_data_mask):
    df = data.copy()
    
    # Get probabilities of each value existing in column 
    distribution = df[~missing_data_mask][column_to_impute].value_counts(normalize=True)
    
    # Get size of missing data
    missing_data_size = df[missing_data_mask].shape[0]
    
    # Impute missing values
    df.loc[missing_data_mask, column_to_impute] = np.random.choice(distribution.index,
                                                                   size=missing_data_size,
                                                                   p=distribution.values)
    return df[column_to_impute]


def recategorize_variable(data, column, categories_to_keep=2):
    df = data.copy()
    
    top_categories = df[column].value_counts().iloc[:categories_to_keep].index
    
    other_vals_mask = df[column].isin(top_categories) == False
    
    df[column] = np.where(other_vals_mask, 'Other', df[column])
    
    return df[column]


def plot_corr_heatmap(data, method='pearson', size=(6, 6)):
    
    # Get Pearson correlation coefficients 
    corr_matrix = np.round(data.corr(method=method), 2)
    
    # Get triangle mask, to cover upper half of the matrix
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Plot heatmap
    plt.figure(figsize=size)
    plt.title(f'{method.title()} correlation heatmap')
    sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1,
                annot=True, cmap="RdBu_r", mask=mask)

    
def get_thresholds(y_true, y_pred_proba):
    # thresholds = [0.0001, 0.001, 0.01, 0.1, 0.4, 0.8, 0.9, 0.99, 0.999, 0.9999]
    thresholds = np.arange(0.01, 1, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    financial_scores = []
    auc_scores = []

    for t in thresholds:
        y_pred = (y_pred_proba > t).astype(int)
        
        f1_scores.append(f1_score(y_true, y_pred))
        precision_scores.append(accuracy_score(y_true, y_pred))
        recall_scores.append(recall_score(y_true, y_pred))
        financial_scores.append(financial_score(y_true, y_pred))
        auc_scores.append(roc_auc_score(y_true, y_pred))
        
    return pd.DataFrame({'thresholds': thresholds,
                         'F1': f1_scores,
                         'precision (PPV)': precision_scores,
                         'recall (TPR)': recall_scores,
                         'financial score': financial_scores,
                         'roc_auc_score': auc_scores})


def financial_score(y_true, y_pred):
    
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    prediction_cost = y_pred.sum() * 100
    prediction_gain = (y_true & y_pred).sum() * 1000
    
    return prediction_gain - prediction_cost


def assess_model(model_name, y_true, y_pred_proba,
                 metric_to_optimise_treshold='F1',
                 class_balancing=None):
    
    # Create empty DF for model assessment
    asmnt_df = pd.DataFrame({'model': model_name}, index=[0])
    
    # Find threshold with highest F1
    thresholds_df = get_thresholds(y_true, y_pred_proba)
    thresholds_df.sort_values(by=metric_to_optimise_treshold,
                              ascending=False, inplace=True)
    best_threshold = thresholds_df.head(1)
    best_threshold.reset_index(inplace=True)
    
    # Concatenate tables
    asmnt_df = pd.concat([asmnt_df, best_threshold], axis=1)
    asmnt_df.drop(columns=['index'], inplace=True)
    asmnt_df['class_balancing'] = class_balancing
    
#     # Get new vector of predictions, according to best threshold found
#     y_pred = np.array(y_pred_proba) > best_threshold['thresholds'][0]
    
#     # Get financial score
#     asmnt_df['financial_score'] = financial_score(y_true, y_pred)
    
    return asmnt_df                                       


def cv_undersampling(X, y, model, size=1, cv=StratifiedKFold(5)):
    # size - stosunek liczby obserwacji klasy 1 do klasy 0
    y_pred_proba = []
    y_true = []
    
    for train_idx, test_idx in cv.split(X, y):
        
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # pobierz indeksy klasy 0 ze zbioru treningowego
        class_0_indexes = np.where(y_train==0)[0]
        
        # wyznacz liczność klasy 1 w zbiorze treningowym
        class_1_amount = np.sum((y_train==1).astype(int))
        
        # wyznacz liczbę wymaganych obserwacji klasy 0 w oparciu o współczynnik size
        class_0_indexes_number = int(size * class_1_amount)
        
        # wylosuj (class_0_indexes_number - elementowy) podzbiór indeksów klasy 0
        # ze wszystkich dostępnych w zbiorze treningowym bez zwracania  
        idx0 = np.random.choice(class_0_indexes, class_0_indexes_number, replace=False)
        
        # pobierz wszystkie indeksy klasy 1
        idx1 = np.where(y_train==1)[0]
        
        # połącz indeksy klasy 0 i 1 we wspólny wektor
        idx_all = np.r_[idx0, idx1]
        
        # wyznacz podzbiór obserwacji z X_train i y_train w oparciu o uzyskane indeksy
        X_train_sub = X_train[idx_all, :]
        y_train_sub = y_train[idx_all]
        
        # zadbaj o nienadpisywanie się modelu
        model_tmp = deepcopy(model)
        
        # naucz model i wykonaj predykcję
        model_tmp.fit(X_train_sub, y_train_sub)
        y_prob = model_tmp.predict_proba(X_test)[:, 1].tolist()
        
        y_pred_proba.extend(y_prob)
        y_true.extend(y_test.tolist())
        
    return np.array(y_pred_proba), np.array(y_true)


def report_top_feat(data_set, features, top = 15):
    indices = np.argsort(features)[::-1]
    for f in range(top):
        print("%d. %s (%f)" % (f + 1, data_set.columns[indices[f]], features[indices[f]]))
    
    indices=indices[:top]
    plt.figure(figsize=[6, 4])
    plt.title("Top features")
    plt.bar(range(top), features[indices], color="r", align="center")
    plt.xticks(range(top), data_set.columns[indices], rotation=90)
    plt.xlim([-1, top])
    plt.show()
    print("Mean Feature Importance %.6f" %np.mean(features), '\n')


def cross_validate_oversampling(X, Y, model, size=1, cv=StratifiedKFold(5,random_state=1)):
    
    preds = []
    true_labels = []
    
    for train_index, test_index in cv.split(X,Y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    

        ind1 = np.resize(np.where(y_train==1)[0], int(size*np.sum(y_train==1)))
        ind0 = np.where(y_train==0)[0]

        ind_final = np.r_[ind1, ind0]
        X_train_subsample = X_train.iloc[ind_final]
        y_train_subsample = y_train.iloc[ind_final]
        
        clf = deepcopy(model)
        clf.fit(X_train_subsample, y_train_subsample)
               
        preds.extend(clf.predict_proba(X_test)[:,1])
        true_labels.extend(y_test)

    return preds, true_labels


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))
    

# transformer = Pipeline([
#     ('features', FeatureUnion(n_jobs=1, transformer_list=[
        
#         # Part 1
#         ('boolean', Pipeline([
#             ('selector', TypeSelector('bool')),
#         ])),  # booleans close
        
#         ('numericals', Pipeline([
#             ('selector', TypeSelector(np.number)),
#             ('scaler', StandardScaler()),
#         ])),  # numericals close
        
#         # Part 2
#         ('categoricals', Pipeline([
#             ('selector', TypeSelector('category')),
#             ('labeler', StringIndexer()),
#             ('encoder', OneHotEncoder(handle_unknown='ignore')),
#         ]))  # categoricals close
#     ])),  # features close
# ])  # pipeline close