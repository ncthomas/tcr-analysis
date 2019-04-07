#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import warnings

import pandas
import numpy
import matplotlib.pyplot
import seaborn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold, train_test_split
import lime
import lime.lime_tabular

import functions.io
import config

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


def get_sample_status(x):
    """Convenience function to create column with sample status"""

    status_int = int(x.split("_")[2])
    
    if status_int <= 3:
        status = 'CTRL'
    elif 3 < status_int <= 6:
        status = 'DIABETES'

    return status

def get_sample_chain(x):
    """Convenience function to create column with sample chain (alpha or beta)"""

    description = x.split("_")
    if 'alpha' in description:
        chain = 'alpha'
    elif 'beta' in description:
        chain = 'beta'

    return chain


def apply_color(x):
    """Convenience function to create a column of strings of colors"""

    if x is np.nan:
        col = 'black'
    else:
        col = 'red'

    return col


# In[ ]:


def train_model(X_train, y_train, n=1):
    
    corpus = [seq for seq in X_train['seq'].tolist()]
    text_model = TfidfVectorizer(analyzer='char', lowercase=False, ngram_range=(n, n))
    matrix_train = text_model.fit_transform(corpus).todense()

    model = LogisticRegression(random_state=123)
    cv = KFold(n_splits=2, shuffle=True)
    gs = GridSearchCV(estimator=model, param_grid={"C": [1, 10, 100]}, cv=cv)
    gs.fit(matrix_train, y_train)
    best_pred_model = gs.best_estimator_

    return best_pred_model, text_model


def get_tfidf_matrix(X, text_model):
    
    corpus = [seq for seq in X['seq'].tolist()]
    matrix = text_model.transform(corpus).todense()

    return matrix


def test_model(best_pred_model, text_model, X_test, y_test, n=1):
    
    corpus = [seq for seq in X_test['seq'].tolist()]
    matrix_test = text_model.transform(corpus).todense()

    predictions = cross_val_predict(best_pred_model, matrix_test, y_test, cv=3)
    score = accuracy_score(y_test, predictions)
    score = numpy.round(100 * score, 2)

    return score

def get_feature_names(X, text_model, n=1):
    
    feature_names = [0] * len(text_model.vocabulary_.items())
    for key, value in text_model.vocabulary_.items():
        feature_names[value] = key

    return feature_names


# In[ ]:


data = functions.io.read_all_samples(config.Config.DATA_DIRECTORY)

data['status'] = data['sample'].apply(lambda x: get_sample_status(x))
samples = functions.io.get_sample_names(data)

# Build model

X = data[['seq']]
y = label_binarize(data['status'], classes=['CTRL', 'DIABETES']).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
X_train = X_train.reset_index().drop('index', axis=1)
X_test = X_test.reset_index().drop('index', axis=1)


# In[ ]:


n_grams = config.Config.N_GRAMS
best_model, text_model = train_model(X_train, y_train, n=n_grams)
score = test_model(best_model, text_model, X_test, y_test, n=n_grams)
print("Accuracy =", score, "%")

matrix_test = get_tfidf_matrix(X_test, text_model)
feature_names = get_feature_names(X_train, text_model, n_grams)
arr = numpy.array(matrix_test)


# In[ ]:


explainer = lime.lime_tabular.LimeTabularExplainer(arr, feature_names=feature_names, class_names=['CTRL', 'DIABETES'])


# In[ ]:


# choose a cdr3 from the test set ...
choose_sample = 2
X_test.iloc[choose_sample]['seq']
exp = explainer.explain_instance(arr[choose_sample], best_model.predict_proba, num_features=len(feature_names))
exp.show_in_notebook(show_table=True)


# In[ ]:


# or create your own cdr3 sequence ...
seq = 'CASSEDTQYF'
df_seq = pandas.DataFrame({'seq': [seq]})
matrix_seq = numpy.array(get_tfidf_matrix(df_seq, text_model))[0]

exp = explainer.explain_instance(matrix_seq, best_model.predict_proba, num_features=len(feature_names))
exp.show_in_notebook(show_table=True)


# In[ ]:




