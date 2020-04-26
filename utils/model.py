import glob
import os
import random
import re
import string
from operator import itemgetter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier

from utils.plot import plot_cm,graphviz

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model(path):
    """

    :param path: weight path
    :return: load model based on the path
    """
    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def dump_model(model, name):
    model_name = WEIGHTS_DIR + name + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def split_meth(features, labels, args):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

    model = DecisionTreeClassifier(criterion=args.criterion)
    model.fit(features_train, labels_train)

    preds = model.predict(features_test)

    cm = confusion_matrix(labels_test, preds)
    score = accuracy_score(labels_test, preds)

    plot_cm(cm, f'cm-accuracy:{score:.2f}DecTree-split-{args.criterion}')

    graphviz(model, list(features.columns), list(labels.unique()),f'DecTree-split-{args.criterion}-graph')

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        dump_model(model, 'DecTree-split-')

    return model


def cv_meth(features, labels, args):
    model = DecisionTreeClassifier(criterion=args.criterion)

    cv_results = cross_validate(model, features, labels, cv=10, return_estimator=True)

    estimator_test_score = zip(list(cv_results['estimator']), cv_results['test_score'])
    model, score = max(estimator_test_score, key=itemgetter(1))

    preds = model.predict(features)

    cm = confusion_matrix(labels, preds)

    plot_cm(cm, f'cm-accuracy:{score:.2f}DecTree-cv-{args.criterion}')

    graphviz(model, list(features.columns), list(labels.unique()),f'DecTree-cv-{args.criterion}-graph')

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        dump_model(model, 'DecTree-cv-')

    return model


training_methods = {
    'split': split_meth,
    'cv': cv_meth,
}


def train_model(features, labels, args):
    if args.gs:
        param_grid = {'max_depth': np.arange(1, 10),
                      'criterion': ['gini', 'entropy'],
                      'max_features':['sqrt','log2',None],
                      }
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

        gs = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid,cv=10)
        gs.fit(features_train, labels_train)

        model = gs.best_estimator_
        preds = model.predict(features_test)

        cm = confusion_matrix(labels_test, preds)
        score = accuracy_score(labels_test, preds)

        best_params = re.sub("[{}' ,()]", '', str(gs.best_params_))

        plot_cm(cm, f'cm-accuracy:{score:.2f}{best_params}')

        graphviz(model,list(features.columns), list(labels.unique()),'DecTree-gs-graph')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model, 'DecTree-gs-')

        return model
    else:
        return training_methods[args.method](features, labels, args)
