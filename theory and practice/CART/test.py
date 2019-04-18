from CART import DecisionTree
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_digits


RANDOM_STATE = 17
digits = load_digits()
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=RANDOM_STATE)

tree = DecisionTree()

tree_params = {'max_depth': range(1,2),
               'criterion': ['gini']}#, 'entropy']}

tree_grid = GridSearchCV(tree, tree_params,
                          n_jobs=1, verbose=True, scoring='accuracy')

tree_grid.fit(X_train, Y_train)