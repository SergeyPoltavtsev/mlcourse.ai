import numpy as np
import pandas as pd
from enum import Enum
import time
from sklearn.base import BaseEstimator


def get_proportions(y):
    '''
        returns proportion of each unique element.
    '''
    return np.unique(y, return_counts=True)[1]/(len(y)*1.0)


def entropy(y):
    '''
        returns entropy for a given sequence of classes.
    '''
    if (len(y) == 0):
        return 0
    portions = get_proportions(y)
    #entropy = -np.sum(portions*np.log2(portions))
    entropy = -np.dot(portions, np.log2(portions))
    return entropy


def gini(y):
    '''
        returns gini impurity for a given sequence of classes.    
    '''
    if (len(y) == 0):
        return 0
    portions = get_proportions(y)
    #gini_impurity = 1 - np.sum(np.square(portions))
    gini_impurity = 1 - np.dot(portions, portions)
    return gini_impurity


def variance(y):
    '''
        Variance (mean quadratic deviation from average) 
    '''
    if (len(y) == 0):
        return 0
    # return np.sum(np.square(y-np.mean(y)))/(len(y)*1.0)
    return np.var(y)


def mad_median(y):
    '''
        Mean deviation from the median 
    '''
    if (len(y) == 0):
        return 0
    # return np.sum(np.abs(y-np.median(y)))/(len(y)*1.0)
    return np.mean(np.abs(y - np.median(y)))


def information_gain(current_uncertainty, left, right, criterion=gini):
    '''
        Calculates information gain
    '''
    proportion = len(left) * 1.0 / (len(left)+len(right))
    return current_uncertainty - proportion * criterion(left) - (1.0 - proportion) * criterion(right)


criterions = {'gini': gini,
              'entropy': entropy,
              'variance': variance,
              'mad_median': mad_median}


class TreeType(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2


class TreeLeaf():
    def __init__(self, y, tree_type):
        self.labels = y
        if tree_type == TreeType.CLASSIFICATION:
            self.predicted_value = np.bincount(self.labels).argmax() #self.labels.value_counts().index[0]
        else:
            self.predicted_value = self.labels.mean()
        self.unique = np.unique(self.labels)
        self.proportions = get_proportions(self.labels)
        self.class_proportions = dict(zip(self.unique, self.proportions))


class TreeNode():
    def __init__(self, node_criterion, left=None, right=None):
        self.node_criterion = node_criterion
        self.left = left
        self.right = right


class NodeCriterion():
    '''
    A critetion is used to partition a dataset.

    This class stores a column name and a column value. 
    The partition methods is used to split a dataset into two   
    based on feature value stored in the criterion.
    '''

    def __init__(self, feature_idx, value):
        self.feature_idx = feature_idx
        self.value = value

    def partition(self, X, y = None):
        '''
        Partitions the set based on the critetion. 
        For a training set partitions both features and labels.
        For a test set partitions only the features.
        '''
        mask = X[:, self.feature_idx] < self.value

        Xl = X[mask, :]
        Xr = X[~mask, :]

        yl = None
        yr = None
        if y is not None:
            yl = y[mask]
            yr = y[~mask]

        return mask, Xl, Xr, yl, yr

    def __repr__(self):
        return "Node criterion: %s < %s" % (self.feature_idx, str(self.value))


class DecisionTree(BaseEstimator):
    def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='gini', debug=False):
        # To make the DecisionTree class work with the GridSearchCV and others when combined in a Pipeline.
        params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'criterion': criterion,
            'debug': debug
        }
        self.set_params(**params)

        if self.debug:
            print("\nDecisionTree params:")
            print("max_depth = {}, min_samples_split = {}, criterion = {}\n".format(
                max_depth, min_samples_split, criterion))

    def set_params(self, **params):
        super().set_params(**params)

        self._critetion_function = criterions[self.criterion]
        self.tree_type = TreeType.CLASSIFICATION if self.criterion in [
            'gini', 'entropy'] else TreeType.REGRESSION
        self.root_node = None

        return self

    def find_best_split(self, X, y):
        '''
        Find the best split for a dataset iteration over the features and feature values
        '''
        best_gain = 0
        best_node_criterion = None
        current_uncertainty = self._critetion_function(y)
        _, n_features = X.shape

        # exclude the last column (targets)
        for feature_idx in range(n_features):
            for value in np.unique(X[:, feature_idx]):
                node_criterion = NodeCriterion(feature_idx, value)
                _, Xl, Xr, yl, yr = node_criterion.partition(X, y)
                
                if Xl.shape[0] == 0 or Xr.shape[0] == 0:
                    continue
                gain = information_gain(
                    current_uncertainty, yl, yr, criterion=self._critetion_function)

                if gain >= best_gain:
                    best_gain = gain
                    best_node_criterion = node_criterion

        return best_gain, best_node_criterion

    def fit(self, X, y):
        self.root_node = self._build(X, y)

    def _build(self, X, y, node_depth=0):

        if len(np.unique(y)) == 1:
            return TreeLeaf(y, self.tree_type)

        if node_depth == self.max_depth:
            return TreeLeaf(y, self.tree_type)

        if X.shape[0] <= self.min_samples_split:
            return TreeLeaf(y, self.tree_type)

        # find the best split
        gain, node_criterion = self.find_best_split(X, y)

        if self.debug:
            print('Best split: %s gives information gain of %s at depth %s' %
                  (node_criterion, gain, node_depth))

        if gain == 0:
            return TreeLeaf(y, self.tree_type)

        _, Xl, Xr, yl, yr = node_criterion.partition(X, y)
        Xl_subtree = self._build(Xl, yl, node_depth+1)
        Xr_subtree = self._build(Xr, yr, node_depth+1)

        return TreeNode(node_criterion, Xl_subtree, Xr_subtree)

    def print_tree(self):
        if(self.root_node == None):
            return
        self._print_node(self.root_node)

    def _print_node(self, node, spacing=''):

        if(isinstance(node, TreeLeaf)):
            print(spacing + 'Predict: ', node.class_proportions)
            return

        print(spacing + str(node.node_criterion))

        print(spacing + '--> True:')
        self._print_node(node.left, spacing + "  ")

        print(spacing + '--> False:')
        self._print_node(node.right, spacing + "  ")

    def predict(self, X):
        return np.array([self._predict_object(x) for x in X])

    def _predict_object(self, x):
        node = self.root_node

        while not isinstance(node, TreeLeaf):
            if x[node.node_criterion.feature_idx] < node.node_criterion.value:
                node = node.left
            else:
                node = node.right
        
        return node.predicted_value