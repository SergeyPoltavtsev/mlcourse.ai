import numpy as np
import pandas as pd
from enum import Enum
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
    entropy = -np.sum(portions*np.log2(portions))
    return entropy


def gini(y):
    '''
        returns gini impurity for a given sequence of classes.    
    '''
    if (len(y) == 0):
        return 0
    portions = get_proportions(y)
    gini_impurity = 1 - np.sum(np.square(portions))
    return gini_impurity


def variance(y):
    '''
        Variance (mean quadratic deviation from average) 
    '''
    if (len(y) == 0):
        return 0
    # return np.var(y)
    return np.sum(np.square(y-np.mean(y)))/(len(y)*1.0)


def mad_median(y):
    '''
        Mean deviation from the median 
    '''
    if (len(y) == 0):
        return 0
    # return np.mean(np.abs(y - np.median(y)))
    return np.sum(np.abs(y-np.median(y)))/(len(y)*1.0)


def information_gain(current_uncertainty, left, right, criterion=gini):
    proportion = len(left) * 1.0 / (len(left)+len(right))
    return current_uncertainty - proportion * criterion(left) - (1.0 - proportion) * criterion(right)


def is_numeric(value):
    '''
    Tests if the value is of numeric type.
    '''
    if(isinstance(value, pd.Series)):
        return np.issubdtype(value.dtype, np.number)
    else:
        return isinstance(value, int) or isinstance(value, float)


criterions = {'gini': gini,
              'entropy': entropy,
              'variance': variance,
              'mad_median': mad_median}


def listLabelsToSeries(Y):
    if isinstance(Y, pd.Series):
        return Y
    return pd.Series(Y, name='target')


def arrayToDataFrame(X):
    '''
    Converts numpy nd array to pandas dataframe
    '''
    if not isinstance(X, (np.ndarray, list)):
        return X

    columns = ['column' + str(i) for i in range(np.shape(X)[1])]
    return pd.DataFrame(data=X, columns=columns)


class TreeType(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2


class TreeLeaf():
    def __init__(self, X):
        self.labels = X.iloc[:, -1]
        self.most_frequent = self.labels.value_counts().index[0]
        self.mean = self.labels.mean()
        self.unique = np.unique(self.labels)
        self.proportions = get_proportions(self.labels)
        self.class_proportions = dict(zip(self.unique, self.proportions))


class TreeNode():
    def __init__(self, node_criterion, left=None, right=None):
        self.node_critetion = node_criterion
        self.left = left
        self.right = right


class NodeCriterion():
    '''
    A critetion is used to partition a dataset.

    This class stores a column name and a column value. 
    The partition methods is used to split a dataset into two   
    based on feature value stored in the criterion.
    '''

    def __init__(self, column_name, value):
        self.column_name = column_name
        self.value = value

    def partition(self, X):
        # Partitions a dataset into two subsets based on a criterion\
        if is_numeric(X[self.column_name]):
            Xl = X[X[self.column_name] >= self.value]
            Xr = X[X[self.column_name] < self.value]
        else:
            Xl = X[X[self.column_name] == self.value]
            Xr = X[X[self.column_name] != self.value]
        return Xl, Xr

    def __repr__(self):
        operator = "=="
        if is_numeric(self.value):
            operator = ">="
        return "Criterion: %s %s %s" % (self.column_name, operator, str(self.value))


class DecisionTree(BaseEstimator):

    def __init__(self, max_depth=np.inf, min_sample_split=2, criterion='gini', debug=False):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.tree_type = TreeType.CLASSIFICATION if criterion in ['gini', 'entropy'] else TreeType.REGRESSION
        # needed for the get_params function to work properly
        self.criterion=criterion
        self.criterion_function = criterions[criterion]
        self.debug = debug
        self.root_node = None

    def find_best_split(self, X):
        '''
        Find the best split for a dataset iteration over the features and feature values
        '''
        best_gain = 0
        best_node_criterion = None
        current_uncertainty = self.criterion_function(X.iloc[:, -1])

        # exclude the last column (targets)
        for feature in X.columns[:-1]:
            for value in X[feature].unique():
                node_criterion = NodeCriterion(feature, value)
                Xl, Xr = node_criterion.partition(X)

                if Xl.shape[0] == 0 or Xr.shape[0] == 0:
                    continue

                gain = information_gain(
                    current_uncertainty, Xl.iloc[:, -1], Xr.iloc[:, -1], criterion=self.criterion_function)

                if gain >= best_gain:
                    best_gain = gain
                    best_node_criterion = node_criterion

        return best_gain, best_node_criterion

    def fit(self, X, Y):
        X = arrayToDataFrame(X)
        Y = listLabelsToSeries(Y)
        traininig_set = pd.concat([X, Y], axis=1)
        self.root_node = self._build(traininig_set, 0)

    def _build(self, X, node_depth):
        gain, node_criterion = self.find_best_split(X)

        if self.debug:
            print('%s gives information gain of %s' % (node_criterion, gain))

        if node_depth == self.max_depth:
            return TreeLeaf(X)

        if X.shape[0] <= self.min_sample_split:
            return TreeLeaf(X)

        if gain == 0:
            return TreeLeaf(X)

        Xl, Xr = node_criterion.partition(X)
        Xl_subtree = self._build(Xl, node_depth+1)
        Xr_subtree = self._build(Xr, node_depth+1)

        return TreeNode(node_criterion, Xl_subtree, Xr_subtree)

    def print_tree(self):
        if(self.root_node == None):
            return
        self._print_node(self.root_node)

    def _print_node(self, node, spacing=''):

        if(isinstance(node, TreeLeaf)):
            print(spacing + 'Predict: ', node.class_proportions)
            return

        print(spacing + str(node.node_critetion))

        print(spacing + '--> True:')
        self._print_node(node.left, spacing + "  ")

        print(spacing + '--> False:')
        self._print_node(node.right, spacing + "  ")

    def predict(self, X):
        X = arrayToDataFrame(X)
        predictions = self._traverse_node(self.root_node, X)
        assert X.shape[0] == predictions.shape[0], 'The amount of predictions does not corresponds to the number of examples'
        return predictions.sort_index()

    def _traverse_node(self, node, X):
        if isinstance(node, TreeLeaf):
            if self.tree_type == TreeType.CLASSIFICATION:
                leaf_value = node.most_frequent
            else:
                leaf_value = node.mean
            return pd.Series(data=[leaf_value] * X.shape[0], index=X.index)

        Xl, Xr = node.node_critetion.partition(X)

        left_labels = pd.Series()
        right_labels = pd.Series()

        if Xl.shape[0] != 0:
            left_labels = self._traverse_node(node.left, Xl)

        if Xr.shape[0] != 0:
            right_labels = self._traverse_node(node.right, Xr)

        return left_labels.append(right_labels)
