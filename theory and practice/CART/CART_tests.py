from CART import gini, entropy, information_gain, DecisionTree, NodeCriterion
import numpy as np
import pandas as pd

balls = [1 for i in range(9)] + [0 for i in range(11)]
same_balls = [1 for i in range(9)]
balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow
balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow 

def get_toy_data():
    testing_data = np.array([[1.3, 2, 5.5],
                    [2.5, 3, 6],
                    [1.1, 4, 5.5],
                    [1.3, 2, 7],
                    [1.3, 2, 7]])

    labels = np.array([0,1,0,2,2])
    return testing_data, labels

def get_test_data():
    test_set = np.array([[1.5, 3, 7],
                    [2.7, 1, 5]])

    return test_set

def test_entropy():
    assert (entropy(balls)==0.9927744539878083), 'Balls assertion fails'
    assert (entropy(balls_left)==0.9612366047228759), 'Balls left assertion fails'
    assert (entropy(balls_right)==0.5916727785823275), 'Balls right assertion fails'
    assert (entropy([1,2,3,4,5,6])==2.584962500721156), 'Assertion fails'

def test_gini():
    assert (gini(same_balls)==0), 'Gini for the same objects is incorrect'

def test_information_gain():
    curr_uncertainty = gini(balls)
    assert (information_gain(curr_uncertainty, balls_left, balls_right)==0.10159340659340645), 'Information gain is incorrect'

def test_node_criterion():
    X, y = get_toy_data()

    cr1 = NodeCriterion(0, 1.3)
    cr2 = NodeCriterion(1, 3)

    _, cr1_Xl, cr1_Xr, cr1_yl, cr1_yr = cr1.partition(X, y)
    _, cr2_Xl, cr2_Xr, cr2_yl, cr2_yr =  cr2.partition(X, y)
    
    assert (np.array_equal(cr1_yl, [0])), 'CR1 partition is wrong'
    assert (np.array_equal(cr1_yr, [0, 1, 2, 2])), 'CR1 partition is wrong'
    assert (np.array_equal(cr2_yl, [0, 2, 2])), 'CR2 partition is wrong'
    assert (np.array_equal(cr2_yr, [1, 0])), 'CR2 partition is wrong'

def test_best_split():
    X, y = get_toy_data()

    tree = DecisionTree()
    gain, node_criterion = tree.find_best_split(X, y)

    assert (gain == 0.37333333333333324), 'Best information gain is incorect'
    assert (node_criterion.value == 7), 'Best node criterion value is incorrect'
    assert (node_criterion.feature_idx == 2), 'Best node criterion column name is incorrect'

def build_tree():
    toy_set, labels = get_toy_data()
    tree = DecisionTree()
    tree.fit(toy_set, labels)
    return tree

def test_tree():
    tree = build_tree()
    tree.print_tree()

    # root node
    assert (tree.root_node.node_criterion.value == 7), 'Best node criterion value is incorrect'
    assert (tree.root_node.node_criterion.feature_idx == 2), 'Best node criterion column name is incorrect'

    # left node
    assert (tree.root_node.left.node_criterion.value == 6), 'Best node criterion value is incorrect'
    assert (tree.root_node.left.node_criterion.feature_idx == 2), 'Best node criterion column name is incorrect'

    # right node
    assert (np.array_equal(tree.root_node.right.labels, [2, 2])), 'Right subtree is incorrect'

def test_tree_with_max_depth():
    toy_set, labels = get_toy_data()
    tree = DecisionTree(max_depth=1)
    tree.fit(toy_set, labels)
    tree.print_tree()

    # root node
    assert (tree.root_node.node_criterion.value == 7), 'Best node criterion value is incorrect'
    assert (tree.root_node.node_criterion.feature_idx == 2), 'Best node criterion column name is incorrect'

    # left node
    assert (np.array_equal(tree.root_node.left.labels, [0, 1, 0])), 'Left subtree is incorrect'

    # right node
    assert (np.array_equal(tree.root_node.right.labels, [2, 2])), 'Right subtree is incorrect'


def test_tree_with_min_split():
    toy_set, labels = get_toy_data()
    tree = DecisionTree(min_samples_split=3)
    tree.fit(toy_set, labels)
    tree.print_tree()
    
    # root node
    assert (tree.root_node.node_criterion.value == 7), 'Best node criterion value is incorrect'
    assert (tree.root_node.node_criterion.feature_idx == 2), 'Best node criterion column name is incorrect'

    # left node
    assert (np.array_equal(tree.root_node.left.labels, [0, 1, 0])), 'Left subtree is incorrect'

    # right node
    assert (np.array_equal(tree.root_node.right.labels, [2, 2])), 'Right subtree is incorrect'

def test_prediction():
    tree = build_tree()
    X_test = get_test_data()
    predictions = tree.predict(X_test)
    assert (np.array_equal(predictions.tolist(),[2,0])), 'Predicted values are wrong'
    print(predictions)

if __name__ == '__main__':
    test_entropy()
    test_gini()
    test_information_gain()
    test_node_criterion()
    test_best_split()
    test_tree()
    print("-----")
    test_tree_with_max_depth()
    print("-----")
    test_tree_with_min_split()
    test_prediction()
    print('Tests passed')