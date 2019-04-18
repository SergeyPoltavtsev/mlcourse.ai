from CART import gini, entropy, information_gain, DecisionTree, NodeCriterion, arrayToDataFrame, listLabelsToSeries
import numpy as np
import pandas as pd

balls = [1 for i in range(9)] + [0 for i in range(11)]
same_balls = [1 for i in range(9)]
balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow
balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow 

def get_toy_data():
    testing_data = [['Green', 2, 'monitor'],
                    ['Orange', 3, 'wine'],
                    ['Yellow', 4, 'monitor'],
                    ['Green', 2, 'mac'],
                    ['Green', 2, 'mac']]

    labels = [0,1,0,2,2]
    return testing_data, labels

def get_set():
    toy_set, labels = get_toy_data()
    X = arrayToDataFrame(toy_set)
    Y = listLabelsToSeries(labels)
    traininig_set = pd.concat([X, Y], axis=1)
    return traininig_set

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

def test_array_to_dataframe():
    df = get_set()
    assert (df.columns.to_list()==['column0','column1','column2', 'target']), 'Column names are not matching'

def test_node_criterion():
    toy_df = get_set()

    cr1 = NodeCriterion(toy_df.columns[0], "Green")
    cr2 = NodeCriterion(toy_df.columns[1], 3)

    cr1_left, cr1_right = cr1.partition(toy_df)
    cr2_left, cr2_right = cr2.partition(toy_df)
    
    assert (np.array_equal(cr1_left.index.values, [0,3,4])), 'CR1 partition is wrong'
    assert (np.array_equal(cr1_right.index.values, [1,2])), 'CR1 partition is wrong'
    assert (np.array_equal(cr2_left.index.values, [1,2])), 'CR2 partition is wrong'
    assert (np.array_equal(cr2_right.index.values, [0,3,4])), 'CR2 partition is wrong'

def test_best_split():
    toy_df = get_set()

    tree = DecisionTree()
    gain, node_criterion = tree.find_best_split(toy_df)

    assert (gain == 0.37333333333333324), 'Best information gain is incorect'
    assert (node_criterion.value == 'monitor'), 'Best node criterion value is incorrect'
    assert (node_criterion.column_name == 'column2'), 'Best node criterion column name is incorrect'

def build_tree():
    toy_set, labels = get_toy_data()
    tree = DecisionTree()
    tree.fit(toy_set, labels)
    return tree

def test_tree():
    tree = build_tree()
    tree.print_tree()
    # TODO: write assert statement

def test_tree_with_max_depth():
    toy_set, labels = get_toy_data()
    tree = DecisionTree(max_depth=1)
    tree.fit(toy_set, labels)
    tree.print_tree()
    # TODO: write assert statement


def test_tree_with_min_split():
    toy_set, labels = get_toy_data()
    tree = DecisionTree(min_sample_split=3)
    tree.fit(toy_set, labels)
    tree.print_tree()
    # TODO: write assert statement

def test_prediction():
    tree = build_tree()
    columns = ['column0', 'column1', 'column2', 'targets']
    X_test = pd.DataFrame(data = [['Orange', 3, 'wine', 1], ['Green', 2, 'monitor', 0]], columns=columns)
    predictions = tree.predict(X_test)
    assert (np.array_equal(predictions.tolist(),[1,0])), 'Predicted values are wrong'
    print(predictions)

if __name__ == '__main__':
    test_entropy()
    test_gini()
    test_information_gain()
    test_node_criterion()
    test_array_to_dataframe()
    test_best_split()
    test_tree()
    print("-----")
    test_tree_with_max_depth()
    print("-----")
    test_tree_with_min_split()
    test_prediction()
    print('Tests passed')