"""
Tests simple classifiers.
"""
import numpy as np
from scipy import stats

from mysklearn import myutils
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyNaiveBayesClassifier, \
    MyDecisionTreeClassifier, \
    MyRandomForestClassifier

# pylint: skip-file

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    """
        Tests MySimpleLinearRegressionClassifier's fit method.
    """
    # y = 2x + some noise
    np.random.seed(0) # for reproducibility
    X_train = [[val] for val in range(100)] # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train] # 1D
    lin_reg = MySimpleLinearRegressionClassifier(discretizer=myutils.discretize_high_low)
    lin_reg.fit(X_train, y_train)
    # check with "desk calculation"
    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    # assert against these solution values
    assert np.isclose(lin_reg.regressor.slope, slope_solution)
    assert np.isclose(lin_reg.regressor.intercept, intercept_solution)
    # ...and check with scipy
    X_vals = [val[0] for val in X_train]
    m, b, _, _, _ = stats.linregress(X_vals, y_train)
    assert np.isclose(lin_reg.regressor.slope, m)
    assert np.isclose(lin_reg.regressor.intercept, b)

    # y = x/3 +- some noise
    X_train_2 = [[val] for val in range(100)] # 2D
    y_train_2 = [row[0] * 1/3 + np.random.normal(-15, 15) for row in X_train_2] # 1D
    lin_reg_2 = MySimpleLinearRegressionClassifier(discretizer=myutils.discretize_high_low)
    lin_reg_2.fit(X_train_2, y_train_2)
    # check with "desk calculation"
    slope_solution_2 = 0.2457473739159945
    intercept_solution_2 = -9.4343028282366898
    # assert against these solution values
    assert np.isclose(lin_reg_2.regressor.slope, slope_solution_2)
    assert np.isclose(lin_reg_2.regressor.intercept, intercept_solution_2)

def test_simple_linear_regression_classifier_predict():
    """
        Tests MySimpleLinearRegressionClassifier's predict method.
    """
    # y = 2x + some noise
    np.random.seed(0) # for reproducibility
    X_train = [[val] for val in range(100)] # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train] # 1D
    X_test = [[150], [-150], [0], [50], [1000]] # y = [300, -300, 0, 100, 2000]
    lin_reg = MySimpleLinearRegressionClassifier(discretizer=myutils.discretize_high_low)
    lin_reg.fit(X_train, y_train)
    y_predicted = lin_reg.predict(X_test)
    y_expected = ["high", "low", "low", "high", "high"]
    assert y_predicted == y_expected

    # y = x/3 +- some noise
    X_train_2 = [[val] for val in range(100)] # 2D
    y_train_2 = [row[0] * 1/3 + np.random.normal(-5, 5) for row in X_train_2] # 1D
    X_test_2 = [[360], [-300], [0], [3000]] # y = [120, -100, 0, 1000]
    lin_reg_2 = MySimpleLinearRegressionClassifier(discretizer=myutils.discretize_high_low)
    lin_reg_2.fit(X_train_2, y_train_2)
    y_predicted_2 = lin_reg_2.predict(X_test_2)
    y_expected_2 = ["high", "low", "low", "high"]
    assert y_predicted_2 == y_expected_2

def test_kneighbors_classifier_kneighbors():
    """
        MyKNeighborsClassifier's kneighbors method.
    """
    # Dataset 1
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    X_test = [[0.33, 1]]
    kneighbors = MyKNeighborsClassifier()
    kneighbors.fit(X_train, y_train)
    distances, indices = kneighbors.kneighbors(X_test)
    expected_distances = [[0.6699999999999999, 1.0, 1.0530432089900206]]
    expected_indices = [[0, 2, 3]]
    assert np.allclose(distances, expected_distances)
    assert np.allclose(indices, expected_indices)

    # Dataset 2
    # assume normalized
    X_train_2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    test_instance = [[2, 3]]
    kneighbor_2 = MyKNeighborsClassifier()
    kneighbor_2.fit(X_train_2, y_train_2)
    distances_2, indices_2 = kneighbor_2.kneighbors(test_instance)
    expected_distances_2 = [[1.4142135623730951, 1.4142135623730951, 2.00]]
    expected_indices_2 = [[0, 4, 6]]
    assert np.allclose(distances_2, expected_distances_2)
    assert np.allclose(indices_2, expected_indices_2)

    # Dataset 3
    # from Bramer
    X_train_3 = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train_3 = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
                 "-", "-", "+", "+", "+", "-", "+"]
    X_test_3 = [[9.1, 11]]
    kneighbor_3 = MyKNeighborsClassifier()
    kneighbor_3.fit(X_train_3, y_train_3)
    distances_3, indices_3 = kneighbor_3.kneighbors(X_test_3)
    expected_distances_3 = [[0.6082762530298216, 1.2369316876852974, 2.202271554554525]]
    expected_indices_3 = [[6, 5, 7]]
    assert np.allclose(distances_3, expected_distances_3)
    assert np.allclose(indices_3, expected_indices_3)

def test_kneighbors_classifier_predict():
    """
        MyKNeighborsClassifier's predict method.
    """
    # Dataset 1
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    X_test = [[0.33, 1]]
    kneighbors = MyKNeighborsClassifier()
    kneighbors.fit(X_train, y_train)
    y_predicted = kneighbors.predict(X_test)
    expected_prediction = ["good"]
    assert y_predicted == expected_prediction

    # Dataset 2
    # assume normalized
    X_train_2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_2 = [[2, 3]]
    kneighbor_2 = MyKNeighborsClassifier()
    kneighbor_2.fit(X_train_2, y_train_2)
    y_predicted_2 = kneighbor_2.predict(X_test_2)
    expected_prediction_2 = ["yes"]
    assert y_predicted_2 == expected_prediction_2

    # Dataset 3
    # from Bramer
    X_train_3 = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train_3 = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
                 "-", "-", "+", "+", "+", "-", "+"]
    X_test_3 = [[9.1, 11]]
    kneighbor_3 = MyKNeighborsClassifier()
    kneighbor_3.fit(X_train_3, y_train_3)
    y_predicted_3 = kneighbor_3.predict(X_test_3)
    expected_prediction_3 = ["+"]
    assert y_predicted_3 == expected_prediction_3

def test_dummy_classifier_fit():
    """
        MyDummyClassifier's fit method.
    """
    # Dataset 1
    X_train = [[val] for val in range(100)] # 2D
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)
    assert dummy.most_common_label == "yes"

    # Dataset 2
    y_train_2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_2 = MyDummyClassifier()
    dummy_2.fit(X_train, y_train_2)
    assert dummy_2.most_common_label == "no"

    # Dataset 3
    y_train_3 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.1, 0.1, 0.8]))
    dummy_3 = MyDummyClassifier()
    dummy_3.fit(X_train, y_train_3)
    assert dummy_3.most_common_label == "maybe"

def test_dummy_classifier_predict():
    """
        MyDummyClassifier's predict method.
    """
    # Dataset 1
    X_train = [[val] for val in range(100)] # 2D
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)
    dummy.predict([[1]])
    assert dummy.most_common_label == "yes"

    # Dataset 2
    y_train_2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_2 = MyDummyClassifier()
    dummy_2.fit(X_train, y_train_2)
    dummy_2.predict([[4]])
    assert dummy_2.most_common_label == "no"

    # Dataset 3
    y_train_3 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.1, 0.1, 0.8]))
    dummy_3 = MyDummyClassifier()
    dummy_3.fit(X_train, y_train_3)
    dummy_3.predict([[8]])
    assert dummy_3.most_common_label == "maybe"


def compare_priors_and_posteriors(nb, priors_solution, posteriors_solution):
    """
    Tests NaiveBayesClassifier's fit by comparing priors and posteriors to the correct solution.
    Order does not matter as long as keys are on the correct nesting level.

    Args:
        nb (NaiveBayesClassifier): The Naive Bayes Classifier to test. Data is assumed to have
            already been fit using fit().
        priors_solution (dictionary of keys(class): value(prior)): The priors solution to test
            against.
        posteriors_solution (dictionary of class: dictionary of attributes:
            dictionary of values: posterior): the posteriors solution to test against.
    """
    assert len(nb.priors) == len(priors_solution)
    for key, val in priors_solution.items():
        assert key in nb.priors
        assert np.isclose(nb.priors[key], val)

    assert len(nb.posteriors) == len(posteriors_solution)
    for key in posteriors_solution.keys():
        assert key in nb.posteriors
        assert len(nb.posteriors[key]) == len(posteriors_solution[key])
        for att in posteriors_solution[key].keys():
            assert att in nb.posteriors[key]
            assert len(nb.posteriors[key][att]) == len(posteriors_solution[key][att])
            for val, posterior in posteriors_solution[key][att].items():
                assert val in nb.posteriors[key][att]
                assert np.isclose(nb.posteriors[key][att][val], posterior)

def test_naive_bayes_classifier_fit():
    # INCLASS EXAMPLE
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    inclass_priors_solution = {"yes": 0.625, "no": 0.375}
    inclass_posteriors_solution = {
        "yes": {
            "att1": {1: 4/5, 2: 1/5},
            "att2": {5: 2/5, 6: 3/5}
        },
        "no": {
            "att1": {1: 2/3, 2: 1/3},
            "att2": {5: 2/3, 6: 1/3}
        }
    }

    nb = MyNaiveBayesClassifier()
    nb.fit(X_train_inclass_example, y_train_inclass_example)
    compare_priors_and_posteriors(nb, inclass_priors_solution, inclass_posteriors_solution)
    
    # RQ5 IPHONE PURCHASES
    iphone_X_train = [
        [1, 3, "fair"], # no
        [1, 3, "excellent"], # no
        [2, 3, "fair"], # yes
        [2, 2, "fair"], # yes
        [2, 1, "fair"], # yes
        [2, 1, "excellent"], # no
        [2, 1, "excellent"], # yes
        [1, 2, "fair"], # no
        [1, 1, "fair"], # yes
        [2, 2, "fair"], # yes
        [1, 2, "excellent"], # yes
        [2, 2, "excellent"], # yes
        [2, 3, "fair"], # yes
        [2, 2, "excellent"], # no
        [2, 3, "fair"] # yes
    ]
    iphone_y_train = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", \
                        "yes", "yes", "no", "yes"]

    iphone_priors_solution = {"yes": 10/15, "no": 5/15}
    iphone_posteriors_solution = {
        "yes": {
            "att1": {1: 2/10, 2: 8/10},
            "att2": {1: 3/10, 2: 4/10, 3: 3/10},
            "att3": {"fair": 7/10, "excellent": 3/10}
        },
        "no": {
            "att1": {1: 3/5, 2: 2/5},
            "att2": {1: 1/5, 2: 2/5, 3: 2/5},
            "att3": {"fair": 2/5, "excellent": 3/5}
        }
    }

    nb.fit(iphone_X_train, iphone_y_train)
    compare_priors_and_posteriors(nb, iphone_priors_solution, iphone_posteriors_solution)

    # Bramer data set
    bramer_X_train = [
        ["weekday", "spring", "none", "none"], # "on time"
        ["weekday", "winter", "none", "slight"], # "on time"
        ["weekday", "winter", "none", "slight"], # "on time"
        ["weekday", "winter", "high", "heavy"], # "late"
        ["saturday", "summer", "normal", "none"], # "on time"
        ["weekday", "autumn", "normal", "none"], # "very late"
        ["holiday", "summer", "high", "slight"], # "on time"
        ["sunday", "summer", "normal", "none"], # "on time"
        ["weekday", "winter", "high", "heavy"], # "very late"
        ["weekday", "summer", "none", "slight"], # "on time"
        ["saturday", "spring", "high", "heavy"], # "cancelled"
        ["weekday", "summer", "high", "slight"], # "on time"
        ["saturday", "winter", "normal", "none"], # "late"
        ["weekday", "summer", "high", "none"], # "on time"
        ["weekday", "winter", "normal", "heavy"], # "very late"
        ["saturday", "autumn", "high", "slight"], # "on time"
        ["weekday", "autumn", "none", "heavy"], # "on time"
        ["holiday", "spring", "normal", "slight"], # "on time"
        ["weekday", "spring", "normal", "none"], # "on time"
        ["weekday", "spring", "normal", "slight"] # "on time"
    ]
    bramer_y_train = ["on time", "on time", "on time", "late", "on time", "very late","on time", \
                      "on time","very late", "on time", "cancelled", "on time", "late", \
                      "on time", "very late", "on time", "on time",  "on time", "on time", "on time"]
    
    bramer_priors_solution = {'cancelled': 1/20, 'late': 2/20, 'on time': 14/20, 'very late': 3/20}
    bramer_posteriors_solution = {
        'cancelled':{
            "att1": {'saturday': 1},
            "att2": {'spring': 1},
            "att3": {'high': 1},
            "att4": {'heavy': 1}
        },
        'late':{
            'att1': {'saturday': 1/2,'weekday': 1/2},
            'att2': {'winter': 1},
            'att3': {'high': 1/2,'normal': 1/2},
            'att4': {'heavy': 1/2,'none': 1/2}
        },
        'on time':{
            'att1': {'holiday': 2/14,'saturday': 2/14,'sunday': 1/14,'weekday': 9/14},
            'att2': {'autumn': 2/14,'spring': 4/14,'summer': 6/14,'winter': 2/14},
            'att3': {'high': 4/14,'none': 5/14,'normal': 5/14},
            'att4': {'heavy': 1/14,'none': 5/14,'slight': 8/14}
        },
        'very late': {
            'att1': {'weekday': 1},
            'att2': {'autumn': 1/3, 'winter': 2/3},
            'att3': {'high': 1/3,'normal': 2/3},
            'att4': {'heavy': 2/3,'none': 1/3}
        }
    }

    nb.fit(bramer_X_train, bramer_y_train)
    compare_priors_and_posteriors(nb, bramer_priors_solution, bramer_posteriors_solution)

def test_naive_bayes_classifier_predict():
    # INCLASS EXAMPLE
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_inclass_example = [[1, 5]]
    predict_inclass_example_solution = ["yes"]

    nb = MyNaiveBayesClassifier()
    nb.fit(X_train_inclass_example, y_train_inclass_example)
    predicted_inclass_example = nb.predict(X_test_inclass_example)
    assert predicted_inclass_example == predict_inclass_example_solution

    # RQ5 IPHONE PURCHASES
    iphone_X_train = [
        [1, 3, "fair"], # no
        [1, 3, "excellent"], # no
        [2, 3, "fair"], # yes
        [2, 2, "fair"], # yes
        [2, 1, "fair"], # yes
        [2, 1, "excellent"], # no
        [2, 1, "excellent"], # yes
        [1, 2, "fair"], # no
        [1, 1, "fair"], # yes
        [2, 2, "fair"], # yes
        [1, 2, "excellent"], # yes
        [2, 2, "excellent"], # yes
        [2, 3, "fair"], # yes
        [2, 2, "excellent"], # no
        [2, 3, "fair"] # yes
    ]
    iphone_y_train = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", \
                        "yes", "yes", "no", "yes"]
    iphone_X_test = [[2, 2, "fair"],
                     [1, 1, "excellent"]]
    iphone_predict_solution = ["yes", "no"]

    nb.fit(iphone_X_train, iphone_y_train)
    iphone_predicted = nb.predict(iphone_X_test)
    assert iphone_predicted == iphone_predict_solution

    # Bramer data set
    bramer_X_train = [
        ["weekday", "spring", "none", "none"], # "on time"
        ["weekday", "winter", "none", "slight"], # "on time"
        ["weekday", "winter", "none", "slight"], # "on time"
        ["weekday", "winter", "high", "heavy"], # "late"
        ["saturday", "summer", "normal", "none"], # "on time"
        ["weekday", "autumn", "normal", "none"], # "very late"
        ["holiday", "summer", "high", "slight"], # "on time"
        ["sunday", "summer", "normal", "none"], # "on time"
        ["weekday", "winter", "high", "heavy"], # "very late"
        ["weekday", "summer", "none", "slight"], # "on time"
        ["saturday", "spring", "high", "heavy"], # "cancelled"
        ["weekday", "summer", "high", "slight"], # "on time"
        ["saturday", "winter", "normal", "none"], # "late"
        ["weekday", "summer", "high", "none"], # "on time"
        ["weekday", "winter", "normal", "heavy"], # "very late"
        ["saturday", "autumn", "high", "slight"], # "on time"
        ["weekday", "autumn", "none", "heavy"], # "on time"
        ["holiday", "spring", "normal", "slight"], # "on time"
        ["weekday", "spring", "normal", "none"], # "on time"
        ["weekday", "spring", "normal", "slight"] # "on time"
    ]
    bramer_y_train = ["on time", "on time", "on time", "late", "on time", "very late","on time", \
                      "on time","very late", "on time", "cancelled", "on time", "late", \
                      "on time", "very late", "on time", "on time",  "on time", "on time", "on time"]
    bramer_X_test = [
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "high", "heavy"],
        ["sunday", "summer", "normal", "slight"]
    ]
    bramer_predict_solution = ["very late", "on time", "on time"]

    nb.fit(bramer_X_train, bramer_y_train)
    bramer_predicted = nb.predict(bramer_X_test)
    assert bramer_predicted == bramer_predict_solution


def test_decision_tree_classifier_fit():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
            ['Attribute', 'att0',
                ['Value', 'Junior', 
                    ['Attribute', 'att3', 
                        ['Value', 'no', 
                            ['Leaf', 'True', 3, 5]
                        ], 
                        ['Value', 'yes', 
                            ['Leaf', 'False', 2, 5]
                        ]
                    ]
                ], 
                ['Value', 'Mid',
                    ['Leaf', 'True', 4, 14]
                ], 
                ['Value', 'Senior', 
                    ['Attribute', 'att2', 
                        ['Value', 'no', 
                            ['Leaf', 'False', 3, 5]
                        ], 
                        ['Value', 'yes', 
                            ['Leaf', 'True', 2, 5]
                        ]
                    ]
                ]
            ]
    dt = MyDecisionTreeClassifier()
    dt.fit(X_train_interview, y_train_interview)
    assert dt.tree == tree_interview

    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ['A', 'B', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'A', 'B', 'B', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'A', 'A', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'B', 'B'],
        ['B', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'B', 'A', 'A'],
        ['B', 'B', 'B', 'A', 'A'],
        ['B', 'B', 'A', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['B', 'A', 'B', 'A', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'B', 'A', 'B', 'B'],
        ['B', 'A', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B']
    ]
    y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                    'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                    'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND']

    tree_degrees = \
        ['Attribute', 'att0', 
            ['Value', 'A', 
                ['Attribute', 'att4', 
                    ['Value', 'A', 
                        ['Leaf', 'FIRST', 5, 14]
                    ], 
                    ['Value', 'B', 
                        ['Attribute', 'att3', 
                            ['Value', 'A', 
                                ['Attribute', 'att1', 
                                    ['Value', 'A', 
                                        ['Leaf', 'FIRST', 1, 2]
                                    ], 
                                    ['Value', 'B', 
                                        ['Leaf', 'SECOND', 1, 2]
                                    ]
                                ]
                            ],
                            ['Value', 'B', 
                                ['Leaf', 'SECOND', 7, 9]
                            ]
                        ]
                    ]
                ]
            ], 
            ['Value', 'B', 
                ['Leaf', 'SECOND', 12, 26]
            ]
        ]

    dt.fit(X_train_degrees, y_train_degrees)
    assert dt.tree == tree_degrees
    
    # iPhone dataset
    X_train_iphone = [
        [1, 3, "fair"], # no
        [1, 3, "excellent"], # no
        [2, 3, "fair"], # yes
        [2, 2, "fair"], # yes
        [2, 1, "fair"], # yes
        [2, 1, "excellent"], # no
        [2, 1, "excellent"], # yes
        [1, 2, "fair"], # no
        [1, 1, "fair"], # yes
        [2, 2, "fair"], # yes
        [1, 2, "excellent"], # yes
        [2, 2, "excellent"], # yes
        [2, 3, "fair"], # yes
        [2, 2, "excellent"], # no
        [2, 3, "fair"] # yes
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", \
                        "yes", "yes", "no", "yes"]
    tree_iphone = \
        ['Attribute', 'att0', 
            ['Value', 1, 
                ['Attribute', 'att1', 
                    ['Value', 1, 
                        ['Leaf', 'yes', 1, 5]
                    ], 
                    ['Value', 2, 
                        ['Attribute', 'att2', 
                            ['Value', 'excellent', 
                                ['Leaf', 'yes', 1, 2]
                            ],
                            ['Value', 'fair', 
                                ['Leaf', 'no', 1, 2]
                            ]
                        ]
                    ], 
                    ['Value', 3, 
                        ['Leaf', 'no', 2, 5]
                    ]
                ]
            ], 
            ['Value', 2, 
                ['Attribute', 'att2', 
                    ['Value', 'excellent', 
                        ['Leaf', 'no', 4, 10]
                    ], 
                    ['Value', 'fair', 
                        ['Leaf', 'yes', 6, 10]
                    ]
                ]
            ]
        ]

    dt.fit(X_train_iphone, y_train_iphone)
    assert dt.tree == tree_iphone

def test_decision_tree_classifier_predict():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    X_test_interview = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_test_interview = ["True", "False"]

    dt = MyDecisionTreeClassifier()
    dt.fit(X_train_interview, y_train_interview)
    predictions = dt.predict(X_test_interview)
    assert predictions == y_test_interview

    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ['A', 'B', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'A', 'B', 'B', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'A', 'A', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'B', 'B'],
        ['B', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'B', 'A', 'A'],
        ['B', 'B', 'B', 'A', 'A'],
        ['B', 'B', 'A', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['B', 'A', 'B', 'A', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'B', 'A', 'B', 'B'],
        ['B', 'A', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B']
    ]
    y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                    'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                    'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND']
    
    X_test_degrees = [["B", "B", "B", "B", "B"],
                      ["A", "A", "A", "A", "A"],
                      ["A", "A", "A", "A", "B"]]
    y_test_degrees = ["SECOND", "FIRST", "FIRST"]

    dt.fit(X_train_degrees, y_train_degrees)
    predictions = dt.predict(X_test_degrees)
    assert predictions == y_test_degrees

    # iPhone dataset
    X_train_iphone = [
        [1, 3, "fair"], # no
        [1, 3, "excellent"], # no
        [2, 3, "fair"], # yes
        [2, 2, "fair"], # yes
        [2, 1, "fair"], # yes
        [2, 1, "excellent"], # no
        [2, 1, "excellent"], # yes
        [1, 2, "fair"], # no
        [1, 1, "fair"], # yes
        [2, 2, "fair"], # yes
        [1, 2, "excellent"], # yes
        [2, 2, "excellent"], # yes
        [2, 3, "fair"], # yes
        [2, 2, "excellent"], # no
        [2, 3, "fair"] # yes
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", \
                        "yes", "yes", "no", "yes"]
    
    X_test_iphone = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_test_iphone = ["yes", "yes"]

    dt.fit(X_train_iphone, y_train_iphone)
    predictions = dt.predict(X_test_iphone)
    assert predictions == y_test_iphone
    dt.print_decision_rules()

def test_random_forest_fit():
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]


    random_state = 2
    N = 3
    M = 2
    F = 2

    tree_1 = ['Attribute', 'att0', 
            ['Value', 'Junior', 
                    ['Leaf', 'True', 1, 10]
                ], 
                ['Value', 'Mid', 
                    ['Leaf', 'True', 6, 10]
                ], 
                ['Value', 'Senior', 
                    ['Attribute', 'att2', 
                        ['Value', 'no', 
                            ['Leaf', 'False', 2, 3]
                        ], 
                        ['Value', 'yes', 
                            ['Leaf', 'True', 1, 3]
                        ]
                    ]
                ]
            ]

    tree_2 = ['Attribute', 'att0', 
                ['Value', 'Junior', 
                    ['Leaf', 'True', 1, 10]
                ], 
                ['Value', 'Mid', 
                    ['Leaf', 'True', 6, 10]
                ], 
                ['Value', 'Senior', 
                    ['Attribute', 'att3', 
                        ['Value', 'no', 
                            ['Leaf', 'False', 1, 3]
                        ], 
                        ['Value', 'yes', 
                            ['Leaf', 'False', 2, 3]
                        ]
                    ]
                ]
            ]

    trees = [tree_1, tree_2]

    rf = MyRandomForestClassifier(N, M, F, random_state=2)
    rf.fit(X_train, y_train)
    for tree in rf.random_forest:
        assert tree.tree in trees

def test_random_forest_predict():
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    X_test= [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_test = ['True', 'True']

    random_state = 2
    N = 3
    M = 2
    F = 2
    rf = MyRandomForestClassifier(N, M, F, random_state=2)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    for i in range(len(predictions)):
        assert predictions[i] == y_test[i]
