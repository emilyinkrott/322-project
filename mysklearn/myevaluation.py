"""
Programmer: Greeley Lindberg
Class: CPSC 322-02, Spring 2021
Programming Assignment #7
4/13/22

Description: implements several basic meachine learninig algorithms.
"""

import numpy as np

from mysklearn import myutils

# pylint: disable=invalid-name
def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set
            (e.g. 0.33 for a 2:1 split) or int for absolute number of instances to
            be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible
            results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order
            of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        for i, _ in enumerate(X):
            rand_index = np.random.randint(0, len(X))
            X[i], X[rand_index] = X[rand_index], X[i]
            y[i], y[rand_index] = y[rand_index], y[i]

    # Split into train and test sets
    if isinstance(test_size, float):
        split_index = len(X) - int(test_size * len(X)) - 1
    elif isinstance(test_size, int):
        if test_size > len(X) or test_size < 0:
            raise IndexError
        split_index = len(X) - test_size
    else:
        raise ValueError

    X_train = X[0:split_index]
    X_test = X[split_index:]
    y_train = y[0:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible
            results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_indices = [i for i in range(len(X))]
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        for i, _ in enumerate(X):
            rand_index = np.random.randint(0, len(X))
            X_indices[i], X_indices[rand_index] = X_indices[rand_index], X_indices[i]

    X_train_folds = []
    X_test_folds = []

    # Split into train and test sets
    start_index = 0
    first_n_samples = len(X) % n_splits   # used for determining test sample size
    for i in range(n_splits):
        if first_n_samples >= 0:
            split_index = start_index + len(X) // n_splits + 1
        else:
            split_index = start_index + len(X) // n_splits
        X_test_folds.append(X_indices[start_index:split_index])
        X_train_folds.append(X_indices[0:start_index] + X_indices[split_index:])
        first_n_samples -= 1
        start_index += len(X_test_folds[i])

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible
            results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_indices = [i for i in range(len(X))]
    if random_state is not None:
        np.random.seed(random_state)

    X_train_folds = []
    X_test_folds = []

    # Split into test set
    _, group_subtables = myutils.group_by(X_indices, y)   # Partition and group by classification
    first_n_samples = len(X) % n_splits   # used for determining test sample size
    group_index = 0   # the index of the current group to deal from

    for i in range(n_splits):
        if i < first_n_samples:
            split_size = len(X) // n_splits + 1
        else:
            split_size = len(X) // n_splits
        # take turns dealing an instance from each subgroup until test sample is full
        test_fold = []
        for _ in range(split_size):
            while len(group_subtables[group_index]) == 0:
                # no instances left in group, go to next non-empty group
                if group_index >= len(group_subtables) - 1:
                    # reset group iterate to 0
                    group_index = 0
                else:
                    # deal from the next group
                    group_index += 1
            if shuffle:
                rand_index = np.random.randint(0, len(group_subtables[group_index]))
                test_fold.append(group_subtables[group_index].pop(rand_index))
            else:
                test_fold.append(group_subtables[group_index].pop(0))
            # increment group index or wrap around
            if group_index >= len(group_subtables) - 1:
                group_index = 0
            else:
                group_index += 1

        X_test_folds.append(test_fold)

    # Fill in train set with remaining indices
    for fold in X_test_folds:
        train_fold = []
        for index in X_indices:
            if index not in fold:
                train_fold.append(index)
        X_train_folds.append(train_fold)

    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is
            automatically set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible
            results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag"
            (parallel to X_out_of_bag). None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if random_state is not None:
        np.random.seed(random_state)
    X_sample = []
    X_out_of_bag = []
    if y is not None:
        y_sample = []
        y_out_of_bag = []
    else:
        y_sample = None
        y_out_of_bag = None
    if n_samples is None:
        n_samples = len(X)
    bag_indices = []

    # build train set (bootstrap)
    for _ in range(n_samples):
        rand_index = np.random.randint(0, len(X))
        X_sample.append(X[rand_index])
        if y is not None:
            y_sample.append(y[rand_index])
        bag_indices.append(rand_index)
    # build test set (out of bag)
    for i, row in enumerate(X):
        if i not in bag_indices:
            X_out_of_bag.append(row)
            if y is not None:
                y_out_of_bag.append(y[i])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []

    for label in labels:
        row = [0] * len(labels)
        for true_val, pred_val in zip(y_true, y_pred):
            if true_val == label:
                row[labels.index(pred_val)] += 1
        matrix.append(row)

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples
            (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_true_vals = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pred_val:
            num_true_vals += 1

    if normalize:
        return num_true_vals / len(y_true)
    else:
        return num_true_vals

def accuracy_score_confidence_interval(accuracy, n_samples, confidence_level=0.95):
    """Compute the classification prediction accuracy score confidence interval.

    Args:
        accuracy(float): Classification accuracy to compute a confidence interval for
        n_samples(int): Number of samples in the test set used to compute the accuracy
        confidence_level(float): Level of confidence to use for computing a confidence interval
            0.9, 0.95, and 0.99 are supported. Default is 0.95

    Returns:
        lower_bound(float): Lower bound of the accuracy confidence interval
        upper_bound(float): Upper bound of the accuracy confidence interval

    Notes:
        Raise ValueError on invalid confidence_level
        Assumes accuracy and n_samples are correct based on training/testing
            set generation method used (e.g. holdout, cross validation, bootstrap, etc.)
            See Bramer Chapter 7 for more details
    """
    std_err = ((accuracy * (1 - accuracy)) / n_samples) ** 0.5

    if confidence_level == 0.95:
        lower_bound = accuracy - 1.96 * std_err
        upper_bound = accuracy + 1.96 * std_err
    elif confidence_level == 0.9:
        lower_bound = accuracy - 1.64 * std_err
        upper_bound = accuracy + 1.64 * std_err
    elif confidence_level == 0.99:
        lower_bound = accuracy - 2.58 * std_err
        upper_bound = accuracy + 2.58 * std_err
    else:
        raise ValueError

    return lower_bound, upper_bound


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    true_pos_count, false_pos_count = 0, 0
    for pred, true in zip(y_pred, y_true):
        if pred == true and pred == pos_label:
            true_pos_count += 1
        else:
            if pred == pos_label:
                false_pos_count += 1

    if (true_pos_count + false_pos_count) == 0:
        return 0
    return true_pos_count / (true_pos_count + false_pos_count)


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where
        tp is the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    true_pos_count, false_neg_count = 0, 0
    for pred, true in zip(y_pred, y_true):
        if true == pos_label:
            if pred == true:
                true_pos_count += 1
            else:
                false_neg_count += 1

    if (true_pos_count + false_neg_count) == 0:
        return 0
    return true_pos_count / (true_pos_count + false_neg_count)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if (precision + recall) == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)
