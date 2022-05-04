"""
Programmer: Greeley Lindberg
Class: CPSC 322-02, Spring 2021
Programming Assignment #7
4/13/22

Description: Implementation of 5 types of classifers: linear regression, kNN, Naive Bayes, Dummy,
             and decision tree.
"""
import operator
import math
import numpy as np

from mysklearn import myutils, myevaluation
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

# pylint: disable=invalid-name
class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor:
            self.regressor.fit(X_train, y_train)
        else:
            slope, intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
                y_train)
            self.regressor = MySimpleLinearRegressor(slope, intercept)
            self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.regressor is None:
            return []

        predictions = self.regressor.predict(X_test)
        discretized_predictions = self.discretizer(predictions)
        return discretized_predictions

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # assumes X_train and X_test have been normalized
        distances = []
        neighbor_indices = []

        for test_instance in X_test:
            row_indexes_dists = []
            for i, train_instance in enumerate(self.X_train):
                dist = myutils.compute_euclidean_distance(train_instance, test_instance)
                row_indexes_dists.append([i, dist])

            # sort the list by each item's distance (at index 1 or -1)
            row_indexes_dists.sort(key=operator.itemgetter(-1))

            # grab the top k
            row_distances = []
            row_neighbor_indices = []
            top_k = row_indexes_dists[:self.n_neighbors]

            for row in top_k:
                row_neighbor_indices.append(row[0])
                row_distances.append(row[1])

            neighbor_indices.append(row_neighbor_indices)
            distances.append(row_distances)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for index in neighbor_indices:
            vals = [self.y_train[i] for i in index]
            values, counts = myutils.get_frequencies(vals)
            y_predicted.append(values[counts.index(max(counts))])

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self, strategy=0):
        """Initializer for DummyClassifier.

        """
        self.strategy = strategy
        self.most_common_label = None
        self.label_frequencies = {}

    def fit(self, _, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        vals, counts = myutils.get_frequencies(y_train)

        if self.strategy == 0:
            self.most_common_label = vals[counts.index(max(counts))]
        else:
            total_counts = sum(counts)
            for i, val in enumerate(vals):
                self.label_frequencies[val] = counts[i] / total_counts

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.strategy == 0:
            return [self.most_common_label] * len(X_test)

        stratified_prediction = np.random.choice(list(self.label_frequencies.keys()), len(X_test),\
                                                 p=list(self.label_frequencies.values()))
        return stratified_prediction


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict of key(classifier): value(prior)): The prior probabilities computed for each
            label in the training set.
        posteriors(dictionary of class: dictionary of attributes: dictionary of values: posterior):
            The posterior probabilities computed for each attribute value/label pair in the
            training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers:
            https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this
            docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior
                probabilities and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        group_names, group_subtables = myutils.group_by(X_train, y_train) # group instances by y

        # compute priors
        self.priors = {}
        total = len(y_train)
        for i, name in enumerate(group_names):
            self.priors[name] = len(group_subtables[i]) / total

        # compute posteriors
        self.posteriors = {}
        for i, group in enumerate(group_subtables):
            # start by inverting group so columns are easily accessible
            inverted_group = myutils.invert_2d_list(group)  # inverts columns and rows
            self.posteriors[group_names[i]] = {}    # make empty dict for group
            # compute posteriors for each attribute in group
            for j, col in enumerate(inverted_group):
                values, counts = myutils.get_frequencies(col)
                posteriors_for_attributes = {}
                for val, count in zip(values, counts):
                    posteriors_for_attributes[val] = count / len(group)
                self.posteriors[group_names[i]]["att"+str(j+1)] = posteriors_for_attributes

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        # put priors into a list of tuples (key, value) and sort descending by value
        # note that I sort the priors high to low. This way if there is a tie in probabilities,
        # I will predict the one with the highest prior beacuase .index takes the first instance.
        sorted_priors = sorted(self.priors.items(), key=lambda x: x[1], reverse=True)
        probabilities = [[prior[1] for prior in sorted_priors] for _ in range(len(X_test))]

        # calculate probabilities for each instance
        for i, instance in enumerate(X_test):
            for j, att_val in enumerate(instance):
                for k, classification in enumerate(sorted_priors):
                    try:
                        probabilities[i][k] *= \
                            self.posteriors[classification[0]]["att"+str(j+1)][att_val]
                    except KeyError:
                        # value doesn't occur for classification, posterior is 0
                        probabilities[i][k] = 0

        # predict using probabilities
        for i in range(len(X_test)):
            prediction_index = probabilities[i].index(max(probabilities[i]))
            y_predicted.append(sorted_priors[prediction_index][0])

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def compute_priors_and_posteriors(self, instances):
        """Computes the "priors" and posteriors of a given instance. Note that the
            "priors" are not the same as in Naive Bayes. Here, they refer to the
            number of occurences of an attribute over all instances (not occurences of
            class label).

        Args:
            instances(list of list of obj): The list of training instances (samples)
                The shape of instances is (n_train_samples, n_features)
        Returns:
            priors(dict obj:float): dictionary mapping classifications to odds
            posteriors(dict obj:list of float): dictionary mapping classifications to odds
        """
        X_instances = [instance[:-1] for instance in instances]
        y_instances = [instance[-1] for instance in instances]

        group_names, group_subtables = myutils.group_by(X_instances, y_instances) # group inst by y

        # compute posteriors
        priors = {}
        total = len(y_instances)
        posteriors = {}
        for i, group in enumerate(group_subtables):
            # start by inverting group so columns are easily accessible
            inverted_group = myutils.invert_2d_list(group)  # inverts columns and rows
            posteriors[group_names[i]] = {}    # make empty dict for group
            # compute posteriors for each attribute in group
            for j, col in enumerate(inverted_group):
                values, counts = myutils.get_frequencies(col)
                posteriors_for_attributes = {}
                priors["att"+str(j)] = {}
                for val, count in zip(values, counts):
                    try:
                        priors["att"+str(j)][val] += count / total
                    except KeyError:
                        priors["att"+str(j)][val] = count / total
                    posteriors_for_attributes[val] = count / len(group)
                posteriors[group_names[i]]["att"+str(j)] = posteriors_for_attributes

        return priors, posteriors

    def select_attribute(self, attribute_domains, instances, attributes):
        """Selects an attribute to split using the lowest intropy attribute.
        Args:
            instances (list of list of obj): list of X_train instances
            attributes (list of obj): list of availible attributes
        Returns:
            selected_attribute (obj): attribute with lowest entropy
        """
        priors, posteriors = self.compute_priors_and_posteriors(instances)
        e_news = []
        class_labels = sorted(list(set(self.y_train)))

        for att in attributes:
            entropies = []
            # compute entropies
            for val in attribute_domains[int(att[-1])]:
                e_val = 0
                for label in class_labels:
                    try:
                        posterior = posteriors[label][att][val]
                        if posterior == 0:
                            # can't take log of 0, e_val is 0
                            e_val = 0
                            break
                        e_val += -posterior * math.log(posterior, 2).real
                    except KeyError:
                        # value doesn't occur for class label, posterior is 0
                        e_val = 0
                        break
                entropies.append(e_val)
            # compute e_new
            e_new = 0
            for domain, entropy in zip(attribute_domains[int(att[-1])], entropies):
                try:
                    e_new += priors[att][domain] * entropy
                except KeyError:
                    e_new += 0
            e_news.append(e_new)

        selected_attribute = attributes[e_news.index(min(e_news))]
        return selected_attribute

    def partition_instances(self, header, attribute_domains, instances, split_attribute):
        """Partition instances based on split_attribute
        Args:
            header (list of str): header of instances
            attribute_domains (dict str:list of str): possible values for each attribute
            instances (list of list of obj): list of X_train instances
            split_attribute (obj): attribute to split instances (and the branch) on
        Returns:
            partitions (dict str:list of obj): dictionary mapping attribute values to instances
        """
        # this is a group by attribute domain
        partitions = {} # key (attribite value): value (subtable)
        att_index = header.index(split_attribute) # e.g. level -> 0
        att_domain = sorted(attribute_domains[att_index]) # e.g. ["Junior", "Mid", "Senior"]
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def all_same_class(self, att_partition):
        """Checks if a partition contains all of the same class label
        Args:
            att_partition (list of list of obj): list of X_train instances
        Returns:
            True if all instances have same class label, False otherwise
        """
        first_val = att_partition[0][-1]
        for instance in att_partition:
            if instance[-1] != first_val:
                return False
        return True

    def tdidt(self, header, attribute_domains, current_instances, available_attributes):
        """Implementation of Top Down Induction of Decision Trees algorithm to recursively
            build a a decision tree.
        Args:
            header (list of str): header of instances
            attribute_domains (dict str:list of str): possible values for each attribute
            current_instances (list of list of obj): list of X_train instances
            availible_attributes (obj): attribute to split instances (and branch) on
        Returns:
            tree[list[list]]: the decision tree
        """
        # select an attribute to split on
        attribute=self.select_attribute(attribute_domains, current_instances, available_attributes)
        available_attributes.remove(attribute)  # can't split on same att twice
        # this subtree
        tree = ["Attribute", attribute] # start to build tree

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions=self.partition_instances(header, attribute_domains, current_instances, attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]
            # CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                branch_sum = 0
                for instances in partitions.values():
                    branch_sum += len(instances)
                leaf = ["Leaf", att_partition[0][-1], len(att_partition), branch_sum]
                value_subtree.append(leaf)
            # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # get frequency of attributes and choose majority (alphabetically first on tie)
                possible_classifiers = [instance[-1] for instance in att_partition]
                classifiers, counts = myutils.get_frequencies(possible_classifiers)
                majority_vote = classifiers[counts.index(max(counts))]
                # note that because classifiers is sorted, majority_vote will be the first value
                # alphabetically in the event of a tie.
                leaf = ["Leaf", majority_vote, len(att_partition), len(current_instances)]
                value_subtree.append(leaf)
            # CASE 3: no more instances to partition (empty partition) =>
            # backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                # get frequency of attributes and choose majority (alphabetically first on tie)
                possible_classifiers = [instance[-1] for instance in current_instances]
                classifiers, counts = myutils.get_frequencies(possible_classifiers)
                majority_vote = classifiers[counts.index(max(counts))]
                tree = ["Leaf", majority_vote, len(current_instances), 0]   # 0 is a placeholder

            else: # none of the previous conditions were true -> recurse
                subtree = self.tdidt(header, attribute_domains, \
                                     att_partition, available_attributes.copy())
                # note the copy
                # if case 3 leaf, replace 0 with the length of its parent's current_instances
                if subtree[0] == "Leaf":
                    subtree[-1] = len(current_instances)
                value_subtree.append(subtree)
            tree.append(value_subtree)

        return tree

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        # programmatically create a header (e.g ["att0", "att1", ...]
        # and create an attribute domains dictionary)
        header = [f"att{i}" for i in range(len(X_train[0]))]
        attribute_domains = {}
        for i, _ in enumerate(header):
            domains = []
            for row in X_train:
                if row[i] not in domains:
                    domains.append(row[i])
            attribute_domains[i] = domains

        # next, stitch X_train and y_train together
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # now, make a copy of the header because the tdidt() is going to modify the list
        available_attributes = header.copy()
        # recall: python is pass by object reference
        tree = self.tdidt(header, attribute_domains, train, available_attributes)
        self.tree = tree

    def traverse_tree(self, tree, instance):
        """
        Recursively traverses a decision tree according to a given instance.

        Args:
            tree (list(list)): working decision tree
            instance (list of obj): instance guiding traversal

        Returns:
            (list) leaf node from traversal
        """
        # return leaf
        if tree[0] == "Leaf":
            return tree[1]

        # traverse the decision tree
        split_attribute = instance[int(tree[1][-1])]
        subtree = []
        # pylint: disable=consider-using-enumerate
        for i in range(len(tree)):
            branch = tree[i]
            try:
                if branch[0] == "Value" and branch[1] == split_attribute:
                    subtree = branch[2]
                    break
                elif branch[0] == "Leaf" and branch[1] == split_attribute:
                    return branch[1]
            except TypeError:
                # not looking at a branch or leaf, keep going
                continue

        return self.traverse_tree(subtree, instance)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = [self.traverse_tree(self.tree, row) for row in X_test]
        return predictions

    def build_rules(self, tree, attributes, values, attribute_names, class_name):
        """
        Recursively builds and print rules generated from decision tree.

        Args:
            tree (list of list): working decision tree
            attributes: list of attributes on lhs (parallel to values)
            values: list of values on lhs (parallel to attributes)
            class_name: A string to use for the class name in the decision rule
        """
        for i, branch in enumerate(tree):
            try:
                if branch == "Attribute":
                    name_index = int(tree[i+1][-1])
                    attributes.append(attribute_names[name_index])
                    for j in range(i, len(tree)):
                        self.build_rules(tree[i+j], attributes.copy(), values.copy(),\
                                         attribute_names, class_name)
                elif branch == "Value":
                    values.append(tree[i+1])
                    self.build_rules(tree[i+2], attributes.copy(), values.copy(),\
                                     attribute_names, class_name)
                elif branch == "Leaf":
                    rule = "IF "
                    for att, val in zip(attributes, values):
                        rule += f"{att} == {val} AND "
                    rule = rule[:-4] + "THEN "
                    rule += f"{class_name} == {tree[i+1]}"
                    print(rule)
                    return
            except TypeError:
                continue


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision
                 rules.
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = [f"att{i}" for i in range(len(self.X_train[0]))]
        attributes, values = [], []
        self.build_rules(self.tree, attributes, values, attribute_names, class_name)

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        N(int): The number of trees to generate
        M(int): The number of most accurate trees to select
        F(int): The number of random attributes to partition from


    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, N=20, M=7, F=2):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.N = N
        self.M = M
        self.F = F

    def random_stratified_test_set(self, X, y, remainder_size=0.66, random_state=0, shuffle=True):
        """Split dataset into test and remainder sets based on a remainser set size.

        Args:
            X(list of list of obj): The list of samples
            y(list of obj): The target y values (parallel to X)
            remainder_size(float or int): float for proportion of dataset to be in original test set
                (e.g. 0.33 for a 2:1 split) 
            random_state(int): integer used for seeding a random number generator for reproducible
                results
                Use random_state to seed your random number generator
                    you can use the math module or use numpy for your generator
                    choose one and consistently use that generator throughout your code
            shuffle(bool): whether or not to randomize the order of the instances before splitting
                Shuffle the rows in X and y before splitting and be sure to maintain the parallel order
                of X and y!!

        Returns:
            X_test(list of list of obj): The list of test samples
            y_test(list of list of obj): The list of target y values samples (parallel to X_test)
            X_remainder(list of obj): The list of remaining X values 
            y_remainder(list of obj): The list of remaining target y values for testing (parallel to X_remainder)
        """
        X_test = []
        y_test = []
        X_remainder = []
        y_remainder = []

        # TODO: perform stratefied split. Use code from myevaluation's train_test_split
        # and stratefied k-fold cross validation for stratifying and splitting

        return X_test, y_test, X_remainder, y_remainder
  
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # Step 1: generate random stratified test set (1/3rd orignal set; 2/3rd remainder set)
        # TODO: finish writing the function called below
        X_test, y_test, X_remainder, y_remainder = self.random_stratified_test_set(X_train, y_train)

        # Step 2: generate N random trees using bootstrapping (giving a training and validation set)
        #  over the remainder set
            # NOTE: At each node, build your decision trees by randomly selecting F of the
            # remaining attributes as candidates to partition on. Use entropy to choose which
            # attribute to actually partition on. This will require a modified DecisionTreeClassifier

        # store trees and training/validation sets parallel to each other
        X_training_sets = []
        y_training_sets = []
        X_validation_sets = []
        y_validation_sets = []
        forest = []

        for _ in range(self.N):
            # bootstrap remainder set
            X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X_remainder, y_remainder)
            X_training_sets.append(X_sample)
            y_training_sets.append(y_sample)
            X_validation_sets.append(X_out_of_bag)
            y_validation_sets.append(y_out_of_bag)
            # make a (modified) decision tree
            tree = MyDecisionTreeClassifier()   # TODO: Make a modified Decision Tree class
            tree.fit(X_sample, y_sample)
            forest.append(tree)

        # Step 3: Select the M most accurate of the N trees using the corresponding validation sets
        accuracies = []
        for i, tree in enumerate(forest):
            y_pred = []
            for instance in X_training_sets[i]:
                y_pred.append(tree.predict(instance))
            accuracies.append(myevaluation.accuracy_score(y_validation_sets[i], y_pred))
        # Select top M trees
        pruned_forest = []
        for _ in range(self.M):
            max_tree_index = accuracies.index(max(accuracies))
            pruned_forest.append(forest.pop(max_tree_index))


    def predict(self, X_test):
        # TODO: Step 4: Use simple majority voting to predict classes using the M trees over the test set

        pass