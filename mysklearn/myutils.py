"""
Programmer: Greeley Lindberg
Class: CPSC 322-02, Spring 2021
Programming Assignment #7
4/13/22

Description: utils file containing various helper functions.
"""

import numpy as np
from tabulate import tabulate

# pylint: disable=invalid-name
def compute_slope_intercept(x, y):
    """Computes slope intercept of x values and y values

        Args:
            x (list of numeric values): list of x values
            y (list of numeric values): list of y values

        Returns:
            m (float): m in the equation y = mx + b
            b (float): b in the equation y = mx + b
        """
    sum_x = 0
    for row in x:
        sum_x += row[0]
    mean_x = sum_x / len(x)
    mean_y = sum(y) / len(y)

    num = sum([(x[i][0] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
    den = sum([(x[i][0] - mean_x) ** 2 for i in range(len(x))])
    m = num / den
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return m, b

def compute_euclidean_distance(v1, v2):
    """Computes euclidean distance between values.

        Args:
            v1 (list of numeric values): list of first set of values
            v2 (list of numeric values): list of second set of values

        Returns:
            distance (float): euclidean distance between v1 and v2
    """
    distance = 0
    # pylint: disable=consider-using-enumerate
    for i in range(len(v1)):
        if isinstance(v1[i], str):
            distance += 0 if v1[i] == v2[i] else 1
        else:
            distance += (v1[i] - v2[i]) ** 2

    return np.sqrt(distance)

def normalize(vals):
    """Normalizes values.

        Args:
            vals (list of numeric values): list of values to normalize

        Returns:
            normalized (list of numeric values): list of normalized values parallel to vals
    """
    min_val = min(vals)
    range_val = max(vals) - min_val
    normalized = []
    for val in vals:
        normalized.append((val - min_val) / range_val)

    return normalized

def discretize_high_low(vals):
    """Discretizes values into "low" (<100) and "high" (>=100) categories

        Args:
            vals (list of numeric values): list of values to discretize

        Returns:
            discretized (list(str)): list of discretized values parallel to vals
    """
    discretized = []
    for val in vals:
        if val < 100:
            discretized.append("low")
        else:
            discretized.append("high")

    return discretized

def discretize_mpg_ratings(vals):
    """Discretizes mpg values into Department of Energy ratings.

        Args:
            vals (list of numeric values): list of mpgs to discretize

        Returns:
            discretized (list(int)): list of discretized values parallel to vals
    """
    discretized = []
    for val in vals:
        if val <= 13:
            discretized.append(1)
        elif val == 14:
            discretized.append(2)
        elif 15 <= val <= 16:
            discretized.append(3)
        elif 17 <= val <= 19:
            discretized.append(4)
        elif 20 <= val <= 23:
            discretized.append(5)
        elif 24 <= val <= 26:
            discretized.append(6)
        elif 27 <= val <= 30:
            discretized.append(7)
        elif 31 <= val <= 36:
            discretized.append(8)
        elif 37 <= val <= 44:
            discretized.append(9)
        else:
            discretized.append(10)

    return discretized

def get_frequencies(values):
    """Finds the frequencies of each value in a set

        Args:
            vals (list of numeric values): list of values to find frequencies of

        Returns:
            freqs (list of objects): list of each type of value in values
            counts (list(int)): list of number of times each type of value occured
    """
    values.sort() # inplace
    # parallel lists
    freqs = []
    counts = []
    for value in values:
        if value in freqs: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            freqs.append(value)
            counts.append(1)

    return freqs, counts

def group_by(X, y):
    """Groups values by classification

        Args:
            X (list of list of obj): instances to group
            y (list of str): classifications for instances (parallel to X)

        Returns:
            group_names (list of str): the set of classification
            group_subtables (list of list of obj): instances pertaining to each classification
                (parallel to group_names)

    """
    group_names = sorted(list(set(y)))
    group_subtables = [[] for _ in group_names] # e.g. [[], [], []]

    for i, row in enumerate(X):
        groupby_val = y[i]
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(row) # make a copy

    return group_names, group_subtables

def invert_2d_list(values):
    """
    Inverts the rows and columns of a 2d list so columns become rows and rows become columns.

    Args:
        values (list of list of objects): the list to invert.

    Returns:
        inverted_list (list of list of objects): list of values with rows and columns inverted.
    """
    inverted_list = [[] for _ in range(len(values[0]))]
    for row in values:
        for i, val in enumerate(row):
            inverted_list[i].append(val)
    return inverted_list

def print_results(title, accuracy, precision, recall, f1):
    """Prints results of classification

        Args:
            title (str): title of test to put on table
            accuracy (float): accuracy score of classifier
            precision (float): precision score of classifier
            recall (float): recall score of classifier
            f1 (float): f1 score of classifier
    """
    print(f"""
===========================================
{title}
===========================================
accuracy score: {round(accuracy, 3)}
error rate: {round(1-accuracy, 3)}
precision score: {round(precision, 3)}
recall score: {round(recall, 3)}
f1 score: {round(f1, 3)}
""")


def print_confusion_matrix(header, matrix, class_labels):
    """Prints results of classification

        Args:
            header (list of str): header for confusion matrix
            matrix (list of obj): list of matrix body
            class_labels (list of str): list of class labels
    """
    for i, row in enumerate(matrix):
        recognized, total = 0, 0
        for j, col in enumerate(row):
            total += col
            if i == j:
                recognized += col
        recognized = recognized/total *100 if total > 0 else 0
        matrix[i] = row + [total, round(recognized, 3)]
    new_header = header + ['Total', 'Recognition (%)']
    print(tabulate(matrix, headers=new_header, showindex=class_labels))

def get_columns(table, header, col_names):
    """returns the columns with the given names

        Args:
            table (list of list): the 2D list of data
            header (list of str): header for the table
            col_name (list of str): names of the desired columns
        Returns: 
            cols (list of list): a table with only the columns for the specified attributes.
    """
    col_indexes = [header.index(name) for name in col_names]
    cols = []
    for row in table:
        new_row = [row[index] for index in col_indexes]
        cols.append(new_row)
    return cols

def get_column(table, header, col_name):
    """Returns column with the given name

        Args:
            table (list of list): the 2D list of data
            header (list of str): header for the table
            col_name (str): name of the desired column
        Returns: 
            col (list): the list of values in the specified column
    """
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col 

def group_by(table, header, groupby_col_name):
    """A second groupby function for unknown domains

        Args:
            table (list of list): the 2D list of data
            header (list of str): header for the table
            groupby_col_name (str): name of the group by column
        Returns: 
            group_names (list of str): the set of classification
            group_subtables (list of list of obj): instances pertaining to each classification
                (parallel to group_names)
    """
    groupby_col_index = header.index(groupby_col_name) # use this later
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col))) # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names] # e.g. [[], [], []]
    
    for row in table:
        groupby_val = row[groupby_col_index] # e.g. this row's modelyear
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(row.copy()) # make a copy
    
    return group_names, group_subtables