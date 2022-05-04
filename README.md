# 322-Project: Pokemon Battle Predictor
Team members: Emily Inkrott and Greeley Lindberg

## How to Download
Simply clone this repository using `git clone https://github.com/emilyinkrott/322-project.git` and `cd` into its location.

## How to Run
Classifiers can easily be tested using `pytest`. Jupyter Notebooks can be run by hitting the "Run All" button in your editor of choice. Make sure your kernel is set to `base(Python 3.9.7)`. 

## Structure
Below you will find the schema of this repository:
* __input_data__: Folder containing all datasets
* __mysklearn__: Data mining package similar to scikitlearn
    * __myclassifier.py__: Implementation of classifiers (kNN, Naive Bayes, Dummy, Decision Tree, Random Forest)
    * __myevaluation.py__: Functions for evaluating classifier performance
    * __mypytable.py__: Class implementation of dataset similar to pandas DataFrame.
    * __mysimplelinearregressor.py__: Implementation of simple linear regressor. Used by some classifiers.
    * __myutils.py__: Utility functions
* __EDA.ipynb__: Exploratory data analysis of dataset attributes and feature selection.
* __plot_utils.py__: Utility functions for plots and graphs
* __project_proposal.ipynb__: Proposal and introductory data collection.
* __technical_report.ipynb__: Full technical analysis of dataset, feature selection, classifier performance comparison, and conclusive findings.
* __test_classifiers.py__: Unit tests of all classifiers