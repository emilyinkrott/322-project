import importlib
import os
import pickle

#plotting imports
from tkinter import CENTER
import matplotlib.pyplot as plt

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable 

import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

# pokemon stats
filename = os.path.join("input_data", "pokemon_combats_advantage.csv")
table = MyPyTable().load_from_file(filename)
print(table.column_names)
y_train = table.get_column("Winner")
X_train = myutils.get_columns(table.data, table.column_names, table.column_names[:-1])
nb = MyNaiveBayesClassifier()

nb.fit(X_train, y_train)

packaged_obj = [table.column_names, nb]
outfile = open("nb.p", "wb") #file type doesn't really matter
print(outfile)
pickle.dump(packaged_obj, outfile)
outfile.close()