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

filename = os.path.join("input_data", "pokemon_combats_advantage.csv")
new_table = MyPyTable().load_from_file(filename)

stat_cols = ["HP","Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
X_stats = myutils.get_columns(new_table.data, new_table.column_names, stat_cols)
y = new_table.get_column("Winner")
print(stat_cols)
print(X_stats[0], y[0])


nb = MyNaiveBayesClassifier()

nb.fit(X_stats, y)

packaged_obj = [new_table.column_names, nb]
outfile = open("nb.p", "wb") #file type doesn't really matter
pickle.dump(packaged_obj, outfile)
outfile.close()