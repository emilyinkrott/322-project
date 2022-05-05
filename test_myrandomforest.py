from mysklearn.myclassifiers import MyRandomForestClassifier
import numpy as np

# interview dataset
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
X_test= [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
y_test = ['True', 'True']

def test_random_forest_fit():
    #np.random.seed(2)
    random_state = 2
    N = 3
    M = 2
    F = 2
    print(np.random.randint(0,4, size=F)) #tree 1 will use level and phd
    print(np.random.randint(0,4, size=F)) #tree 2 will ue lang and level
    print(np.random.randint(0,4, size=F)) #tree 3 will use tweets and phd

    rf = MyRandomForestClassifier(N, M, F, random_state=2)
    rf.fit(X_train, y_train)
    for tree in rf.random_forest:
        assert tree.tree in trees

def test_random_forest_predict():
    random_state = 2
    N = 3
    M = 2
    F = 2
    rf = MyRandomForestClassifier(N, M, F, random_state=2)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    for i in range(len(predictions)):
        assert predictions[i] == y_test[i]


    