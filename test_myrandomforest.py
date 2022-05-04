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

# tree 1 uses attribute level and phd
tree_1 = ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid", 
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ]
        ]



def test_random_forest_fit():
    np.random.seed(2)
    N = 3
    M = 2
    F = 2
    print(np.random.randint(0,4, size=F)) #tree 1 will use level and phd
    print(np.random.randint(0,4, size=F)) #tree 2 will ue lang and level
    print(np.random.randint(0,4, size=F)) #tree 3 will use tweets and phd

    rf = MyRandomForestClassifier(N, M, F)
    rf.fit(X_train, y_train)
    for tree in rf.random_forest:
        print(tree)
        print()

    assert False is True

test_random_forest_fit()
def test_random_forest_predict():
    assert False is True