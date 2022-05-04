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

# tree 1 uses attributes level (att0) and phd (att3)
tree_1 = ['Attribute', 'att0',
            ['Value', 'Junior',
                ['Attribute', 'att1',
                    ['Value', 'Python',
                            ['Leaf', 'True', 3, 4]
                    ], 
                    ['Value', 'R', 
                        ['Leaf', 'False', 1, 4]
                    ]
                ]
            ],
            ['Value', 'Mid',
                ['Leaf', 'True', 4, 10]
            ],
            ['Value', 'Senior', 
                ['Leaf', 'True', 2, 10]
            ]
        ]

#tree 2 uses attributes level (att0) and lang (att1)
tree_2 = ['Attribute', 'att2',
            ['Value', 'no', 
                ['Leaf', 'True', 6, 10]
            ], 
            ['Value', 'yes', 
                ['Attribute', 'att3', 
                    ['Value', 'no', 
                        ['Leaf', 'True', 2, 4]
                    ],
                    ['Value', 'yes', 
                        ['Leaf', 'False', 2, 4]
                    ]
                ]
            ]
        ]

#tree 3 uses attributes tweets (att2) and phd (att3)
tree_3 = ["Attribute", "att2",
            ["Value", "no",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "False", 4, 7]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 3, 7]
                    ]
                ]
            ],
            ["Value", "yes",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 4, 7]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "True", 3, 7]
                    ]
                ]
            ]
        ]
trees = [tree_1, tree_2, tree_3]

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
        print(tree.tree)
        print()
        # assert tree.tree in trees
    
    assert False is True

test_random_forest_fit()

def test_random_forest_predict():
    assert False is True