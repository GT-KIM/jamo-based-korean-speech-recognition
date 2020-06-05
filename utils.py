import pandas as pd

def load_split(path) :
    train_split = pd.read_csv(path + "train_list.csv")
    test_split = pd.read_csv(path + "test_list.csv")

    return list(train_split['name']), list(test_split['name'])
