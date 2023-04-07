import pandas as pd


def data_loader(path):
    table = pd.read_csv(path, index_col=[0])
    return table
