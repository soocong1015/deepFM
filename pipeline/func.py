import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pipeline.load import data_loader

df = data_loader('./data/laptops.csv')


def columns_tolist(table):
    columns = table.columns.values.tolist()


columns = columns_tolist(df)


def slice_df(table, col_name):
    test_df = table[table[col_name].isnull() == True]
    train_df = table[table[col_name].isnull() == False]
    return test_df, train_df


test_df, train_df = slice_df(df, 'rating')


def slice_sparse(*col_names):
    sparse_list = []
    for col in col_names:
        sparse_list.append(col)
    print(sparse_list)


def slice_dense(*col_names):
    dense_list = []
    for col in col_names:
        dense_list.append(col)
    print(dense_list)


slice_sparse = slice_sparse(
    'img_link', 'name', 'processor', 'ram', 'os', 'storage')

slice_sparse = slice_sparse('price(in Rs.)', 'display(in inch)')
