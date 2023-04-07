from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd


def data_loader(path):
    table = pd.read_csv(path, index_col=[0])
    return table


df = data_loader('./data/laptops.csv')


def slice_df(table, col_name):
    test_df = table[table[col_name].isnull() == True]
    train_df = table[table[col_name].isnull() == False]
    return test_df, train_df


test_df, train_df = slice_df(df, 'rating')


def slice_sparses(*col_names):
    sparse_list = []
    for col in col_names:
        sparse_list.append(col)
    return sparse_list


def slice_denses(*col_names):
    dense_list = []
    for col in col_names:
        dense_list.append(col)
    return dense_list


slice_sparse = slice_sparses(
    'img_link', 'name', 'processor', 'ram', 'os', 'storage')

slice_dense = slice_denses('price(in Rs.)', 'display(in inch)')


def scaler_encoder(sparese, dense, target_df):
    target_df = target_df.copy()
    mms = MinMaxScaler(feature_range=(0, 1))
    target_df[dense] = mms.fit_transform(target_df[dense])
    for feat in sparese:
        lbe = LabelEncoder()
        target_df[feat] = lbe.fit_transform(target_df[feat])

    return target_df


train_df_transform = scaler_encoder(slice_sparse, slice_dense, train_df)

test_df_transform = scaler_encoder(slice_sparse, slice_dense, test_df)


def fixlen_feature(target_df, slice_sparse_col, slice_dense_col):
    fixlen_feature_columns = [SparseFeat(feat, target_df[feat].max() + 1, embedding_dim=4) for feat in slice_sparse] +\
        [DenseFeat(feat, 1, ) for feat in slice_dense]
    return fixlen_feature_columns


train_fixlen_feature = fixlen_feature(
    train_df_transform, slice_sparse, slice_dense)

test_fixlen_feature = fixlen_feature(
    test_df_transform, slice_sparse, slice_dense)


def clone_fixlen_feature(fixlen_features):
    dnn_feature_col = fixlen_features
    linear_feature_col = fixlen_features
    return get_feature_names(linear_feature_col + dnn_feature_col)


train_last_feature = clone_fixlen_feature(train_fixlen_feature)
