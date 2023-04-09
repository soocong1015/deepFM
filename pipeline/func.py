from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd


def slice_df(table, col_name):
    test_df = table[table[col_name].isnull() == True]
    train_df = table[table[col_name].isnull() == False]
    return test_df, train_df

def slice_columns(*col_names):
    list = []
    for col in col_names:
        list.append(col)
    return list

def scaler_encoder(sparese, dense, target_df):
    target_df = target_df.copy()
    mms = MinMaxScaler(feature_range=(0, 1))
    target_df[dense] = mms.fit_transform(target_df[dense])
    for feat in sparese:
        lbe = LabelEncoder()
        target_df[feat] = lbe.fit_transform(target_df[feat])
    return target_df

def fixlen_feature(target_df, slice_sparse, slice_dense):
    fixlen_feature_columns = [SparseFeat(feat, target_df[feat].max() + 1, embedding_dim=4) for feat in slice_sparse] +\
        [DenseFeat(feat, 1, ) for feat in slice_dense]
    return fixlen_feature_columns

def clone_feature(fix):
    dnn = fix
    linear = fix
    return dnn, linear

def get_feature_fun(dnn_feat, linear_feat):
    final = get_feature_names(linear_feat + dnn_feat)
    return final

def train_test_slice(target_df, size, random, feature_col):
    train, test = train_test_split(target_df, test_size = size, random_state = random)
    train_model_input = {name : train[name] for name in feature_col}
    test_model_input = {name : test[name] for name in feature_col}
    return train, test, train_model_input, test_model_input


def train_test_nonslice(target_df, feature_col):
    pred_y = {name: target_df[name] for name in feature_col}
    return pred_y

def pred_input(transform_target_df, pred_y_val, pred_y_colname):
    transform_target_df[pred_y_colname] = pred_y_val
    return transform_target_df


def concat_col(predict_df, old_train_df):
    df_new = (pd.concat([predict_df, old_train_df], axis = 0)).sort_index()
    return df_new