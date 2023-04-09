from pipeline.load import *
from pipeline.func import *
from pipeline.model import *

def data_loader(path):
    table = pd.read_csv(path, index_col=[0])
    return table

df = data_loader('./data/laptops.csv')

def slice_df(table, col_name):
    test_df = table[table[col_name].isnull() == True]
    train_df = table[table[col_name].isnull() == False]
    return test_df, train_df

test_df, train_df = slice_df(df, 'rating')

def slice_columns(*col_names):
    list = []
    for col in col_names:
        list.append(col)
    return list

sparse_col = slice_columns('img_link', 'name', 'processor', 'ram', 'os', 'storage')
dense_col = slice_columns('price(in Rs.)', 'display(in inch)')

def scaler_encoder(sparese, dense, target_df):
    target_df = target_df.copy()
    mms = MinMaxScaler(feature_range=(0, 1))
    target_df[dense] = mms.fit_transform(target_df[dense])
    for feat in sparese:
        lbe = LabelEncoder()
        target_df[feat] = lbe.fit_transform(target_df[feat])
    return target_df

train_df_transform = scaler_encoder(sparse_col, dense_col, train_df)
test_df_transform = scaler_encoder(sparse_col, dense_col, test_df)

def fixlen_feature(target_df, slice_sparse, slice_dense):
    fixlen_feature_columns = [SparseFeat(feat, target_df[feat].max() + 1, embedding_dim=4) for feat in slice_sparse] +\
        [DenseFeat(feat, 1, ) for feat in slice_dense]
    return fixlen_feature_columns

fixlen_feature_col = fixlen_feature(train_df_transform, sparse_col, dense_col)

def clone_feature(fix):
    dnn = fix
    linear = fix
    return dnn, linear

dnn, linear = clone_feature(fixlen_feature_col)

def get_feature_fun(dnn_feat, linear_feat):
    final = get_feature_names(linear_feat + dnn_feat)
    return final

final_feature_name = get_feature_fun(dnn, linear)

def train_test_slice(target_df, size, random, feature_col):
    train, test = train_test_split(target_df, test_size = size, random_state = random)
    train_model_input = {name : train[name] for name in feature_col}
    test_model_input = {name : test[name] for name in feature_col}
    return train, test, train_model_input, test_model_input


train_train, train_test, train_train_input, train_test_input = train_test_slice(train_df_transform, 0.2, 2020, final_feature_name)

def make_model(linear_feature_columns, dnn_feature_columns, activation):
    DEFAULT_GROUP_NAME = "default_group"
    model = DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), dnn_hidden_units=(256, 128, 64),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation=activation, dnn_use_bn=False, task='regression')
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
    return model


deepfm_model = make_model(linear, dnn, 'relu')

def fit_model(model, input_data, output_data, frequency):
    hist = model.fit(input_data, output_data, batch_size=256, epochs=frequency, verbose=2, validation_split=0.2)
    return hist

history = fit_model(deepfm_model, train_train_input, train_train['rating'].values, 300)

def model_predict(model, test_input):
    pred = model.predict(test_input)
    return pred

train_train_pred = model_predict(deepfm_model, train_test_input)

def test_plot(test_output, test_pred_output):
    plt.plot(test_output.reset_index(drop=True))
    plt.plot(test_pred_output)
    plt.show()


# test_plot(train_test['rating'], train_train_pred)

def loss_plot(histo):
    plt.plot(histo.history['loss'])
    plt.plot(histo.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

# loss_plot(history)


def train_test_nonslice(target_df, feature_col):
    pred_y = {name: target_df[name] for name in feature_col}
    return pred_y


test_test = train_test_nonslice(test_df_transform, final_feature_name)

test_test_pred = model_predict(deepfm_model, test_test)

def pred_input(transform_target_df, pred_y_val, pred_y_colname):
    transform_target_df[pred_y_colname] = pred_y_val
    return transform_target_df


predict_test_df = pred_input(test_df_transform, test_test_pred, 'rating')

def concat_col(predict_df, old_train_df):
    df_new = (pd.concat([predict_df, old_train_df], axis = 0)).sort_index()
    return df_new


df_new = concat_col(predict_test_df, train_df_transform)


if __name__ == "__main__":
    pass