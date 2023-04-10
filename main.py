from pipeline.load import *
from pipeline.func import *
from pipeline.model import *


def main():
    df = data_loader('./data/laptops.csv')

    test_df, train_df = slice_df(df, 'rating')

    sparse_col = slice_columns(
        'img_link', 'name', 'processor', 'ram', 'os', 'storage')
    dense_col = slice_columns('price(in Rs.)', 'display(in inch)')

    train_df_transform = scaler_encoder(sparse_col, dense_col, train_df)
    test_df_transform = scaler_encoder(sparse_col, dense_col, test_df)

    fixlen_feature_col = fixlen_feature(
        train_df_transform, sparse_col, dense_col)

    dnn, linear = clone_feature(fixlen_feature_col)

    final_feature_name = get_feature_fun(dnn, linear)

    train_train, train_test, train_train_input, train_test_input = train_test_slice(
        train_df_transform, 0.2, 2020, final_feature_name)

    deepfm_model = make_model(linear, dnn, 'relu')

    history = fit_model(deepfm_model, train_train_input,
                        train_train['rating'].values, 300)

    train_train_pred = model_predict(deepfm_model, train_test_input)

    # test_plot(train_test['rating'], train_train_pred)

    # loss_plot(history)

    test_test = train_test_nonslice(test_df_transform, final_feature_name)

    test_test_pred = model_predict(deepfm_model, test_test)

    predict_test_df = pred_input(test_df_transform, test_test_pred, 'rating')

    df_new = concat_col(predict_test_df, train_df_transform)


if __name__ == "__main__":
    print('시작!! ╰(*°▽°*)╯')
    main()
    print('끝!! (oﾟvﾟ)ノ')
