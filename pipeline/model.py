from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

def make_model(linear_feature_columns, dnn_feature_columns, activation):
    DEFAULT_GROUP_NAME = "default_group"
    model = DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), dnn_hidden_units=(256, 128, 64),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation=activation, dnn_use_bn=False, task='regression')
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
    return model

    
def fit_model(model, input_data, output_data, frequency):
    hist = model.fit(input_data, output_data, batch_size=256, epochs=frequency, verbose=2, validation_split=0.2)
    return hist

def model_predict(model, test_input):
    pred = model.predict(test_input)
    return pred