import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt



boston = load_boston()
x, y = boston.data, boston.target
sz = 30
x_df = DataFrame(x, columns= boston.feature_names)[['CRIM',    'ZN' ]].to_numpy()[0:sz]
y = y[0:sz]

x_df = np.array([[x[0],1] for x in x_df])

x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.15)

# defining parameters
params = {
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learnnig_rage': 0.05,
    'metric': {'l2', 'l1'},
    'verbose': -1
}


# laoding data
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


# fitting the model
model = lgb.train(params,
                 train_set=lgb_train,
                 valid_sets=lgb_eval,
                 early_stopping_rounds=30)

# prediction
y_pred = model.predict(x_test)

# accuracy check
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** (0.5)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % rmse)

MSE: 7.66
RMSE: 2.77





