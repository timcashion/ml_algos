from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import numpy as np
import pickle as pkl

train_files = os.listdir('data/')
train_files = [x for x in train_files if x.find('train') > -1]
models = [LinearRegression(), RandomForestRegressor()]

for file in train_files:
    train_data = pd.read_csv(f'data/{file}')
    data_label = file.replace('train_', '')
    data_label = data_label.replace('.csv', '')
    for model in models:
        x = np.array(train_data['x'])
        x = x.reshape(-1, 1)
        model.fit(x, train_data['y'])
        model_label = str(model)
        model_label = model_label.replace('Regressor()', '')
        model_label = model_label.replace('Regression()', '')
        model_label = model_label.lower()
        pkl.dump(model, open(f'models/{data_label}_{model_label}.pkl', 'wb'))
