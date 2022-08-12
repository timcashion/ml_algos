import random
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Generate random datasets of various relationships 
n = 10000
def output_data(x, y, label):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df.to_csv(f'data/{label}.csv', index=False)

def generate_random_data(n, data_type):
    x_values = [random.random() * 100 for x in range(n)]
    noise_values = [random.gauss(0,10) for x in range(n)]
    y_none = [(random.random() * 100) + z for z in noise_values]
    y_linear = [(2 * x) + z for x,z in zip(x_values, noise_values)]
    y_exponentional = [(x ** 2) + z for x,z in zip(x_values, noise_values)]
    y_logarithmic = [math.log(x) + z for x,z in zip(x_values, noise_values)]
    y_discrete = [(x * 2) + z if x < 50 else (x * 3) + z for x,z in zip(x_values, noise_values)]
    output_data(x_values, y_none, f'{data_type}_none')
    output_data(x_values, y_linear, f'{data_type}_linear')
    output_data(x_values, y_exponentional, f'{data_type}_exponential')
    output_data(x_values, y_logarithmic, f'{data_type}_logarithmic')
    output_data(x_values, y_discrete, f'{data_type}_discrete')


generate_random_data(10000, 'train')
generate_random_data(1000, 'test')
