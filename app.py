from cProfile import run
import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle as pkl
import plotly.express as px
from plotly import graph_objects as go

st.title('Machine Learning Algorithms')

@st.cache
def load_trained_models():
    linear = pkl.load(open('linear.pkl'))
    random_forest = pkl.load(open('random_forest.pkl', 'r'))
    

# Define inputs
data_relationship = st.selectbox(
    'What relationship is likely between your independent and dependent variables?',
    ('None', 'Linear', 'Discrete', 'Exponential', 'Logarithmic')
)

@st.cache
def load_test_data(data_relationship):
    dat = pd.read_csv(f'data/test_{data_relationship.lower()}.csv')
    return dat

model_types = ('Linear', 'RandomForest')
model_1_type = st.selectbox(
    'Model 1',
    model_types
)
model_2_type = st.selectbox(
    'Model 2',
    model_types
)

def load_models(model_1_type, model_2_type):
    model1 = pkl.load(open(f'models/{data_relationship.lower()}_{model_1_type.lower()}.pkl', 'rb'))
    model2 = pkl.load(open(f'models/{data_relationship.lower()}_{model_2_type.lower()}.pkl', 'rb'))
    return model1, model2

    
# Define outputs
dat = load_test_data(data_relationship)
x = np.array(dat['x'])
x = x.reshape(-1, 1)
model1, model2 = load_models(model_1_type, model_2_type)
model_1_pred = model1.predict(x)
model_2_pred = model2.predict(x)

fig1 = go.Figure()
scatter_dict = {
    'size': 3,
    'opacity': 0.5,
    'color': 'LightSkyBlue',
}
line_dict = {
    'color': 'red',
    'width': 6,
}
fig1.add_trace(go.Scatter(x=dat['x'], y=model_1_pred,
                    mode='lines',
                    name=f'{model_1_type}', 
                    line=line_dict))
fig1.add_trace(go.Scatter(x=dat['x'], y=dat['y'],
                    mode='markers',
                    name='True',
                    marker=scatter_dict))
st.plotly_chart(fig1)
st.text(f'Fig 1. Modeled Predictions for {model_1_type} model')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dat['x'], y=model_2_pred,
                    mode='lines',
                    name=f'{model_2_type}',
                    line=line_dict))
fig2.add_trace(go.Scatter(x=dat['x'], y=dat['y'],
                    mode='markers',
                    name='True',
                    marker=scatter_dict))
st.plotly_chart(fig2)
st.text(f'Fig 2. Modeled Predictions for {model_2_type} model')
# 
st.text('Table 1. Error Metrics')