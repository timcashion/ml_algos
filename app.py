from statistics import mean
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from plotly import graph_objects as go
from sklearn.metrics import mean_squared_error

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

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dat['x'], y=model_2_pred,
                    mode='lines',
                    name=f'{model_2_type}',
                    line=line_dict))
fig2.add_trace(go.Scatter(x=dat['x'], y=dat['y'],
                    mode='markers',
                    name='True',
                    marker=scatter_dict))

# col1, col2 = st.columns([4,2])
# with col1:
# with col2:


# Error table 
def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** (1/2)

metrics = [mean_squared_error, root_mean_squared_error] # Easy to add additional metrics
table = []
for metric in metrics:
    output = {}
    output[model_1_type] = metric(dat['y'], model_1_pred)
    output[model_2_type] = metric(dat['y'], model_2_pred)
    df = pd.DataFrame(output, index=[metric.__name__])
    table.append(df)
table = pd.concat(table)

note_text = '''
Notes:  
Models are trained on 1,000 observations of a given relationship and tested on another 1,000 observations with a standardized relationship.  
Gaussian noise is introduced to both (error with a mean of 0).  
All models have a single independent variable. This makes it easier to visualize the relationships in the data.  
'''
# Render outputs:
st.title('Machine Learning Algorithms')
st.plotly_chart(fig1)
st.text(f'Fig 1. Modeled Predictions for {model_1_type} model')
st.plotly_chart(fig2)
st.text(f'Fig 2. Modeled Predictions for {model_2_type} model')
st.header('Table 1. Error Metrics')
st.dataframe(table.style.format("{:.2f}")) # Format to 2 decimals places 
st.markdown(note_text)
