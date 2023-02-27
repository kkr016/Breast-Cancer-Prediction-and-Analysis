import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Page title

st.set_page_config(page_title="cancer prediction", page_icon = ":tada", layout = "wide")

with st.container():
        
    st.write("""
    # Breast Cancer Prediction
    This app predicts the **Breast Cancer** for the given data
    Data obtained from the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) 
    by Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian""")
st.write("---")
with st.container():
    
    st.sidebar.header('User Input Features')

    def user_input_features():
        
        concave_points_worst = st.sidebar.slider('Concave points worst', 0.000000,0.291000,0.015)
        radius_worst = st.sidebar.slider('Radius worst', 7.930000,36.040000,9.0)
        concavity_mean = st.sidebar.slider('Concavity mean', 0.000000,0.426800,0.39)        
        concave_points_mean = st.sidebar.slider('Concave points mean', 0.000000,0.426800,0.06)
        area_worst = st.sidebar.slider('Area worst', 185.200000,4254.000000,201.0)
        perimeter_worst = st.sidebar.slider('Perimeter worst', 50.410000,251.200000,60.9)
        area_mean = st.sidebar.slider('Area_mean', 143.500000,2501.000000,157.2)
        perimeter_mean = st.sidebar.slider('Perimeter mean', 43.790000,188.500000,57.2)
        concavity_worst = st.sidebar.slider('Concavity worst', 0.000000,1.252000,1.0)
        radius_mean = st.sidebar.slider('Radius mean', 6.981000,28.110000,7.0)
        
        
        
        data = {'concave_points_worst': concave_points_worst,
                'radius_worst': radius_worst,
                'concavity_mean': concavity_mean,
                'concave_points_mean': concave_points_mean,
                'area_worst': area_worst,
                'perimeter_worst': perimeter_worst,
                'area_mean': radius_worst,
                'perimeter_mean': perimeter_mean,
                'concavity_worst': concavity_worst,
                'radius_mean': radius_mean                
                 }
        
        features = pd.DataFrame(data, index=[0])
        features = list(features.loc[0])
        return features
    
    input_df = user_input_features()

with st.container():

    # Displays the user input features
    st.subheader('User Input Data for 10 features')
    columns = ['concave_points_worst', 'radius_worst', 'concavity_mean', 'concave_points_mean',
               'area_worst', 'perimeter_worst', 'area_mean', 'perimeter_mean', 'concavity_worst', 'radius_mean']

    data1 = {'concave_points_worst': input_df[0],
            'radius_worst': input_df[1],
            'concavity_mean': input_df[2],
            'concave_points_mean': input_df[3],
            'area_worst': input_df[4],
            'perimeter_worst': input_df[5],
            'area_mean': input_df[6],
            'perimeter_mean': input_df[7],
            'concavity_worst': input_df[8],
            'radius_mean': input_df[9]
            }
    
    data0 = [input_df]
    
    sample = pd.DataFrame(data0, columns=columns).T

    st.write(sample.T)

# Reads in saved classification model
load_clf = pickle.load(open('D:\cancer_diagnosis\cancer_diagnosis.pkl', 'rb'))

# Scalling the unser input function
orginal_data = pd.read_csv('D:\cancer_diagnosis\_final_data.csv')
X = orginal_data.drop('diagnosis',axis=1).values
std = StandardScaler()
std.fit(X)

scalled_input = std.transform(np.array(input_df).reshape(1, -1))


# Apply model to make predictions
prediction = load_clf.predict(scalled_input)
prediction_proba = load_clf.predict_proba(scalled_input)

with st.container():

    st.write("---")
    st.subheader('Prediction')

    cancer_type = np.array(['Bening', 'Malignant'])
    st.write(cancer_type[prediction])

    st.write("---")
    st.write("##")
    st.subheader('Prediction Probability')
    st.write(prediction_proba)