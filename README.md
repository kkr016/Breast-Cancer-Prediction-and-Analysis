# Breast-Cancer-Prediction-and-Analysis Using SVM model and Streamlit web application
This is a system that uses a Breast Cancer Prediction model built with the Streamlit framework and scikit-learn library. Users are able to input tumor details into the model, and based on the data entered, the model will predict whether the tumor is malignant or benign.

## Dataset
Breast Cancer Wisconsin (Diagnostic) Data Set Source: Creators: 1. Dr. William H. Wolberg, 2. W. Nick Street, 3. Olvi L. Donor: Nick Street
[Click here to visit Data set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## Features
1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

## Breast Cancer Prediction model
The Breast Cancer Prediction model uses support vector machine (SVM) with a linear kernel to predict whether a tumor is malignant or benign. The model was trained on a dataset and achieved an accuracy of 95.10%, an F1 score of 0.95, and an AUC score of 0.99. The test data was then used to evaluate the model, and it achieved an accuracy of 95.80%, an F1 score of 0.96, and an AUC score of 1.00

To achieve these high accuracy scores, the Breast Cancer Prediction model was trained on an imbalanced dataset where there were a higher number of benign cases compared to malignant cases. To address this imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was used to oversample the minority class and balance the dataset. Moreover, to further improve the model's performance, the important 10 features were extracted from the dataset using an Extra Trees Classifier. These techniques not only help to balance the dataset but also increase the accuracy and robustness of the model, making it a valuable tool for healthcare professionals in diagnosing breast cancer

## Web Application for the Model
The user interface for the Breast Cancer Prediction model was developed using the Streamlit framework. It features a slider bar where the user can select input data, and upon selection, the 10 most important features are displayed. The model then predicts the type of cancer based on the input data and displays the probability of each type. The user-friendly interface allows for easy use of the model

## Demo Video


## Usage
To run the Breast Cancer Prediction model, 

clone the repository to your local machine
install the required libraries using pip install -r requirements.txt
Finally, run the app.py file using streamlit run app.py command

Once the app is running on the local host, select an input from the slider bar, and the model will automatically provide the prediction and probability for the given data. This simple and straightforward process enables easy and quick use of the model

