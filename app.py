import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle 

# Load the model
model = tf.keras.models.load_model("model.h5")

# Load the encoders

with open("label_encoder_gender.pkl","rb") as file:
    encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl","rb") as file:
    encoder_geo = pickle.load(file)

# Load the scaler

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# "creditscore	geography	gender	age	tenure	balance	numofproducts	hascrcard	isactivemember	estimatedsalary"

# streamlit app
st.title("Customer Churn Prediction")

#User inputs
geography = st.selectbox("Geography",encoder_geo.categories_[0])
gender = st.selectbox("Gender", encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Prepare the input data
input_data = pd.DataFrame({
    "creditscore" : [credit_score],
    "gender" : [encoder_gender.transform([gender])[0]],
    "age" : [age],
    "tenure" : [tenure],
    "balance" : [balance],
    "numofproducts" : [num_of_products],
    "hascrcard" : [has_cr_card],
    "isactivemember" : [is_active_member],
    "estimatedsalary" : [estimated_salary]
}, index=[0])

# Encode the geography

geo_df = pd.DataFrame(encoder_geo.transform([[geography]]).toarray(), columns=encoder_geo.get_feature_names_out(), dtype="int", index=[0])

# Append geography to our input_df

input_data = pd.concat([input_data, geo_df], axis=1)

# scaling

input_data_scaled = scaler.transform(input_data)

# Prediction

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Decision

st.write(prediction_proba)

if prediction_proba > 0.5:
    st.error("The customer will likely churn")
else:
    st.success("Customer will not churn")   

