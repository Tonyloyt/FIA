#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#Saving best model 
import joblib

import warnings
warnings.filterwarnings('ignore')


#load the model from disk

model = joblib.load(r"model.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Financial Inclusion in Africa Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict the likehood of an individual to have bank account
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

     #Setting Application sidebar default
    image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict the likehood of customer to possess bank account')
    st.sidebar.image(image)

    if add_selectbox == "Online":

        st.info("Input data below")

        #Based on our optimal features selection
        st.subheader("Personal informations:")
        location_type = st.selectbox('Customer Location:', ('Rural', 'Urban'))
        cellphone_access = st.selectbox('Cellphone Access:', ('Yes', 'No'))
        household_size = st.number_input('Household Size:',min_value=0, max_value=50, value=0)

        age_of_respondent = st.number_input('Age of Respondent:',min_value=1, max_value=200, value=1)

        gender_of_respondent = st.selectbox('Gender of respondent:', ('Female', 'Male'))

        relationship_with_head = st.selectbox('Relationship with Head of the Family:', ('Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent','Other non-relatives'))

        marital_status = st.selectbox('Marital Status:', ('Married/Living together', 'Widowed', 'Single/Never Married','Divorced/Seperated', 'Dont know'))
        education_level = st.selectbox('Highest Education Level:', ('Secondary education', 'No formal education','Vocational/Specialised training', 'Primary education','Tertiary education', 'Other/Dont know/RTA'))
        job_type = st.selectbox('Job Type:', ('Self employed', 'Government Dependent','Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing','Remittance Dependent', 'Other Income','Dont Know/Refuse to answer', 'No Income'))


        data = {
           'location_type':location_type,
           'cellphone_access':cellphone_access,
           'household_size':household_size,
           'age_of_respondent':age_of_respondent,
           'gender_of_respondent':gender_of_respondent,
           'relationship_with_head':relationship_with_head,
           'marital_status':marital_status,
           'education_level':education_level,
           'job_type':job_type 
           }

        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

          #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer has Bank Account.')
            else:
                st.success('No, the customer  has No Bank Account.')
                
     # batch prediction
    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer has Bank Account.', 
                                                    0:'No, the customer  has No Bank Account.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
        main()

