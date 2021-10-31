import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import warnings
warnings.filterwarnings('ignore')

def preprocess(df, option):
    
    #Drop values based on operational options
    if (option == "Online"):
        df['location_type'] = le.fit_transform(df['location_type'])
        df['cellphone_access'] = le.fit_transform(df['cellphone_access'])
        df['gender_of_respondent'] = le.fit_transform(df['gender_of_respondent'])
        df['relationship_with_head'] = le.fit_transform(df['relationship_with_head'])
        df['education_level'] = le.fit_transform(df['education_level'])
        df['marital_status'] = le.fit_transform(df['marital_status'])
        df['job_type'] = le.fit_transform(df['job_type'])
    elif (option == "Batch"):
        pass
        df = df[['location_type', 'cellphone_access', 'household_size','age_of_respondent', 'gender_of_respondent', 'relationship_with_head','marital_status', 'education_level', 'job_type']]
        # columns = ['location_type', 'cellphone_access','gender_of_respondent', 'relationship_with_head','marital_status', 'education_level', 'job_type']
        #covertion
        df['location_type'] = le.fit_transform(df['location_type'])
        df['cellphone_access'] = le.fit_transform(df['cellphone_access'])
        df['gender_of_respondent'] = le.fit_transform(df['gender_of_respondent'])
        df['relationship_with_head'] = le.fit_transform(df['relationship_with_head'])
        df['education_level'] = le.fit_transform(df['education_level'])
        df['marital_status'] = le.fit_transform(df['marital_status'])
        df['job_type'] = le.fit_transform(df['job_type'])
    else:        #Encoding the other categorical categoric features with more than two categories

        print("Incorrect operational options")

    return df

