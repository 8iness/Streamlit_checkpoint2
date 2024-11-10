#I-Install the necessary packages
import pandas as pd
import streamlit as st 
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# give a title to our app 
st.title('Streamlit checkpoint 2') 

#II-Import you data and perform basic data exploration phase
df = pd.read_csv('Financial_inclusion_dataset.csv')

#1-Display general information about the dataset
    #Display the first lines
st.subheader("Display the first lines")    
st.write(df.head())
    #Data information
st.subheader("Data information")

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text(s)
    #Statistical summary of the dataset
st.subheader("Statistical summary of the dataset")
st.write(df.describe())

#2-Duplicates values
st.subheader("Duplicates values")
duplicates = df.duplicated().sum()
st.write(f"Number of duplicate values ​​in dataset: {duplicates}")

#3-Handle outliers, if they exist

#Show numeric columns
st.subheader("Numeric columns")
numeric_columns = df.select_dtypes(include=['number']).columns
st.write (numeric_columns)

#Display box plots for numeric column
st.subheader("box plots for numeric column")
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.show()

st.pyplot(plt)

#Create box plot for the household_size column
st.subheader("box plots for household_size")
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['household_size'])
plt.title('Box Plot for household_size')
plt.show()

st.pyplot(plt)

#remove outliers from numeric columns
for column in numeric_columns:
    # Calculate the first quartile (Q1) and the third quartile (Q3)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    # Calculate the interquartile range (IQR
    IQR = Q3 - Q1

    # Set lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter rows where value is within limits
    df1 = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Display the bounds and number of rows removed
    st.write(f"Outliers for {column}:")
    st.write(f"Lower Bound: {lower_bound}")
    st.write(f"Upper Bound: {upper_bound}")
    st.write(f"Number of rows removed: {df[column].isna().sum()}")

# Check dataframe without outliers
df1.reset_index(drop=True, inplace=True)
st.subheader("data without outliers")

df1.head()
st.subheader("Data information")

buffer = io.StringIO()
df1.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

#4-Encode categorical features

#Show categorical features
st.subheader("categorial_columns")
categorial_columns = df1.select_dtypes(include=['object']).columns
st.write (categorial_columns)

#Show categorical value counts
st.subheader("categorial value columns")

st.subheader("country value columns")
country_counts = df1['country'].value_counts()
st.write (country_counts)

st.subheader("uniqueid value columns")
uniqueid_counts = df1['uniqueid'].value_counts()
st.write (uniqueid_counts)

st.subheader("bank_account")
bank_account_counts = df1['bank_account'].value_counts()
st.write (bank_account_counts)

st.subheader("location_type value columns")
location_type_counts = df1['location_type'].value_counts()
st.write (location_type_counts)

st.subheader("cellphone_access value columns")
cellphone_access_counts = df1['cellphone_access'].value_counts()
st.write (cellphone_access_counts)

st.subheader("gender_of_respondent value columns")
gender_of_respondent_counts = df1['gender_of_respondent'].value_counts()
st.write (gender_of_respondent_counts)

st.subheader("relationship_with_head value columns")
relationship_with_head_counts = df1['relationship_with_head'].value_counts()
st.write (relationship_with_head_counts)

st.subheader("marital_status value columns")
marital_status_counts = df1['marital_status'].value_counts()
st.write (marital_status_counts)

st.subheader("education_level value columns")
education_level_counts = df1['education_level'].value_counts()
st.write (education_level_counts)

st.subheader("job_type value columns")
job_type_counts = df1['job_type'].value_counts()
st.write (job_type_counts)

#encode coulumns
st.subheader("Encode categorical columns")
label_encoder = LabelEncoder()
categorial_columns=['country', 'bank_account','location_type', 'cellphone_access', 'gender_of_respondent','relationship_with_head', 'marital_status','education_level', 'job_type']

for column in categorial_columns:
    df1[column]= label_encoder.fit_transform(df1[column])

    st.write(f"Encoded values for columns{column}:")
st.write(df1[column].head())
buffer = io.StringIO()
df1.info(buf=buffer)
s = buffer.getvalue()
st.text(s)


#Display the encoded data
st.subheader("Encoded data")
df_encoded = df1
st.write(df_encoded.head())

buffer = io.StringIO()
df_encoded.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

#III-Based on the previous data exploration train and test a machine learning classifier
st.subheader('Prepare data for the modelling phase')

#1-Select your target variable and the features
st.subheader('Select your target and features')
target = 'bank_account'
features=['country','location_type', 'cellphone_access', 'gender_of_respondent','relationship_with_head', 'marital_status','education_level', 'job_type']

st.write("variable X")
X=df_encoded[features]
st.dataframe(X)

st.write("variable y")
y=df_encoded[target]
st.dataframe(y)

#splitting data with test size of 30%
from sklearn.model_selection import train_test_split
st.write("splitting data with test size of 30%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.write("X_train")
st.write("X_test")
st.write("y_train")
st.write("y_test")

#Apply a random forest
st.subheader('Apply a random forest model')

from sklearn.ensemble import RandomForestClassifier

#Creating a random forest
randomforest = RandomForestClassifier(n_estimators=20, random_state=42)

#Training the model
randomforest.fit(X_train, y_train)

#testing the model
y_pred = randomforest.predict(X_test)

# Checking the accuracy
accuracy = randomforest.score(X_test, y_test)
st.write("Random Forest model accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

#IV-Add input fields for your features and a validation button at the end of the form
st.title('Bank account predect')

# Add numeric input fields
st.subheader("Enter Customer information")

country_input = st.number_input("Enter the country code", min_value=0, max_value=3)
location_type_input = st.number_input("Enter the location type code", min_value=0, max_value=1)
cellphone_access_input = st.number_input("Enter the cellphone access code", min_value=0, max_value=1)
gender_input = st.number_input("Enter the gender code", min_value=0, max_value=1)
relationship_input = st.number_input("Enter the relationship with head code", min_value=0, max_value=5)
marital_status_input = st.number_input("Enter the marital status code", min_value=0, max_value=4)
education_level_input = st.number_input("Enter the education level code", min_value=0, max_value=5)
job_type_input = st.number_input("Enter the job type code", min_value=0, max_value=9)

# Splitting the data with a test size of 30%
st.write("Splitting the data into training and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import and train a RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
st.subheader('Training the RandomForest Classifier')

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Add a validation button
if st.button('Validate'):
    input_data= pd.DataFrame([[country_input,location_type_input,cellphone_access_input,gender_input,relationship_input,marital_status_input,education_level_input,job_type_input]], columns=features)
        # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the prediction result
    result = "Has a bank account" if prediction[0] == 1 else "Does not have a bank account"
    st.write("Prediction result:", result)

