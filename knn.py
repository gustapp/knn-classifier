#%% [markdown]
# # K-Nearest Neighbors Classifier for adult dataset

#%%
# Load Data
import pandas as pd
import numpy as np

def pre_process_data(data_file_path):
    df = pd.read_csv(data_file_path)
    df.head()

    #%%
    # Import LabelEncoder
    from sklearn import preprocessing
    #creating labelEncoder
    le = preprocessing.LabelEncoder()

    #%%
    # Converting string labels into numbers.
    workclass_encoded=le.fit_transform(df['workclass'])
    education_encoded=le.fit_transform(df['education'])
    marital_status_encoded=le.fit_transform(df['marital-status'])
    occupation_encoded=le.fit_transform(df['occupation'])
    relationship_encoded=le.fit_transform(df['relationship'])
    race_encoded=le.fit_transform(df['race'])
    sex_encoded=le.fit_transform(df['sex'])
    native_country_encoded=le.fit_transform(df['native-country'])

    #%%
    #combinig weather and temp into single listof tuples
    return (list(zip(df['age'],workclass_encoded, df['fnlwgt'], education_encoded, df['education-num'], marital_status_encoded, occupation_encoded, relationship_encoded, race_encoded, sex_encoded, df['capital-gain'], df['capital-loss'], df['hours-per-week'], native_country_encoded)), le.fit_transform(df['income']))


#%%
#combinig weather and temp into single listof tuples
X_train, y_train=pre_process_data('./adult/adult.data')

#%%
# Create KNN model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=11)

#%%
# Train the model using the training sets
model.fit(X_train, y_train)

#%%
#Predict Output from training set
predicted= model.predict([X_train[0]]) # 0:>50K, 1:<=50K
print(predicted)

#%% [markdown]
# Evaluation

#%%
# Load Test Data
X_test, y_test=pre_process_data('./adult/adult.test')

#%%
# Predict for the test dataset
y_pred = model.predict(X_test)

#%%
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))