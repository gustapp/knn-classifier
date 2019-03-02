#%% [markdown]
# # K-Nearest Neighbors Classifier for adult dataset

#%%
# Load Data
import pandas as pd
import numpy as np

df = pd.read_csv('./adult/adult.data')
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

income_encoded=le.fit_transform(df['income'])

#%%
#combinig weather and temp into single listof tuples
# features=list(zip(df['age'],workclass_encoded, df['fnlwgt'], education_encoded, df['education-num'], marital_status_encoded, occupation_encoded, relationship_encoded, race_encoded, sex_encoded, df['capital-gain'], df['capital-loss'], df['hours-per-week'], native_country_encoded))
features=list(zip(df['age'],workclass_encoded, df['fnlwgt'], education_encoded, df['education-num'], marital_status_encoded, occupation_encoded, relationship_encoded, race_encoded, sex_encoded, df['capital-gain'], df['capital-loss'], df['hours-per-week'], native_country_encoded))

#%%
# Create KNN model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=500)

#%%
# Train the model using the training sets
model.fit(features,income_encoded)

#%%
#Predict Output from training set
predicted= model.predict([features[0]]) # 0:>50K, 1:<=50K
print(predicted)

#%% [markdown]
# Evaluation

#%%
# Load Test Data
import pandas as pd

df = pd.read_csv('./adult/adult.test')
df.head()

#%%
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

#%%
# Converting string labels into numbers.
workclass_encoded_test=le.fit_transform(df['workclass'])
education_encoded_test=le.fit_transform(df['education'])
marital_status_encoded_test=le.fit_transform(df['marital-status'])
occupation_encoded_test=le.fit_transform(df['occupation'])
relationship_encoded_test=le.fit_transform(df['relationship'])
race_encoded_test=le.fit_transform(df['race'])
sex_encoded_test=le.fit_transform(df['sex'])
native_country_encoded_test=le.fit_transform(df['native-country'])

y_test=le.fit_transform(df['income'])

#%%
#combinig weather and temp into single listof tuples
# features=list(zip(df['age'],workclass_encoded_test, df['fnlwgt'], education_encoded_test, df['education-num'], marital_status_encoded_test, occupation_encoded_test, relationship_encoded_test, race_encoded_test, sex_encoded_test, df['capital-gain'], df['capital-loss'], df['hours-per-week'], native_country_encoded_test))
X_test=list(zip(df['age'],workclass_encoded_test, df['fnlwgt'], education_encoded_test, df['education-num'], marital_status_encoded_test, occupation_encoded_test, relationship_encoded_test, race_encoded_test, sex_encoded_test, df['capital-gain'], df['capital-loss'], df['hours-per-week'], native_country_encoded_test))

#%%
# Predict for the test dataset
y_pred = model.predict(X_test)

#%%
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))