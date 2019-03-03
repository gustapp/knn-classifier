#%% [markdown]
# # K-Nearest Neighbors Classifier for adult dataset

#%%
# Import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def pre_process_data(data_file_path):

    #%%
    # Categorical features.
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    #%%
    # Load data from file
    df = pd.read_csv(data_file_path)
    df.head()

    #%% [markdown]
    # ## Input missing values

    #%%
    # Count number of missing values `?`
    df = df.replace(' ?', np.nan)
    df.isnull()

    #%%
    # Fill missing values with mode for each column
    mode = df[categorical_features].mode()

    df[categorical_features] = df[categorical_features].fillna(mode.iloc[0])

    #%%
    # Check if all missing values were inputed
    df.isnull() # expected to be `0`

    #%% [markdown]
    # ## Encode categorical features

    #%%
    #creating labelEncoder
    le = LabelEncoder()

    df[categorical_features] = df[categorical_features].apply(le.fit_transform)

    target = le.fit_transform(df.pop('income'))

    features = [tuple(value) for value in df.values]

    #%% [markdown]
    # ## Scale the features

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    #%%
    #combinig all features into single listof tuples
    return (features_scaled, target)



#%%
#combinig weather and temp into single listof tuples
X_train, y_train=pre_process_data('./adult/adult.data')

#%%
# Create KNN model
# Import K-Nearest Neighbors classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

# @param: n_neighbors `K: hyperparameter`
model = KNeighborsClassifier(n_neighbors=2)

#%%
# Train the model using the training sets
model.fit(X_train, y_train)

#%% [markdown]
# Evaluation

#%%
# Load Test Data
X_test, y_test=pre_process_data('./adult/adult.test')

#%%
# Predict for one instance
y_pred = model.predict([X_test[0]])
print(y_pred) # <=50K: 0 | >50K: 1

#%%
# Predict for the test dataset
y_pred = model.predict(X_test)

#%%
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%% [markdown]
# ## Evaluate model for different `K`

#%%
# Import libraries
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, accuracy_score
from math import sqrt
import matplotlib.pyplot as plt

#%%
# Retrive empirical error rates for different k
rmse_val = []
accy_val = []
for K in range(1, 100, 2):
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    y_pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    rmse_val.append(error) #store rmse values

    model = neighbors.KNeighborsClassifier(n_neighbors = K)
    model.fit(X_train, y_train)  #fit the model
    y_pred=model.predict(X_test) #make prediction on test set
    accuracy = accuracy_score(y_test, y_pred)
    accy_val.append(accuracy)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    print('RMSE value for k= ' , K , 'is:', error, '\n')

#plotting the rmse values against k values
rmse_curve = pd.DataFrame(rmse_val) #elbow curve 
rmse_curve.plot()
rmse_curve.to_csv('./rmse_curve.csv')

#plotting the accuracy values against k values
accy_curve = pd.DataFrame(accy_val)
accy_curve.plot()
accy_curve.to_csv('./accy_curve.csv')
