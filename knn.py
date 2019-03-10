#%% [markdown]
# # K-Nearest Neighbors Classifier for adult dataset

#%%
# Import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def pre_process_data(data_file_path):

    # Categorical features.
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    # Load data from file
    df = pd.read_csv(data_file_path)
    df.head()

    # Input missing values
    # Count number of missing values `?`
    df = df.replace(' ?', np.nan)
    df.isnull()

    # Fill missing values with mode for each column
    mode = df[categorical_features].mode()

    df[categorical_features] = df[categorical_features].fillna(mode.iloc[0])

    # Check if all missing values were inputed
    df.isnull() # expected to be `0`

    # Encode categorical features
    # creating labelEncoder
    le = LabelEncoder()

    df[categorical_features] = df[categorical_features].apply(le.fit_transform)

    target = le.fit_transform(df.pop('income'))

    features = [tuple(value) for value in df.values]

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # combinig all features into single listof tuples
    return (features_scaled, target)

#%%
# Preprocess Train data
X_train, y_train=pre_process_data('./adult/adult.data')

#%% [markdown]
# Evaluation

#%%
# Preprocess Test Data
X_test, y_test=pre_process_data('./adult/adult.test')

#%% [markdown]
# ## Evaluate model for different `K`

#%%
# Import libraries
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from math import sqrt
import matplotlib.pyplot as plt

#%%
# Retrive empirical error rates for different k
metrics_val = []
k_index =  []
for K in range(1, 100, 2):
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    y_pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse

    model = neighbors.KNeighborsClassifier(n_neighbors = K)
    model.fit(X_train, y_train)  #fit the model
    y_pred=model.predict(X_test) #make prediction on test set

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    k_index.append(K)
    metrics_val.append([error, accuracy, precision, recall, f1])

    print('K: ', K)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('RMSE:', error, '\n')

#%%
# Check metrics
metrics_curve = pd.DataFrame(data=metrics_val, columns=['error', 'accuracy', 'precision', 'recall', 'f1-score'], index=k_index)
metrics_curve.head()
metrics_curve.to_csv('./metrics_curve.csv')

#%%
# Plotting the quality metrics for different K's
plt.style.context('default')

rmse_ax = metrics_curve.plot(y=['error'], color='red', legend=False, title='Model Evaluation-Quality Metrics')
rmse_ax.set_ylabel('Root Mean Squared Error (rmse)', color='black')
rmse_ax.set_xlabel('Number of Neghbors (K)')

accy_ax = rmse_ax.twinx()
metrics_curve.plot(y=['accuracy'], color='magenta', legend=False, secondary_y=True, ax=accy_ax)
plt.ylabel('Accuracy Score')

f1_ax = accy_ax.twinx()
f1_ax.set_ylabel('F1 Score')

rspine = f1_ax.spines['right']
rspine.set_position(('axes', 1.15))
metrics_curve.plot(y=['f1-score'], color='blue', legend=False, ax=f1_ax)

f1_ax.legend([rmse_ax.get_lines()[0], accy_ax.right_ax.get_lines()[0], f1_ax.get_lines()[0]],\
           ['error','accuracy','f1-score'], bbox_to_anchor=(0.95, 0.5))

#%%
# Plot confusion matrix
from sklearn_examples import plot_confusion_matrix

np.set_printoptions(precision=2)

class_names = np.array(['<=50K', '>50K'])

#%%
# Plot confusion matrix whithout normalization
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot confusion matrix normalized
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#%%
# Create KNN model
# Import K-Nearest Neighbors classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

# @param: n_neighbors `K: hyperparameter`
model = KNeighborsClassifier(n_neighbors=27)

# Train the model using the training sets
model.fit(X_train, y_train)

# Make prediction on test set
y_pred=model.predict(X_test)
