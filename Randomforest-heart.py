#   RANDOM FOREST   #
# Importing libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing datasets.
data=pd.read_csv('heart.csv')
df=pd.DataFrame(data)
print(df)
print("Actual Dataset")
print(df.to_string())
#To display the first five rows of the data.
df.head()
#To determine the dimensions of a Data.
df.shape
#To generate summary statistics for the numerical columns in a Data.
df.describe()
# Checking for null values.
df.isnull().sum()
# Checking for duplicate values.
df.duplicated().sum()
print(df.info())
#Extracting Independent and dependent Variables.
x = df.iloc[:,0:13].values
y=  df.iloc[:,13].values
# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
#Fitting Decision Tree classifier  to the training set.
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")
classifier.fit(x_train, y_train)
#Predicting the test set result.
y_pred= classifier.predict(x_test)
print("------------PREDICTION----------")
df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2.to_string())
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Evaluate predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))