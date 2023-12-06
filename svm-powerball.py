#    SVM   #
#Importing Libraries
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
#Load the data
data_set= pd.read_csv('POWERBALL.csv')
df=pd.DataFrame(data_set)
print(df.to_string())
#To display the first five rows of the data.
df.head()
#To determine the dimensions of a Data.
df.shape
df.info()
#To generate summary statistics for the numerical columns in a Data.
df.describe()
# Checking for null values.
df.isnull().sum()
# Checking for duplicate values.
df.duplicated().sum()
#Extracting Independent and dependent Variables.
x= df.iloc[:,4:11].values
y= df.iloc[:, 11].values
print(x)
print(y)
#Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
#Feature Scaling.
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
from sklearn.svm import SVC
 # "Support vector classifier"
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)
#Predicting the test set result.
y_pred= classifier.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)*100
print(accuracy)
df2=pd.DataFrame({"Actual Y_Test":y_test,"PredictionData":y_pred})
print("prediction status")
print(df2.to_string())
