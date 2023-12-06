#   LINEAR REGRESSION   #
#import libraries
import numpy as np
import matplotlib.pyplot as mtp
import pandas
#Load data
data_set= pd.read_csv('Salary_dataset.csv')
print(data_set.describe())
# print data
print("Dataset")
df=pd.DataFrame(data_set)
print(df.to_string())
# Checking for null values.
df.isnull().sum()
# Pick columns for x .
X= data_set.iloc[:,1].values
print(X)
# Pick columns for y.
y = df.iloc[:, 2].values
print(y)
x=X.reshape(-1,1)
print(x)
#load dataset slicing module and  Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= .2, random_state=0)
#load liniear regression class
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
#create an instance of linear regression.
regressor.fit(x_train,y_train)
x_pred= regressor.predict(x_train)
print("Prediction result on Test Data")
y_pred = regressor.predict(x_test)
dfs=pd.DataFrame(x_test)
print("X-test")
print(dfs)
df2 = pd.DataFrame({'Actual Y-Data': y_test,'Predicted Y-Data': y_pred})
print(df2.to_string())
#visualizing the Test set results
mtp.scatter(x_train, y_train, color="blue")
mtp.plot(x_train, x_pred, color="red")
mtp.title("Salary and Experience (Training Dataset)")
mtp.xlabel("Experience")
mtp.ylabel("Salary")
mtp.show()
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")
mtp.scatter(y_test,y_pred,c="pink")
mtp.xlabel('y test')
mtp.ylabel('predicted y')
mtp.grid()