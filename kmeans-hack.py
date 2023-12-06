#    KMEANS CLUSTERING   #
#Importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('hack_data.csv')
df=pd.DataFrame(dataset)
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
x = df.iloc[:, [1, 6]].values
x
from sklearn.cluster import KMeans
wcss_list= [] #Initializing the list for the values of WCSS
#Using for loop for iterations from 1 to 10.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
print(wcss_list)
mtp.plot(range(1, 11), wcss_list)
mtp.title('The Elobw Method Graph')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('wcss_list')
mtp.show()
#training the K-means model on a dataset
kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(x)
print(y_predict)
mtp.scatter(x[y_predict==0,0],x[y_predict==0,1],s=50,c='red',label='cluster 1')
mtp.scatter(x[y_predict==1,0],x[y_predict==1,1],s=50,c='blue',label='cluster 2')
mtp.scatter(x[y_predict==2,0],x[y_predict==2,1],s=50,c='green',label='cluster 3')
mtp.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='yellow',label='centroids')
mtp.title('Clusters of Customers')
mtp.xlabel('Annual Income')
mtp.ylabel('Spending Score(1-100)')
mtp.legend()
