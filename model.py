import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle


df = pd.read_csv("Mall_Customers.csv") # Heart-disease
df.drop(["CustomerID"], axis =1 , inplace = True)
X1=df.loc[:, ["Age", "Spending Score (1-100)"]].values



# split into train and test 
#Initialize the class object
kmeans = KMeans (n_clusters= 4)
#predict the Labels of clusters.

model = kmeans.fit_predict(X1)
# Pickle
pickle.dump(model,open("model.pkl","ab"))    