import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv("heart-disease.csv") # Heart-disease

# split data into train and test
x = df.drop("target" , axis = 1)
y = df["target"]
np.random.seed(42)

# split into train and test 
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2)


#Initialize the class object
model = LogisticRegression(max_iter = 120)
model.fit(x_train , y_train)

# Pickle
pickle.dump(model,open("model.pkl","ab"))    
