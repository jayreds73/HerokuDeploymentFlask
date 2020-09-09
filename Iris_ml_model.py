import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read Data
data = pd.read_csv('iris.csv')
print(data.head())

# Assign to X and y
X= data.iloc[:,:-1]
#print(X.head())
y=data.target
#print(y.head())

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train.head())

# Create model
model = RandomForestClassifier()
model.fit(X_train,y_train)

# model prediction
y_pred = model.predict(X_test)

# model Evaluation
c_matrix=pd.crosstab(y_test,y_pred)
print(c_matrix)
print(accuracy_score(y_test,y_pred))

# create pickle dump
pickle_out = open("model.pkl",'wb')
pickle.dump(model,pickle_out)
pickle_out.close()
