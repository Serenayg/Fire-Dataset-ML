#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:

df = pd.read_csv("data/clean_fire_dataset.csv")
df.info()


# In[3]:


X = df.drop("STATUS", axis=1)
y = df["STATUS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_test.shape , y_test.shape


# In[4]:

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
)

model.fit(X_train, y_train)


# In[6]:


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train Accuracy:", round(model.score(X_train, y_train), 3))
print("Test Accuracy :", round(model.score(X_test, y_test), 3))



import joblib

joblib.dump(model, "model.pkl")


