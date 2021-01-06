#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


# In[21]:


df = pd.read_csv("Mall_Customers.csv")


# In[22]:


df.head()


# In[23]:


x = df[["Annual Income (k$)", "Spending Score (1-100)"]].values


# In[24]:


plt.scatter(x[:, 0], x[:,1])
plt.show()


# In[28]:


sse = []
X = []
for i in range(2, 10):
    model = KMeans(n_clusters=i)
    model = model.fit(x)
    sse.append(model.inertia_)
    X.append(i)


# In[29]:


plt.plot(X, sse)
plt.show()


# In[30]:


model = KMeans(n_clusters=5)
pred = model.fit_predict(x)


# In[31]:


df["Target"] = pred


# In[35]:


centroid = model.cluster_centers_


# In[36]:


centroid


# In[41]:


for i in range(5):
    data = df[df.Target == i].values
    data = data[:, -3:-1]
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x,y)
    plt.scatter(centroid[i][0], centroid[i][1], color = "g", marker="*", s = 300)


# In[ ]:




