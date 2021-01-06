#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("Online Retail.csv")


# In[4]:


df.info()


# In[5]:


df = df.dropna(subset=["CustomerID"])


# In[6]:


df.info()


# In[7]:


df.keys()


# In[8]:


df.head()


# In[ ]:


df = df.loc[df["Quantity"] > 0]


# In[ ]:


df.shape


# In[ ]:


df["Sales"] = (df["Quantity"] * df["UnitPrice"]).values


# In[47]:


## Convert InvoiceDate from Object Type to Date Type
df.InvoiceDate = pd.to_datetime(df.InvoiceDate)


# ### RFM - Recency, Frequence and Monetary

# ##### 1. Monetary Value

# In[54]:


df.shape


# In[55]:


df.CustomerID.nunique()


# In[68]:


Monetry = df.groupby("CustomerID").Sales.sum()
Monetry = Monetry.reset_index()
Monetry


# ##### 1. Frequency Value

# In[69]:


df.keys()


# In[70]:


frequency = df.groupby("CustomerID").InvoiceNo.count()
frequency = frequency.reset_index()
frequency


# In[75]:


MaxDate = max(df.InvoiceDate) + pd.DateOffset(days = 1)
MaxDate


# In[76]:


df["Difference"] = MaxDate - df.InvoiceDate


# In[78]:


recency = df.groupby("CustomerID").Difference.min()
recency = recency.reset_index()
recency


# In[79]:


recency.Difference = recency.Difference.dt.days


# In[80]:


recency


# In[81]:


MF = Monetry.merge(frequency, on="CustomerID")


# In[83]:


RFM = MF.merge(recency, on="CustomerID")


# In[85]:


RFM.columns = ["CustomerID", "Sales", "Frequency", "Recency"]


# In[86]:


RFM


# In[87]:


## Standard Scaler Value


# In[88]:


from sklearn.preprocessing import StandardScaler
scaller = StandardScaler()


# In[93]:


# OUT = (x-mean)/std(data)


# In[89]:


RFM_New = scaller.fit_transform(RFM)


# <p style="color:darkblue; font-size:30px">Apply K-Means with Some K</p>

# In[96]:


from sklearn.cluster import KMeans


# In[145]:


SSD = []
K = []
for i in range(1, 10):
    model = KMeans(n_clusters=i)
    model.fit(RFM_New)
    SSD.append(model.inertia_)
    K.append(i)


# In[146]:


plt.plot(K, SSD, c ="darkblue", lw = 2)
plt.scatter(K, SSD, marker="*", s = 100, c = "darkblue")
plt.show()


# In[110]:


model = KMeans(n_clusters=5)
ClusterId = model.fit_predict(RFM_New)


# In[111]:


RFM["ClusterId"] = ClusterId


# In[113]:


RFM.ClusterId.unique()


# In[117]:


RFM.head()


# In[119]:


All_Sales = RFM.groupby("ClusterId").Sales.mean()
All_Frequency = RFM.groupby("ClusterId").Frequency.mean()
All_Recency = RFM.groupby("ClusterId").Recency.mean()


# In[127]:


final_data = pd.DataFrame({
    "Sales": All_Sales,
    "Frequency":All_Frequency,
    "Recency":All_Recency
})


# In[128]:


final_data


# In[129]:


## Plot Pie Plot for all_data


# In[150]:


fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(1, 3, 1)
plt.title("Sales Mean")
ax.pie(final_data.Sales, labels = [0,1,2,3,4], radius = 1.2)


ax = fig.add_subplot(1, 3, 2)
plt.title("Frequency Mean")
ax.pie(final_data.Frequency, labels = [0,1,2,3,4], radius = 1.2)

ax = fig.add_subplot(1, 3, 3)
plt.title("Recency Mean")
ax.pie(final_data.Recency, labels = [0,1,2,3,4], radius = 1.2)


plt.show()


# In[158]:


df2 = RFM[RFM.ClusterId == 1]


# In[160]:


df2.to_csv("My_First_Cluster.csv")


# In[ ]:




