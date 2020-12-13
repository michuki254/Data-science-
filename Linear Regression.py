#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# In[2]:


data = pandas.read_csv('cost_revenue_clean.csv')


# In[3]:


data.describe()


# In[4]:


3.29e7


# In[5]:


X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])


# In[6]:


plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()


# In[7]:


regression = LinearRegression()
regression.fit(X, y)


# Slope coefficient:

# In[8]:


regression.coef_    # theta_1


# In[9]:


#Intercept
regression.intercept_


# In[17]:


plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)

# Adding the regression line here:
plt.plot(X, regression.predict(X), color='red', linewidth=3)

plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()


# In[12]:


#Getting r square from Regression
regression.score(X, y)


# In[11]:




