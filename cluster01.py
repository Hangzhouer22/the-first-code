#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Creating the dummy dataset


# In[2]:


get_ipython().system('pip install scikit-learn-extra')


# In[3]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')    # To get rid of warning messages

from sklearn import datasets         # To create dummy dataset

from sklearn.cluster import KMeans   

from sklearn_extra.cluster import KMedoids

from sklearn.mixture import GaussianMixture

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN

# Remove scientific notations and display numbers with 2 decimal points instead
pd.options.display.float_format = '{:,.2f}'.format        

# Update default background style of plots
sns.set_style(style='darkgrid')


# In[4]:


np.random.seed(1)   # Setting the seed to get reproducible results

conc_circles = datasets.make_circles(n_samples = 2000, factor = .5, noise = .05)


# In[5]:


X, y = conc_circles         # Separating the features and the labels


# In[6]:


df = pd.DataFrame(X)

df.columns = ['X1', 'X2']

df['Y'] = y

df


# In[ ]:


#Visualizing the data


# In[7]:


# Scatter plot of original lables
sns.scatterplot(x = 'X1', y = 'X2', data = df, hue = 'Y')

plt.show()


# In[ ]:


#The above scatter plot shows two concentric circles, each belonging to a different class. 
#The objective is to visualize the clusters we get from different clustering algorithms by
#using the features X1 and X2, and see #how well each clustering algorithm can perform in 
#terms of identifying the underlying pattern of concentric circles.


# In[8]:


from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)


# In[9]:


kmeans = KMeans(n_clusters = 2, random_state = 12)

kmeans.fit(X)

df['KmeansLabels'] = kmeans.predict(X)

sns.scatterplot(x = 'X1', y = 'X2', data = df, hue = 'KmeansLabels')

plt.show()


# In[ ]:


#The K-means clustering algorithm is not able to identify the original clusters in the data.


# In[10]:


kmedo = KMedoids(n_clusters = 2, random_state = 12)

kmedo.fit(X)

df['KMedoidLabels'] = kmedo.predict(X)

sns.scatterplot(x = 'X1', y = 'X2', data = df, hue = 'KMedoidLabels')

plt.show()


# In[11]:


gmm = GaussianMixture(n_components = 2, random_state = 12)

gmm.fit(X) 

df['GmmLabels'] = gmm.predict(X)

sns.scatterplot(x = 'X1', y = 'X2', data = df, hue = 'GmmLabels')

plt.show()


# In[12]:


aglc = AgglomerativeClustering(n_clusters = 2, linkage = 'single')

df['AggLabels'] = aglc.fit_predict(X)

sns.scatterplot(x = 'X1', y = 'X2', data = df, hue = 'AggLabels')

plt.show()


# In[13]:


dbs = DBSCAN(eps = 0.3)

df['DBSLabels'] = dbs.fit_predict(X)

sns.scatterplot(x = 'X1', y = 'X2', data = df, hue = 'DBSLabels')

plt.show()


# In[ ]:




