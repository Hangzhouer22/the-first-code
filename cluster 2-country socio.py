#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

# To scale the data using z-score 
from sklearn.preprocessing import StandardScaler

# Importing clustering algorithms
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

from sklearn_extra.cluster import KMedoids

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN

# Silhouette score
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("/Users/yutaoyan/Desktop/Sociaecon/Country.csv")

data.head()


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


data[data.duplicated()]


# In[6]:


data.describe().T


# In[7]:


for col in data.columns[1:]:
    print(col)
    
    print('Skew :', round(data[col].skew(), 2))
    
    plt.figure(figsize = (15, 4))
    
    plt.subplot(1, 2, 1)
    
    data[col].hist(bins = 10, grid = False)
    
    plt.ylabel('count')
    
    plt.subplot(1, 2, 2)
    
    sns.boxplot(x = data[col])
    
    plt.show()


# In[8]:


plt.figure(figsize  = (10, 10))

sns.heatmap(data.corr(), annot = True, cmap = "YlGnBu")

plt.show()


# In[9]:


#There is a strong positive correlation between gdpp and income. This makes sense.
#The life expectancy is positively correlated with gdpp. This indicates that people live longer 
#in richer countries.
#There is a strong negative correlation between life expectancy and child mortality. This is understandable.
#The child mortality is also seen to have a strong positive correlation with the fertility rate.


# In[10]:


#Scaling the data¶
#Clustering algorithms are distance-based algorithms, and all distance-based algorithms 
#are affected by the scale of the variables. Therefore, we will scale the data before applying clustering.
#We will drop the variables 'country' variable because it is unique for each country and would not 
#add value to clustering.
#We will also drop the 'gdpp' variable for now, because we want to see if we can identify clusters of 
#countries without relying on GDP and see later if these clusters correspond to an average GDP value 
#for the countries in each cluster.


# In[11]:


data_new = data.drop(columns = ["country", "gdpp"])


# In[12]:


# Scaling the data and storing the output as a new DataFrame

scaler = StandardScaler()

data_scaled = pd.DataFrame(scaler.fit_transform(data_new), columns = data_new.columns)

data_scaled.head()


# In[13]:


# Creating copy of the data to store labels from each algorithm
data_scaled_copy = data_scaled.copy(deep = True)


# In[14]:


# Empty dictionary to store the SSE for each value of K
sse = {} 

# Iterate for a range of Ks and fit the scaled data to the algorithm. 
# Use inertia attribute from the clustering object and store the inertia value for that K 
for k in range(1, 10):
    kmeans = KMeans(n_clusters = k, random_state = 1).fit(data_scaled)
    
    sse[k] = kmeans.inertia_

# Elbow plot
plt.figure()

plt.plot(list(sse.keys()), list(sse.values()), 'bx-')

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()


# In[ ]:


#Observations:

#We can see from the plot that there is a consistent dip from 2 to 8 and there doesn't seem 
#to be a clear 'elbow' here. We may choose any number of clusters from 2 to 8.
#So, let's look at another method to get a 'second opinion'. Let's create a plot with Silhouette 
#scores to see how it varies with K.


# In[15]:


# Empty dictionary to store the Silhouette score for each value of K
sc = {} 

# Iterate for a range of Ks and fit the scaled data to the algorithm. Store the Silhouette score for that K 
for k in range(2, 10):
    kmeans = KMeans(n_clusters = k, random_state = 1).fit(data_scaled)
    
    labels = kmeans.predict(data_scaled)
    
    sc[k] = silhouette_score(data_scaled, labels)

# Elbow plot
plt.figure()

plt.plot(list(sc.keys()), list(sc.values()), 'bx-')

plt.xlabel("Number of cluster")

plt.ylabel("Silhouette Score")

plt.show()


# In[16]:


#Observation:

#We observe from the plot that the silhouette score is the highest for K=3. Let's first
#understand these 3 clusters.


# In[17]:


kmeans = KMeans(n_clusters = 3, random_state = 1)

kmeans.fit(data_scaled)

# Adding predicted labels to the original data and the scaled data 
data_scaled_copy['KMeans_Labels'] = kmeans.predict(data_scaled)

data['KMeans_Labels'] = kmeans.predict(data_scaled)


# In[18]:


data['KMeans_Labels'].value_counts()


# In[19]:


#Observation:

#This looks like a very skewed clustering, with only three observations in one cluster and 
#more than a hundred in another. Let's check out the profiles of these clusters.


# In[20]:


# Calculating the mean and the median of the original data for each label
mean = data.groupby('KMeans_Labels').mean()

median = data.groupby('KMeans_Labels').median()

df_kmeans = pd.concat([mean, median], axis = 0)

df_kmeans.index = ['group_0 Mean', 'group_1 Mean', 'group_2 Mean', 'group_0 Median', 'group_1 Median', 'group_2 Median']

df_kmeans.T


# In[21]:


#Observations:
#It looks like Cluster 2 belongs to high income countries which also have high gdpp.
#Cluster 1 seems to be of low income countries, with low mean gdp as well.
#The remaining countries are in Cluster 0 which also happens to be the biggest cluster. 
#Since the number of developing countries is larger than the group of highly developed 
#countries, this intuitively makes sense.


# In[ ]:


#Let us now visualize the summary statistics of these clusters below.


# In[22]:


cols_visualise = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for col in cols_visualise:
    sns.boxplot(x = 'KMeans_Labels', y = col, data = data)
    plt.show()


# In[23]:


#Cluster Profiles:

#Cluster 2 has only 3 observations. As observed from the scatter plots and the boxplots, this group 
#consists of outlier high income countries with the highest percentages of imports and exports in terms of GDP.
#Cluster 1 seems to have countries with less desirable values for many indicators. These countries seem 
#to have the highest inflation rates, the lowest GDP per capita, the lowest exports as well as 
#imports - all signaling a very poor economic situation. These countries also have the highest child 
#mortalities, the highest fertility rates, and the lowest life expectancies. These characteristics are 
#traits of underdeveloped or developing countries. These countries also seem to have a trade deficit, 
#i.e., more imports than exports, and as a consequence, may be more reliant on borrowing and lines of
#credit to finance their economy.
#Cluster 0 is the largest cluster with traits of countries that fall in the middle of the development 
#spectrum. These countries have a comparatively better state of affairs than the countries in cluster 1. 
#However, this cluster has a large range of values, indicating that it is a mix of many different types of 
#countries. Ideally, we do not want a cluster to be like this as the fundamental idea behind clustering is 
#to 'group similar things' and this cluster seems to have a lot of 'dissimilarity' within it.
#Overall, this clustering solution does give us good insights into potential clusters of similar countries
#but is not very useful as it is impacted by outlier countries resulting in one very small cluster and 
#two very big clusters. We should try other algorithms to see if we can do better.


# In[24]:


#But before that, let's validate if these clusters relate well with the GDP of the country.


# In[25]:


cols_visualise = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for col in cols_visualise:
    sns.scatterplot(x = col, y = 'gdpp', data = data, hue = 'KMeans_Labels', palette = 'Dark2')
    
    plt.show()


# In[26]:


#Observations:

#The countries with higher fertility rates also seem to have higher populations, corresponding
#with lower per capita income in these countries.
#The child mortality also seems to be negatively correlated with the GDP of the country. 
#The high child mortality in such countries could be due to several reasons such as high poverty or 
#lower net income per person and a relative lack of health facilities among others.


# In[27]:


#Let's try another algorithm


# In[28]:


#K-Medoids Clustering


# In[29]:


kmedo = KMedoids(n_clusters = 3, random_state = 1)

kmedo.fit(data_scaled)

data_scaled_copy['kmedoLabels'] = kmedo.predict(data_scaled)

data['kmedoLabels'] = kmedo.predict(data_scaled)


# In[30]:


data.kmedoLabels.value_counts()


# In[31]:


# Calculating the mean and the median of the original data for each label
original_features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

mean = data.groupby('kmedoLabels').mean()

median = data.groupby('kmedoLabels').median()

df_kmedoids = pd.concat([mean, median], axis = 0)

df_kmedoids.index = ['group_0 Mean', 'group_1 Mean', 'group_2 Mean', 'group_0 Median', 'group_1 Median', 'group_2 Median']

df_kmedoids[original_features].T


# In[32]:


#Observations:

#It looks like Cluster 0 belongs to high income countries, Cluster 2 has poorer countries with low 
#incomes, and the remaining countries are in Cluster 1, which happens to be the biggest cluster as well.


# In[33]:


for col in cols_visualise:
    sns.boxplot(x = 'kmedoLabels', y = col, data = data)
    
    plt.show()


# In[34]:


#Cluster Profiles:

#Cluster 2 countries have the highest average child mortality rate, trade deficit, inflation 
#rate and least average GDP and net income per person. But the large range of values for different 
#variables implies that cluster 2 contains a variety of countries, from underdeveloped to developing ones.
#Cluster 1 shows traits of developing countries with comparatively higher GDP, net income per person 
#and significantly lower child mortality rate as compared to cluster 2. The cluster consists of 
#some outliers but majorly it consists of countries with low to medium GDP, with a comparatively 
#higher percentage of imports and exports vs GDP.
#Cluster 0 shows traits of highly developed countries with a low child mortality rate and a 
#higher net income per person, life expectancy, and GDP. These countries have the highest average
#expenditure on health as a percentage of GDP.

#Observations:

#The number of observations for each cluster from K-Medoids is more evenly distributed in comparison 
#to K-Means clustering.
#This is because the clusters from K_Medoids are less affected by outliers from the data. As we observe,
#the three outlier countries from K-Means (in terms of imports and exports) are now included in cluster
#1 and do not form a separate cluster like in K-Means.
#Unlike in K-Means, the cluster for developed countries is much bigger but still retains the overall 
#characteristics of developed countries, as reflected in the higher values for income per person,
#life expectancy, and especially in health expenditure as a percentage of GDP.


# In[35]:


#Now, let's see what we get with Gaussian Mixture Model.


# In[36]:


#Gaussian Mixture Model


# In[37]:


gmm = GaussianMixture(n_components = 3, random_state = 1)

gmm.fit(data_scaled)

data_scaled_copy['GmmLabels'] = gmm.predict(data_scaled)

data['GmmLabels'] = gmm.predict(data_scaled)


# In[38]:


data.GmmLabels.value_counts()


# In[39]:


# Calculating the mean and the median of the original data for each label
original_features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

mean = data.groupby('GmmLabels').mean()

median = data.groupby('GmmLabels').median()

df_gmm = pd.concat([mean, median], axis = 0)

df_gmm.index = ['group_0 Mean', 'group_1 Mean', 'group_2 Mean', 'group_0 Median', 'group_1 Median', 'group_2 Median']

df_gmm[original_features].T


# In[40]:


#Cluster 1 belongs to high income countries, Cluster 0 belongs to lower income countries, 
#and the rest of the countries are in Cluster 2.


# In[41]:


cols_visualise = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for col in cols_visualise:
    sns.boxplot(x = 'GmmLabels', y = col, data = data)
    
    plt.show()


# In[42]:


#Cluster Profiles:

#This clustering solution looks very similar to the once created using K-Medoids with
#one cluster of 'high income' countries, one of 'low income' and one of 'all the others'. 
#But on closer inspection, we can identify some important differences in this solution using GMM.

#Cluster 1 seems to be of 'developed' countries but this time the median values for all the key 
#indicators have all improved in comparison to the same cluster obtained from K-Medoids, with a 
#higher GDP per capita, higher income, higher exports and imports and marginally higher life 
#expectancy. At the same time, it has lower inflation rates, lower child mortality rates, 
#and lower fertility as well. Overall, we can say that this cluster has become more 'pure' in
#comparison to the one from K-Medoids.

#Cluster 0 seems to be of 'underdeveloped' countries but this time the median values for all the 
#key indicators have improved in comparison to the corresponding K-Medoids cluster. For e.g., it has 
#higher GDP per capita, higher income per person, higher exports and imports, and slightly better
#health expenditure and life expectancy. That means that this cluster of 'underdeveloped' countries has 
#become less 'pure'.

#Both of the above points can give an idea of what might have happened to the third cluster, i.e., 
#Cluster 2. It was a mix of 'underdeveloped' & 'developing' countries and continues to be so, but it 
#has gained some countries on the rich end of the spectrum, and some countries on the 'underdeveloped' 
#end have moved to the last cluster.

#Overall, this is a slightly more evenly distributed clustering solution than K-Medoids.


# In[43]:


#Hierarchical Clustering


# In[44]:


#Let's try to create clusters using Agglomerative Hierarchical clustering.
#Here, we decide the number of clusters using a concept called Dendrogram which is a tree-like 
#diagram that records the sequences of merges or splits.


# In[45]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[46]:


# The List of all linkage methods to check
methods = ['single',
           'average', 
           'complete']

# Create a subplot image
fig, axs = plt.subplots(len(methods), 1, figsize = (20, 15))

# Enumerate through the list of all methods above, get linkage and plot dendrogram
for i, method in enumerate(methods):
    Z = linkage(data_scaled, metric = 'euclidean', method = method)
    
    dendrogram(Z, ax = axs[i]);
    
    axs[i].set_title(f'Dendrogram ({method.capitalize()} Linkage)')
    
    axs[i].set_ylabel('Distance')


# In[47]:


#Observations:

#We can see that the complete linkage gives better separated clusters. A cluster is 
#considered better separated if the vertical distance connecting those clusters is higher.
#Now, we can set a threshold distance and draw a horizontal line. The number of clusters
#will be the number of vertical lines which are being intersected by the line drawn using
#the threshold.
#The branches of this dendrogram are cut at a level where there is a lot of ‘space’ to 
#cut them, that is where the jump in levels of two consecutive nodes is large
#Here, we can choose to cut it at ~9 since the space between the two nodes is largest.


# In[51]:


plt.figure(figsize = (20, 7))  
plt.title("Dendrograms")  

dend = dendrogram(linkage(data_scaled, method = 'complete'))

plt.axhline(y = 9, color = 'r', linestyle = '--')


# In[52]:


#Observations:

#We can see that the if we create a horizontal line at threshold distance ~ 9, it cuts 4 vertical 
#lines, i.e., we get 4 different clusters.
#Let's fit the algorithms using 4 as the number of clusters.


# In[53]:


# Clustering with 4 clusters
hierarchical = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'complete')

hierarchical.fit(data_scaled)


# In[54]:


data_scaled_copy['HCLabels'] = hierarchical.labels_

data['HCLabels'] = hierarchical.labels_


# In[55]:


data.HCLabels.value_counts()


# In[56]:


#Observations:

#The count of observations in the resulting 4 clusters is unevenly distributed.
#We have two clusters with only 3 countries and 1 country, respectively. Let's check the 
#countries in these clusters.


# In[57]:


# Checking 3 countries in cluster 2
data[data.HCLabels == 2]


# In[58]:


#Observations:

#Similar to K-Means, we got a separate cluster for 3 small countries with the highest values for 
#imports and exports - Luxembourg, Malta, Singapore.


# In[59]:


# Checking 1 country in cluster 3
data[data.HCLabels == 3]


# In[60]:


#Observations:

#Cluster 3 consists of just one country - Nigeria.
#Nigeria has an inflation rate of 104 which is the highest inflation rate in this dataset. 
#This might have made its distance with the other clusters significantly higher not allowing it 
#to merge with any of those data points.


# In[61]:


# Calculating the mean and the median of the original data for each label
original_features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

mean = data.groupby('HCLabels').mean()

median = data.groupby('HCLabels').median()

df_hierachical = pd.concat([mean, median], axis = 0)

df_hierachical.index = ['group_0 Mean', 'group_1 Mean', 'group_2 Mean', 'group_3 Mean', 'group_0 Median', 'group_1 Median', 'group_2 Median', 'group_3 Median']

df_hierachical[original_features].T


# In[62]:


#Observations:

#It looks like Cluster 2 has only 3 countries with high income and high gdpp, Cluster 1 has 
#low income and low gdpp countries, and the rest of the countries are in cluster 0 except 
#for one country which is in cluster 3.


# In[63]:


#Let's try to visualize the boxplots of different attributes for each cluster to see if we can spot some 
#more granular patterns.


# In[64]:


cols_visualise = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for col in cols_visualise:
    sns.boxplot(x = 'HCLabels', y = col, data = data)
    plt.show()


# In[65]:


#Observations:

#The results from hierarchical clustering seem to be difficult to distinguish and comment 
#on especially because of one cluster which contains 103 countries


# In[66]:


#DBSCAN algorithm


# In[67]:


dbs = DBSCAN(eps = 1)

data_scaled_copy['DBSLabels'] = dbs.fit_predict(data_scaled)

data['DBSLabels'] = dbs.fit_predict(data_scaled)


# In[68]:


data['DBSLabels'].value_counts()


# In[69]:


# Calculating the mean and the median of the original data for each label
original_features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

mean = data.groupby('DBSLabels').mean()

median = data.groupby('DBSLabels').median()

df_hierachical = pd.concat([mean, median], axis = 0)

df_hierachical.index = ['group_-1 Mean', 'group_0 Mean', 'group_1 Mean', 'group_2 Mean', 'group_-1 Median', 'group_0 Median', 'group_1 Median', 'group_2 Median']

df_hierachical[original_features].T


# In[70]:


#Observations:

#DBSCAN returns 4 clusters. The countries in 3 of these clusters have similar profiles to the 
#results seen in the other clustering algorithms - high income, low income and moderately developed countries.
#The country profile of the last cluster (cluster -1) seems uncertain. This cluster has 
#a large difference between the mean values and the median values of various attributes 
#implying the presence of outliers in the cluster.


# In[71]:


#Let's visualize the box plots to comment further on these clusters


# In[72]:


for col in cols_visualise:
    sns.boxplot(x = 'DBSLabels', y = col, data = data)
    
    plt.show()


# In[73]:


#Observations

#We can see that while the three clusters (0, 1, and 2) seem to be way more compact across 
#all attributes, cluster -1 consists of extreme outliers on at least one attribute.
#Therefore, it is not adding any value to our cluster analysis. We can explore it further to 
#understand which type of countries it consists of.


# In[74]:


#Conclusion
#The choice of clustering algorithm here will depend on the context and use case. But purely
#based on foundations of 'what good clustering looks like', one can propose K-Medoids as it has 
#extreme clusters that are more distinct from each other.


# In[ ]:




