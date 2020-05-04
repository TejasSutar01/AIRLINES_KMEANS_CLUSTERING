# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:35:42 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
df=pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\COMPLETED\\CLUSTERING\\AIRLINES\\EastWestAirlines.csv")
############Normalizing the data#################
def norm_func(i):
    x=(i-i.min()/i.std())
    return (x)
df_norm=norm_func(df.iloc[:,1:])

###########Elbow Curve###############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

############Taking 5 clusters#########
model=KMeans(n_clusters=5)
model.fit(df_norm)

model.labels_
clusters=pd.Series(model.labels_)
df["Clusters"]=clusters

final_df=df.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
final_df.iloc[:,1:].groupby(df.Clusters).mean()

final_df.to_csv("Flights_k.csv")
pwd
