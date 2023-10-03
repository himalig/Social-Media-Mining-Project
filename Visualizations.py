#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[1]:


import json
with open("networks.json") as f:
    network = json.load(f)


# In[2]:


with open("node_classification.json") as f:
    classification = json.load(f)


# In[4]:


with open("threads_data_new.json") as f:
    data_load = json.load(f)


# In[5]:


account_list=[]
for d in data_load.keys():
    account_list.append(data_load[d]["account_id"])


# In[7]:


dist_users=list(set(account_list))


# In[8]:


edge=[]
count=0
for d in data_load.keys():
    if data_load[d]["in_reply_to_id"] is not None: 
        try:
            account_id=data_load[str(data_load[d]["in_reply_to_id"])]["account_id"]
            edge.append((data_load[d]["account_id"],account_id))
        except:
            count+=1
print(count)
len(data_load)


# In[9]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.DiGraph()

no_nodes=0
for node in dist_users:
    G.add_node(node)
for e in edge:
    if e[0]!=e[1]:
        G.add_edge(e[0],e[1])
# G=G.reverse()
d = dict(G.degree)

pos = nx.spring_layout(G,scale=70, k=5/np.sqrt(G.order()))
nx.draw(G, pos, with_labels=False, node_size=[d[k]*20 for k in d], node_color='blue', font_color='black')
plt.title('Graph of all the users')
plt.show()
# G.number_of_nodes()
G.order()
# G.number_of_edges()


# In[11]:


G = nx.DiGraph()

no_nodes=0
for node in dist_users:
    G.add_node(node)
for e in edge:
    if e[0]!=e[1]:
        G.add_edge(e[0],e[1])
# G=G.reverse()
d = dict(G.degree)
node_colors = [classification[str(node)] for node in G.nodes()]
pos = nx.spring_layout(G,scale=70, k=6/np.sqrt(G.order()))
nx.draw(G, pos, with_labels=False, node_size=[d[k]*20 for k in d], node_color=node_colors, font_color='black')
plt.title('Graph of all the users')
plt.show()
# G.number_of_nodes()
G.order()


# In[18]:


#Plot the 3 metrics
light_green = '#90EE90'  # Light Green
light_red = '#FFA07A'    # Light Salmon
light_blue = '#ADD8E6'   # Light Blue
plt.figure(figsize=(12, 3))
degree_sequence_prothread=[]
degree_sequence_antithread=[]
degree_sequence_neutralthread=[]
for node, degree in G.degree():
    if classification[str(node)]=="green":
        degree_sequence_prothread.append(degree)
    elif classification[str(node)]=="red":
        degree_sequence_antithread.append(degree)
    else:
        degree_sequence_neutralthread.append(degree)
plt.subplot(1,3,3)
plt.hist(degree_sequence_prothread,  color=light_green, edgecolor='black')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Pro-Threads- Degree Distribution')

plt.subplot(1,3,2)
plt.hist(degree_sequence_antithread,  color=light_red, edgecolor='black')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Anti-Threads- Degree Distribution')

plt.subplot(1,3,1)
plt.hist(degree_sequence_neutralthread,  color=light_blue, edgecolor='black')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Neutral- Degree Distribution')
plt.show()

print()
plt.figure(figsize=(12, 3))
bet_centrality = nx.betweenness_centrality(G)
betweenness_values_prothread=[]
betweenness_values_antithread=[]
betweenness_values_neutralthread=[]
for key in bet_centrality.keys():
    if classification[str(key)]=="green":
        betweenness_values_prothread.append(bet_centrality[key])
    elif classification[str(key)]=="red":
        betweenness_values_antithread.append(bet_centrality[key])
    else:
        betweenness_values_neutralthread.append(bet_centrality[key])

plt.subplot(1,3,3)
plt.hist(betweenness_values_prothread, color=light_green, edgecolor='black')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Number of nodes')
plt.title('Pro-Threads- Betweenness Centrality')

plt.subplot(1,3,2)
plt.hist(betweenness_values_antithread, color=light_red, edgecolor='black')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Number of nodes')
plt.title('Anti-Threads- Betweenness Centrality')

plt.subplot(1,3,1)
plt.hist(betweenness_values_neutralthread, color=light_blue, edgecolor='black')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Number of nodes')
plt.title('Neutral- Betweenness Centrality')
plt.show()

print()
plt.figure(figsize=(12, 3))
cl_centrality = nx.closeness_centrality(G)
closeness_values_prothread=[]
closeness_values_antithread=[]
closeness_values_neutralthread=[]
for key in cl_centrality.keys():
    if classification[str(key)]=="green":
        closeness_values_prothread.append(cl_centrality[key])
    elif classification[str(key)]=="red":
        closeness_values_antithread.append(cl_centrality[key])
    else:
        closeness_values_neutralthread.append(cl_centrality[key])

plt.subplot(1,3,3)
plt.hist(closeness_values_prothread,  color=light_green, edgecolor='black')
plt.xlabel('Closeness Centrality')
plt.ylabel('Number of nodes')
plt.title(' Pro-Threads- Closeness Centrality')

plt.subplot(1,3,2)
plt.hist(closeness_values_antithread,  color=light_red, edgecolor='black')
plt.xlabel('Closeness Centrality')
plt.ylabel('Number of nodes')
plt.title('Anti-Threads- Closeness Centrality')

plt.subplot(1,3,1)
plt.hist(closeness_values_neutralthread,  color=light_blue, edgecolor='black')
plt.xlabel('Closeness Centrality')
plt.ylabel('Number of nodes')
plt.title('Neutral-Closeness Centrality')


plt.show()

