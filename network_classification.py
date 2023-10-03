#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import json
with open("/kaggle/input/classification/threads_327_data.json") as f:
    data_load = json.load(f)


# In[3]:


account_list=[]
for d in data_load.keys():
    account_list.append(data_load[d]["account_id"])
    


# In[12]:


dist_users=list(set(account_list))


# In[11]:


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


# In[16]:


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


# In[69]:


network = {}
for ed in edge:
    if ed[0] in network:
        network[ed[0]].append(ed[1])
    else:
        network[ed[0]] = [ed[1]]


# In[71]:


with open("/kaggle/working/networks.json", "w") as json_file:
    json.dump(network, json_file, indent=4)


# In[17]:


get_ipython().system('pip install sentencepiece==0.1.99')
get_ipython().system('pip install transformers==4.31.0')
get_ipython().system('pip install accelerate==0.21.0')
get_ipython().system('pip install bitsandbytes==0.41.1')


# In[18]:


from transformers import LlamaForCausalLM, LlamaTokenizer
hf_access_token = "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"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_access_token)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", load_in_8bit=True, device_map="auto", token=hf_access_token)


# In[21]:


all_posts=list(sorted(data_load.keys()))


# In[20]:


def llama2_system_talk(system, text):
    gen_len = 512

    generation_kwargs = {
          "max_new_tokens": gen_len,
          "top_p": 0.9,
          "temperature": 0.6,
          "repetition_penalty": 1.2,
          "do_sample": True,
      }

    B_INST, E_INST = "[INST]", "[/INST]"

    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    system_prompt = system

    prompt_text = text

    prompt = f"{B_INST} {B_SYS} {system_prompt} {E_SYS} {prompt_text} {E_INST}"  # Special format required by the Llama2 Chat Model where we can use system messages to provide more context about the task

    prompt_ids = tokenizer(prompt, return_tensors="pt")

    prompt_size = prompt_ids['input_ids'].size()[1]

    generate_ids = model.generate(prompt_ids.input_ids.to(model.device), **generation_kwargs)

    generate_ids = generate_ids.squeeze()

    response = tokenizer.decode(generate_ids.squeeze()[prompt_size+1:], skip_special_tokens=True).strip()

    # print("Llama response: ", response)
    return response


# In[25]:


classification={}


# In[40]:


count=0
import re
system_prompt="I am sharing with you a conversation that various users have through tweets on Mastodon. Every tweet is separated by “:::::::” . First tweet is the parent tweet, following tweets separated by “:::::::” are the replies to the tweet before it. Based on this conversation, Can you rate the opinions of the last person on the conversation which i provided, on a scale of -1 to 1. Here -1 means that the user is Anti-Thread. 1 rating means that the user is Pro-Thread and 0 rating means that the user is Neutral or unbiased about threads. These ratings should be deduced by considering the entire conversation and what the last person has to say about it. Also if the last user’s opinion is Neutral, then classify that user based on what the first user tweets. If the user feels positive towards Threads, classify it as 1. If the user feels negative towards Threads, classify it as -1. Do not provide any further explanation. I am only concerned with the actual Rating. Do not provide any other textual response."

count=0
for toot in all_posts:
  # value=toot.values()
#   if count<5:
    content =data_load[toot]["content"]
#     print(content)
    response=llama2_system_talk(system_prompt, content)
#     print("Response ->>>",response)
    result = re.findall(r"[-+]?(?:\d*\.*\d+)",response)

    classify_rate=classification.get(data_load[toot]["account_id"],[])
    classify_rate+=result
    classification[data_load[toot]["account_id"]]=classify_rate
    count+=1
    print(count)




# In[43]:


node_classification={}
import statistics
for cl in classification.keys():
    tmp=[]
    for i in classification.get(cl):
        tmp.append(int(i))
    tmp=sorted(tmp)
    if tmp:
        if tmp[0]==-1:
            node_classification[cl]="red"
        else:
            mode_nodes=statistics.mode(tmp)
            if mode_nodes==0:
                node_classification[cl]="blue"
            else:
                node_classification[cl]="green"
    else:
        node_classification[cl]="blue"
        
    
    


# In[44]:


node_classification


# In[41]:


with open("/kaggle/working/classification4.json", "w") as json_file:
    json.dump(classification, json_file, indent=4)


# In[45]:


with open("/kaggle/working/node_classification.json", "w") as json_file:
    json.dump(node_classification, json_file, indent=4)


# In[47]:


G = nx.DiGraph()

no_nodes=0
for node in dist_users:
    G.add_node(node)
for e in edge:
    if e[0]!=e[1]:
        G.add_edge(e[0],e[1])
# G=G.reverse()
d = dict(G.degree)
node_colors = [node_classification[node] for node in G.nodes()]
pos = nx.spring_layout(G,scale=70, k=6/np.sqrt(G.order()))
nx.draw(G, pos, with_labels=False, node_size=[d[k]*20 for k in d], node_color=node_colors, font_color='black')
plt.title('Graph of all the users')
plt.show()
# G.number_of_nodes()
G.order()


# In[68]:


#Plot the 3 metrics
light_green = '#90EE90'  # Light Green
light_red = '#FFA07A'    # Light Salmon
light_blue = '#ADD8E6'   # Light Blue
plt.figure(figsize=(12, 3))
degree_sequence_prothread=[]
degree_sequence_antithread=[]
degree_sequence_neutralthread=[]
for node, degree in G.degree():
    if node_classification[int(node)]=="green":
        degree_sequence_prothread.append(degree)
    elif node_classification[int(node)]=="red":
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
    if node_classification[int(key)]=="green":
        betweenness_values_prothread.append(bet_centrality[key])
    elif node_classification[int(key)]=="red":
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
    if node_classification[int(key)]=="green":
        closeness_values_prothread.append(cl_centrality[key])
    elif node_classification[int(key)]=="red":
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


# In[37]:


len(all_posts)

