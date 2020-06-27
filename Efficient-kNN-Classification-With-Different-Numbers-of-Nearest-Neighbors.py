#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
user_col = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J']
data = pd.read_table("abalone.data" , sep = ',', header = None, names = user_col)


# In[108]:


data


# In[109]:


data['A'].unique()


# In[110]:


data['A'].value_counts()


# In[111]:


data.A = pd.Categorical(data.A).codes


# In[112]:


data


# In[113]:


data['A'].value_counts()


# In[114]:


X = data.values


# In[115]:


X


# In[116]:


min_max_scalar = preprocessing.MinMaxScaler()
X_scaled = min_max_scalar.fit_transform(X)


# In[117]:


X[1].shape


# In[118]:


X.shape


# In[119]:


np.random.seed(0)


# In[120]:


W = np.random.randn(X.shape[0], X.shape[1])


# In[121]:


W


# In[244]:


class kNNClassification:
    
    def __init__(self):
        
        pass
    
    def normalise_data(self, X):
        
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        
        return X_scaled
    
    def forward_pass(self, X, W):
        
        return np.multiply(X, W)
    
    def error(self, X, W):
        
        return self.forward_pass(X, W) - X
    
    def loss(self, X, W):
        
        return np.sum((self.forward_pass(X, W) - X) / X.shape[0])
    
    
    def fit(self, X, epochs = 100, mini_batch_size = 32, display_loss = True):
        m = X.shape[0]
        X = self.normalise_data(X)
        W = np.random.randn(X.shape[0], X.shape[1])
        print(X.shape, W.shape)
        Loss  = []
        for i in tqdm(range(epochs)):
            for i in range(0, m, mini_batch_size):
                
                W[i:i+mini_batch_size] -= self.error(X[i:i+mini_batch_size], W[i:i+mini_batch_size])
            Loss.append(self.loss(X[i:i+mini_batch_size], W[i:i+mini_batch_size]))
                
        if display_loss:
            plt.style.use('ggplot')
            plt.figure(figsize = (8, 8))
            plt.plot(Loss)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        return W


# In[245]:


model = kNNClassification()
Weights = model.fit(X)


# In[247]:


Weights[1]


# In[249]:


Weights[:10]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




