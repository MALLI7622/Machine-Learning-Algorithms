#!/usr/bin/env python
# coding: utf-8

# In[423]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

user_col = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J']
data = pd.read_table("abalone.data" , sep = ',', header = None, names = user_col)


# In[424]:


data


# In[425]:


data['A'].unique()


# In[426]:


data['A'].value_counts()


# In[427]:


data.A = pd.Categorical(data.A).codes


# In[428]:


data


# In[429]:


data['A'].value_counts()


# In[430]:


data = data.to_numpy()


# In[431]:


data


# In[410]:


class EffficientkNNclassification:
    
    def __init__(self):
        
        np.random.seed(0)
        pass
        
    def normalise_data(self, X):
        
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        
        return X_scaled
    
    def forward_pass(self, X, W, rho1, rho2):
        
        a = np.sqrt((np.matmul(X, W) - X) **2)
        b = np.sum(np.absolute(W), axis = 1)
        L = csgraph.laplacian(X[:, 1]*X[:, 1][:, np.newaxis], normed=False)
        c = np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(W), np.transpose(X)), L), X), W) 
        
        return a + (rho1 * b) + (rho2 * np.trace(c))
    
    def loss_fn(self, X, W):
        
        return np.sum(np.sqrt(np.matmul(X, W) ** 2)) / X.shape[1]
    
    def update_weights(self, W):
        
        return np.sqrt(W**2) / (2 * np.sqrt(np.absolute(W)))
    
    
    def fit(self, X, epochs = 1, rho1 = 1, rho2 = 1, display_loss = True):
        
        X = self.normalise_data(X)
        X = np.transpose(X)
        W = np.random.randn(X.shape[1], X.shape[1])
        loss = []
        
        for i in tqdm(range(epochs)):
            
            self.forward_pass(X, W, rho1, rho2)
            loss.append(self.loss_fn(X, W))
            W = np.absolute(W) - 0.1 *self.update_weights(W)
            
        if display_loss:
            plt.style.use('ggplot')
            plt.figure(figsize = (5,5))
            plt.plot(loss, '-o', markersize = 5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()


# In[411]:


model = EffficientkNNclassification()


# In[435]:


get_ipython().run_cell_magic('time', '', 'model.fit(data, epochs = 100, rho1 = 0.2, rho2 = 0.2)')


# In[ ]:





# In[ ]:





# In[ ]:




