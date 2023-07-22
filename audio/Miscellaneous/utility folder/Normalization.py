#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
std_Scaler=StandardScaler()
std_Scaler


# In[2]:


def normalize_train(train):
    norm_train=std_Scaler.fit_transform(train)
    #norm_test=pd.DataFrame(std_Scaler.transform(train))
    return norm_train


# In[3]:


def normalize_test(test):
    norm_test=std_Scaler.transform(test)
    return norm_test

