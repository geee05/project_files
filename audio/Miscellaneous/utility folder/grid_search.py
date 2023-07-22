#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.experimental import enable_halving_search_cv


# In[8]:


from sklearn.model_selection import HalvingGridSearchCV


# In[9]:


param_grid= {'C':[0.1,1,10,100,1000],
             'gamma':[1,0.1,0.01,0.001,0.0001],
             'kernel':['rbf']
            }


# In[10]:


from sklearn.svm import SVC


# In[11]:


grid= HalvingGridSearchCV(SVC(),param_grid=param_grid,scoring='f1_macro',refit=True,verbose=3)


# In[12]:


def best_param(X,Y):
    grid.fit(X,Y)
    p=grid.best_params_
    return p

