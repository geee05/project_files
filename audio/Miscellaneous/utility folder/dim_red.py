#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as  pd 
import numpy as np
from sklearn.utils import resample


# In[4]:


#statistical analysis on features:
df=pd.read_csv('/home/vboxuser/VoiceFiles/ImageFolderopensmile_daic_dev_stat.csv')
#train and test data:
df_train=pd.read_csv('/home/vboxuser/Downloads/VoiceFiles/opensmile_daic_train.csv')


# In[5]:


req_features=df[(df['p_HC_DP']<0.05) & (abs(df['p_HC_DP.1'])>0.4)]
feature_list=req_features['Unnamed: 0'].tolist()


# In[6]:


df_train.replace({'class':{"HC":0,"DP":1}},inplace=True)
X_train=df_train[df_train.columns.intersection(feature_list)]
Y_train=df_train['class']


# In[7]:


df_1=df_train[df_train['class']==1]
df_other=df_train[df_train['class']!=1]
df_upsampled=resample(df_1,random_state=42,n_samples=126,replace=True)
df_train_upsampled=pd.concat([df_upsampled,df_other])
df_train_upsampled.reset_index(inplace=True)


# In[8]:


X_train_resampled=df_train_upsampled[df_train_upsampled.columns.intersection(feature_list)]
Y_train_resampled=df_train_upsampled['class']


# In[9]:


import Normalization as n


# In[10]:


X=n.normalize_train(X_train_resampled)
Y=Y_train_resampled


# In[12]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[14]:


class dim_red_models:
    def PCA(self,train,test):
        self.train=train
        self.test=test
        pca=PCA(n_components=4)
        x_trainpca=pca.fit_transform(train)
        x_testpca=pca.transform(test) 
        return x_trainpca,x_testpca
        
    def tsne(self,train):
        self.train=train
        time_start=time.time()
        tsne=TSNE(n_components=2,verbose=1,perplexity=40,n_iter=300)
        xtrain_tsne=tsne.fit_transform(train)
        return xtrain_tsne
        
    def lda(self,train,test,Y):   
        self.train=train
        self.test=test
        self.Y=Y
        lda=LDA(n_components=1)
        xtrain_lda=lda.fit_transform(train,Y)
        x_testlda=lda.transform(test)
        return xtrain_lda,x_testlda

