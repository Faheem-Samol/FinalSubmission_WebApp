#!/usr/bin/env python
# coding: utf-8

# ## Import dependencies

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

apps_training = apps.loc[apps['Split'] == 'Train']

apps_training.shape

apps_training.head()

apps_testing = apps.loc[apps['Split'] == 'Test']
apps_testing.shape
apps_testing.head()


user_history_training = user_history.loc[user_history['Split'] =='Train']


user_history_training = user_history.loc[user_history['Split'] =='Train']
user_history_testing = user_history.loc[user_history['Split'] =='Test']
apps_training = apps.loc[apps['Split'] == 'Train']
apps_testing = apps.loc[apps['Split'] == 'Test']
users_training = users.loc[users['Split']=='Train']
users_testing = users.loc[users['Split']=='Test']

user_history_training.shape

user_history_training.head()
user_history_testing = user_history.loc[user_history['Split'] =='Test']
user_history_testing.shape
user_history_testing.head()
users_training = users.loc[users['Split']=='Train']
users_training.shape
users_training.head()
users_testing = users.loc[users['Split']=='Test'
users_testing.shape
users_testing.head()
apps_training.head()
user_history_training.head()
users_training.head(5).transpose()
jobs.head()
jobs.groupby(['City','State','Country']).size().reset_index(name='Locationwise')
jobs.groupby(['Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',
                                                                             ascending=False).head()
Country_wise_job = jobs.groupby(['Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',
                                                                             ascending=False)