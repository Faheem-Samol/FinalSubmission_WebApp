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

apps = pd.read_csv(r'F:\MSusMaterial\Semester 4\CPSC 597-2\code\Front End\engine\Job_recommendation_engine-master\input_data\apps.tsv', delimiter='\t',encoding='utf-8')


# In[7]:


user_history = pd.read_csv(r'F:\MSusMaterial\Semester 4\CPSC 597-2\code\Front End\engine\Job_recommendation_engine-master\input_data\user_history.tsv', delimiter='\t',encoding='utf-8')


# In[8]:


jobs = pd.read_csv(r'F:\MSusMaterial\Semester 4\CPSC 597-2\code\Front End\engine\Job_recommendation_engine-master\input_data\jobs.tsv', delimiter='\t',encoding='utf-8', error_bad_lines=False)


# In[ ]:


users = pd.read_csv(r'F:\MSusMaterial\Semester 4\CPSC 597-2\code\Front End\engine\Job_recommendation_engine-master\input_data\users.tsv' ,delimiter='\t',encoding='utf-8')


# In[ ]:


test_users = pd.read_csv(r'F:\MSusMaterial\Semester 4\CPSC 597-2\code\Front End\engine\Job_recommendation_engine-master\input_data\test_users.tsv', delimiter='\t',encoding='utf-8')
