#!/usr/bin/env python
# coding: utf-8

# In[1]:


#First we import all the libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.utils import resample
import numpy as np
import pandas as pd
import datetime
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing
from pandas.plotting import scatter_matrix


# In[1]:


from platform import python_version

print(python_version())


# In[4]:


df = pd.read_csv('E:/work/Supermicro/modeling/rsd_node_20211228_15.csv')


# In[294]:


# new data frame with split value columns
new = df["date"].str.split(".", n = 1, expand = True)
# making separate first name column from new data frame
df["Timestamp"]= new[0]
# making separate last name column from new data frame
df["micro_sec"]= new[1]
# Dropping old Name columns
df.drop(columns =["date"], inplace = True)
df.drop(columns =["micro_sec"], inplace = True)
# filter Timestamp alone
df = df[df['Timestamp'] != 'date']
# convert Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')


# In[295]:


df


# In[296]:


# CREATE VALIDATION DATA 
pd.Timestamp('now') - pd.Timedelta(10, 'minutes')
# Use this if the timestamp is the index of the DataFrame
last_ts = df.Timestamp.iloc[-1]
last_ts = df["Timestamp"].iloc[-1]
# IMPORTANT : you need to change only below line not require to change in all the line 
first_ts = last_ts - pd.Timedelta(15, 'minutes')
# Use this if the Timestamp is in a column
validation_data = df[df["Timestamp"] >= first_ts]
# Use this if the Timestamp is the index of the DataFrame
validation_data = df[df.Timestamp >= first_ts]


# In[297]:


validation_data


# In[ ]:




