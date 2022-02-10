#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reference Link: https://predictivehacks.com/feature-importance-in-python/

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 14, 7
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
#First we import all the libraries
#!pip install xgboost
import numpy as np
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from numpy import int64
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.utils import resample
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing
from pandas.plotting import scatter_matrix
#import library psycopyg2
import psycopg2
#import library pandas
import pandas as pd
#import library sqlio
import pandas.io.sql as sqlio


# In[1]:


#create database connection variable 
#conn = psycopg2.connect(user="user", password="password", host="xxx.xxx.xxx.xxx", database="db_name")
conn = psycopg2.connect(user="data_user", password="kgtopg8932", host="localhost", database="rawData")


# In[3]:


#define query
query = "select * from analytics1.backblaze_events"


# In[4]:


#execute query and save it to a variable
dataset = sqlio.read_sql_query(query,conn)
dataset


# In[5]:


# Load data
#data = pd.read_csv('E:/work/Supermicro/modeling/data_Q2_2019/2019-06-06.csv')
#data.fillna(0)


# In[6]:


import pandas as pd
import numpy as np
 
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
 
#we used only the train dataset from Titanic


# In[ ]:


# missing value treatment wth Dropping columns or rows with missing value rate higher than threshold
threshold = 0.7

#Dropping rows with missing value rate higher than threshold
dataset = dataset.loc[dataset.isnull().mean(axis=1) < threshold]

#Dropping columns with missing value rate higher than threshold
dataset = dataset[dataset.columns[dataset.isnull().mean() < threshold]]


# In[ ]:


dataset


# In[ ]:


dataset['smart_200_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_200_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_189_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_184_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_184_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_189_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_191_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_191_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_195_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_195_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_187_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_188_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_187_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_190_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_190_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_188_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_242_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_242_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_241_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_241_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_240_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_240_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_193_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_193_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_3_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_3_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_4_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_4_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_5_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_5_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_7_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_7_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_10_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_10_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_199_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_199_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_198_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_198_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_197_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_197_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_12_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_1_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['serial_number'] = dataset['capacity_bytes'].fillna(0)
dataset['model'] = dataset['capacity_bytes'].fillna(0)
dataset['capacity_bytes'] = dataset['capacity_bytes'].fillna(0)
dataset['failure'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_1_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_194_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_192_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_192_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_194_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_12_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_9_normalized'] = dataset['capacity_bytes'].fillna(0)
dataset['smart_9_raw'] = dataset['capacity_bytes'].fillna(0)
dataset['event_date'] = dataset['capacity_bytes'].fillna(0)


# In[ ]:


data=dataset[['smart_200_normalized', 
'smart_200_raw', 
'smart_189_normalized', 
'smart_184_normalized', 
'smart_184_raw', 
'smart_189_raw', 
'smart_191_raw', 
'smart_191_normalized', 
'smart_195_raw', 
'smart_195_normalized', 
'smart_187_raw', 
'smart_188_raw', 
'smart_187_normalized', 
'smart_190_normalized', 
'smart_190_raw', 
'smart_188_normalized', 
'smart_242_normalized', 
'smart_242_raw', 
'smart_241_raw', 
'smart_241_normalized', 
'smart_240_normalized', 
'smart_240_raw', 
'smart_193_raw', 
'smart_193_normalized', 
'smart_3_normalized', 
'smart_3_raw', 
'smart_4_normalized', 
'smart_4_raw', 
'smart_5_normalized', 
'smart_5_raw', 
'smart_7_normalized', 
'smart_7_raw', 
'smart_10_normalized', 
'smart_10_raw', 
'smart_199_raw', 
'smart_199_normalized', 
'smart_198_raw', 
'smart_198_normalized', 
'smart_197_raw', 
'smart_197_normalized', 
'smart_12_raw', 
'smart_1_raw', 
'serial_number', 
'model', 
'capacity_bytes', 
'failure', 
'smart_1_normalized', 
'smart_194_raw', 
'smart_192_raw', 
'smart_192_normalized', 
'smart_194_normalized', 
'smart_12_normalized', 
'smart_9_normalized', 
'smart_9_raw', 
'event_date'
]]


# In[ ]:


data


# In[ ]:


#data.dropna(inplace=True)


# In[ ]:


model=LogisticRegression(random_state=1)


# In[ ]:


data


# In[ ]:


features=pd.get_dummies(data[['smart_200_normalized', 
'smart_200_raw', 
'smart_189_normalized', 
'smart_184_normalized', 
'smart_184_raw', 
'smart_189_raw', 
'smart_191_raw', 
'smart_191_normalized', 
'smart_195_raw', 
'smart_195_normalized', 
'smart_187_raw', 
'smart_188_raw', 
'smart_187_normalized', 
'smart_190_normalized', 
'smart_190_raw', 
'smart_188_normalized', 
'smart_242_normalized', 
'smart_242_raw', 
'smart_241_raw', 
'smart_241_normalized', 
'smart_240_normalized', 
'smart_240_raw', 
'smart_193_raw', 
'smart_193_normalized', 
'smart_3_normalized', 
'smart_3_raw', 
'smart_4_normalized', 
'smart_4_raw', 
'smart_5_normalized', 
'smart_5_raw', 
'smart_7_normalized', 
'smart_7_raw', 
'smart_10_normalized', 
'smart_10_raw', 
'smart_199_raw', 
'smart_199_normalized', 
'smart_198_raw', 
'smart_198_normalized', 
'smart_197_raw', 
'smart_197_normalized', 
'smart_12_raw', 
'smart_1_raw', 
'serial_number', 
'model', 
'capacity_bytes', 
'failure', 
'smart_1_normalized', 
'smart_194_raw', 
'smart_192_raw', 
'smart_192_normalized', 
'smart_194_normalized', 
'smart_12_normalized', 
'smart_9_normalized', 
'smart_9_raw', 
'event_date'
]],drop_first=True)


# In[ ]:


#features['smart_255_normalized']=data['smart_255_normalized']


# In[ ]:


percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
percent


# In[ ]:


model.fit(features,data['failure'])


# In[ ]:


feature_importance=pd.DataFrame({'feature':list(features.columns),'feature_importance':[abs(i) for i in model.coef_[0]]})
feature_importance.sort_values('feature_importance',ascending=False)
 


# In[ ]:


feature_importance.to_csv(r'E:\work\Supermicro\modeling\data_Q2_2019\feature_importance.csv')


# In[ ]:


## 2nd model for feature imporatance 
model=RandomForestClassifier()
 
model.fit(features,data['failure'])
 
feature_importances=pd.DataFrame({'features':features.columns,'feature_importance':model.feature_importances_})
feature_importances.sort_values('feature_importance',ascending=False)


# In[ ]:


feature_importances.to_csv(r'E:\work\Supermicro\modeling\data_Q2_2019\feature_importance2.csv')


# In[ ]:


# 3rd methord
model=XGBClassifier()
 
model.fit(features,data['failure'])
 
feature_importances=pd.DataFrame({'features':features.columns,'feature_importance':model.feature_importances_})
print(feature_importances.sort_values('feature_importance',ascending=False))


# In[ ]:


# 4th methord 
model=smf.logit('Survived~Sex+Age+Embarked+Pclass+SibSp+Parch',data=data)
result = model.fit()
 
feature_importances=pd.DataFrame(result.conf_int()[1]).rename(columns={1:'Coefficients'}).eval("absolute_coefficients=abs(Coefficients)")
feature_importances.sort_values('absolute_coefficients',ascending=False).drop('Intercept')[['absolute_coefficients']]

