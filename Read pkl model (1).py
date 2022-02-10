#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries 
import numpy as np
import numpy as np
import pandas as pd 
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.datasets import load_breast_cancer
from numpy import int64
from sklearn import svm
from sklearn.svm import SVC 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 
from pyhive import hive
#from impala.dbapi import connect
from hdfs import InsecureClient
from pyhive import hive
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


# In[11]:


#hive
conn = hive.Connection(host='172.27.27.60', port=10000, password='Hive#Pa55', username='smicro', auth='CUSTOM')
# final
#dataset = pd.read_sql("SELECT * FROM events1.disk_smartdata_dtls", conn)
dataset = pd.read_sql("SELECT * FROM sda_hdd_db.ml_smart_data_view limit 100", conn)
#dataset = pd.read_sql("SELECT * FROM sda_hdd_db.sda_hdd_all", conn)
# sda_hdd_db.sda_hdd_backblaze
# sda_hdd_db.sda_hdd_all
dataset


# In[56]:


#select * from sda_hdd_db.sda_hdd_backblaze order by date asc limit 10;# -- first 10 rows
#select * from sda_hdd_db.sda_hdd_backblaze order by date desc limit 10;# -- last 10 rows
#dataset1 = pd.read_sql("SELECT * from sda_hdd_db.sda_hdd_backblaze order by `date` asc limit 10", conn)
#dataset2 = pd.read_sql("SELECT * from sda_hdd_db.sda_hdd_backblaze order by `date` desc limit 10", conn)


# In[64]:


dataset10 = pd.read_sql("SELECT process_date, serial_number, model from events1.disk_smartdata_dtls where extract (hour from process_date) || ":" || extract (minute from process_date) in ('1:0', '2:0', '3:0') limit 10", conn)


# In[3]:


dataset7=dataset
dataset8=dataset


# In[6]:


conn = hive.Connection(host='172.27.27.60', port=10000, password='Hive#Pa55', username='smicro', auth='CUSTOM')
dataset = pd.read_sql("select t.* from (select t.*, dense_rank() over (order by t.id asc) as seqnum from sda_hdd_db.sda_hdd_backblaze t) t where seqnum <= 10", conn)


# In[10]:


# validation2
dataset=dataset7.head(500)


# In[11]:


dataset


# In[12]:


dataset.rename(columns={'sda_hdd_all.date':'date',
'sda_hdd_all.serial_number':'serial_number',
'sda_hdd_all.model':'model',
'sda_hdd_all.capacity_bytes':'capacity_bytes',
'sda_hdd_all.failure':'failure',
'sda_hdd_all.smart_1_normalized':'smart_1_normalized',
'sda_hdd_all.smart_1_raw':'smart_1_raw',
'sda_hdd_all.smart_2_normalized':'smart_2_normalized',
'sda_hdd_all.smart_2_raw':'smart_2_raw',
'sda_hdd_all.smart_3_normalized':'smart_3_normalized',
'sda_hdd_all.smart_3_raw':'smart_3_raw',
'sda_hdd_all.smart_4_normalized':'smart_4_normalized',
'sda_hdd_all.smart_4_raw':'smart_4_raw',
'sda_hdd_all.smart_5_normalized':'smart_5_normalized',
'sda_hdd_all.smart_5_raw':'smart_5_raw',
'sda_hdd_all.smart_7_normalized':'smart_7_normalized',
'sda_hdd_all.smart_7_raw':'smart_7_raw',
'sda_hdd_all.smart_8_normalized':'smart_8_normalized',
'sda_hdd_all.smart_8_raw':'smart_8_raw',
'sda_hdd_all.smart_9_normalized':'smart_9_normalized',
'sda_hdd_all.smart_9_raw':'smart_9_raw',
'sda_hdd_all.smart_10_normalized':'smart_10_normalized',
'sda_hdd_all.smart_10_raw':'smart_10_raw',
'sda_hdd_all.smart_11_normalized':'smart_11_normalized',
'sda_hdd_all.smart_11_raw':'smart_11_raw',
'sda_hdd_all.smart_12_normalized':'smart_12_normalized',
'sda_hdd_all.smart_12_raw':'smart_12_raw',
'sda_hdd_all.smart_13_normalized':'smart_13_normalized',
'sda_hdd_all.smart_13_raw':'smart_13_raw',
'sda_hdd_all.smart_15_normalized':'smart_15_normalized',
'sda_hdd_all.smart_15_raw':'smart_15_raw',
'sda_hdd_all.smart_16_normalized':'smart_16_normalized',
'sda_hdd_all.smart_16_raw':'smart_16_raw',
'sda_hdd_all.smart_17_normalized':'smart_17_normalized',
'sda_hdd_all.smart_17_raw':'smart_17_raw',
'sda_hdd_all.smart_18_normalized':'smart_18_normalized',
'sda_hdd_all.smart_18_raw':'smart_18_raw',
'sda_hdd_all.smart_22_normalized':'smart_22_normalized',
'sda_hdd_all.smart_22_raw':'smart_22_raw',
'sda_hdd_all.smart_23_normalized':'smart_23_normalized',
'sda_hdd_all.smart_23_raw':'smart_23_raw',
'sda_hdd_all.smart_24_normalized':'smart_24_normalized',
'sda_hdd_all.smart_24_raw':'smart_24_raw',
'sda_hdd_all.smart_160_normalized':'smart_160_normalized',
'sda_hdd_all.smart_160_raw':'smart_160_raw',
'sda_hdd_all.smart_161_normalized':'smart_161_normalized',
'sda_hdd_all.smart_161_raw':'smart_161_raw',
'sda_hdd_all.smart_163_normalized':'smart_163_normalized',
'sda_hdd_all.smart_163_raw':'smart_163_raw',
'sda_hdd_all.smart_164_normalized':'smart_164_normalized',
'sda_hdd_all.smart_164_raw':'smart_164_raw',
'sda_hdd_all.smart_165_normalized':'smart_165_normalized',
'sda_hdd_all.smart_165_raw':'smart_165_raw',
'sda_hdd_all.smart_166_normalized':'smart_166_normalized',
'sda_hdd_all.smart_166_raw':'smart_166_raw',
'sda_hdd_all.smart_167_normalized':'smart_167_normalized',
'sda_hdd_all.smart_167_raw':'smart_167_raw',
'sda_hdd_all.smart_168_normalized':'smart_168_normalized',
'sda_hdd_all.smart_168_raw':'smart_168_raw',
'sda_hdd_all.smart_169_normalized':'smart_169_normalized',
'sda_hdd_all.smart_169_raw':'smart_169_raw',
'sda_hdd_all.smart_170_normalized':'smart_170_normalized',
'sda_hdd_all.smart_170_raw':'smart_170_raw',
'sda_hdd_all.smart_173_normalized':'smart_173_normalized',
'sda_hdd_all.smart_173_raw':'smart_173_raw',
'sda_hdd_all.smart_174_normalized':'smart_174_normalized',
'sda_hdd_all.smart_174_raw':'smart_174_raw',
'sda_hdd_all.smart_175_normalized':'smart_175_normalized',
'sda_hdd_all.smart_175_raw':'smart_175_raw',
'sda_hdd_all.smart_176_normalized':'smart_176_normalized',
'sda_hdd_all.smart_176_raw':'smart_176_raw',
'sda_hdd_all.smart_177_normalized':'smart_177_normalized',
'sda_hdd_all.smart_177_raw':'smart_177_raw',
'sda_hdd_all.smart_178_normalized':'smart_178_normalized',
'sda_hdd_all.smart_178_raw':'smart_178_raw',
'sda_hdd_all.smart_179_normalized':'smart_179_normalized',
'sda_hdd_all.smart_179_raw':'smart_179_raw',
'sda_hdd_all.smart_180_normalized':'smart_180_normalized',
'sda_hdd_all.smart_180_raw':'smart_180_raw',
'sda_hdd_all.smart_181_normalized':'smart_181_normalized',
'sda_hdd_all.smart_181_raw':'smart_181_raw',
'sda_hdd_all.smart_182_normalized':'smart_182_normalized',
'sda_hdd_all.smart_182_raw':'smart_182_raw',
'sda_hdd_all.smart_183_normalized':'smart_183_normalized',
'sda_hdd_all.smart_183_raw':'smart_183_raw',
'sda_hdd_all.smart_184_normalized':'smart_184_normalized',
'sda_hdd_all.smart_184_raw':'smart_184_raw',
'sda_hdd_all.smart_187_normalized':'smart_187_normalized',
'sda_hdd_all.smart_187_raw':'smart_187_raw',
'sda_hdd_all.smart_188_normalized':'smart_188_normalized',
'sda_hdd_all.smart_188_raw':'smart_188_raw',
'sda_hdd_all.smart_189_normalized':'smart_189_normalized',
'sda_hdd_all.smart_189_raw':'smart_189_raw',
'sda_hdd_all.smart_190_normalized':'smart_190_normalized',
'sda_hdd_all.smart_190_raw':'smart_190_raw',
'sda_hdd_all.smart_191_normalized':'smart_191_normalized',
'sda_hdd_all.smart_191_raw':'smart_191_raw',
'sda_hdd_all.smart_192_normalized':'smart_192_normalized',
'sda_hdd_all.smart_192_raw':'smart_192_raw',
'sda_hdd_all.smart_193_normalized':'smart_193_normalized',
'sda_hdd_all.smart_193_raw':'smart_193_raw',
'sda_hdd_all.smart_194_normalized':'smart_194_normalized',
'sda_hdd_all.smart_194_raw':'smart_194_raw',
'sda_hdd_all.smart_195_normalized':'smart_195_normalized',
'sda_hdd_all.smart_195_raw':'smart_195_raw',
'sda_hdd_all.smart_196_normalized':'smart_196_normalized',
'sda_hdd_all.smart_196_raw':'smart_196_raw',
'sda_hdd_all.smart_197_normalized':'smart_197_normalized',
'sda_hdd_all.smart_197_raw':'smart_197_raw',
'sda_hdd_all.smart_198_normalized':'smart_198_normalized',
'sda_hdd_all.smart_198_raw':'smart_198_raw',
'sda_hdd_all.smart_199_normalized':'smart_199_normalized',
'sda_hdd_all.smart_199_raw':'smart_199_raw',
'sda_hdd_all.smart_200_normalized':'smart_200_normalized',
'sda_hdd_all.smart_200_raw':'smart_200_raw',
'sda_hdd_all.smart_201_normalized':'smart_201_normalized',
'sda_hdd_all.smart_201_raw':'smart_201_raw',
'sda_hdd_all.smart_202_normalized':'smart_202_normalized',
'sda_hdd_all.smart_202_raw':'smart_202_raw',
'sda_hdd_all.smart_206_normalized':'smart_206_normalized',
'sda_hdd_all.smart_206_raw':'smart_206_raw',
'sda_hdd_all.smart_210_normalized':'smart_210_normalized',
'sda_hdd_all.smart_210_raw':'smart_210_raw',
'sda_hdd_all.smart_218_normalized':'smart_218_normalized',
'sda_hdd_all.smart_218_raw':'smart_218_raw',
'sda_hdd_all.smart_220_normalized':'smart_220_normalized',
'sda_hdd_all.smart_220_raw':'smart_220_raw',
'sda_hdd_all.smart_222_normalized':'smart_222_normalized',
'sda_hdd_all.smart_222_raw':'smart_222_raw',
'sda_hdd_all.smart_223_normalized':'smart_223_normalized',
'sda_hdd_all.smart_223_raw':'smart_223_raw',
'sda_hdd_all.smart_224_normalized':'smart_224_normalized',
'sda_hdd_all.smart_224_raw':'smart_224_raw',
'sda_hdd_all.smart_225_normalized':'smart_225_normalized',
'sda_hdd_all.smart_225_raw':'smart_225_raw',
'sda_hdd_all.smart_226_normalized':'smart_226_normalized',
'sda_hdd_all.smart_226_raw':'smart_226_raw',
'sda_hdd_all.smart_231_normalized':'smart_231_normalized',
'sda_hdd_all.smart_231_raw':'smart_231_raw',
'sda_hdd_all.smart_232_normalized':'smart_232_normalized',
'sda_hdd_all.smart_232_raw':'smart_232_raw',
'sda_hdd_all.smart_233_normalized':'smart_233_normalized',
'sda_hdd_all.smart_233_raw':'smart_233_raw',
'sda_hdd_all.smart_234_normalized':'smart_234_normalized',
'sda_hdd_all.smart_234_raw':'smart_234_raw',
'sda_hdd_all.smart_235_normalized':'smart_235_normalized',
'sda_hdd_all.smart_235_raw':'smart_235_raw',
'sda_hdd_all.smart_240_normalized':'smart_240_normalized',
'sda_hdd_all.smart_240_raw':'smart_240_raw',
'sda_hdd_all.smart_241_normalized':'smart_241_normalized',
'sda_hdd_all.smart_241_raw':'smart_241_raw',
'sda_hdd_all.smart_242_normalized':'smart_242_normalized',
'sda_hdd_all.smart_242_raw':'smart_242_raw',
'sda_hdd_all.smart_245_normalized':'smart_245_normalized',
'sda_hdd_all.smart_245_raw':'smart_245_raw',
'sda_hdd_all.smart_247_normalized':'smart_247_normalized',
'sda_hdd_all.smart_247_raw':'smart_247_raw',
'sda_hdd_all.smart_248_normalized':'smart_248_normalized',
'sda_hdd_all.smart_248_raw':'smart_248_raw',
'sda_hdd_all.smart_250_normalized':'smart_250_normalized',
'sda_hdd_all.smart_250_raw':'smart_250_raw',
'sda_hdd_all.smart_251_normalized':'smart_251_normalized',
'sda_hdd_all.smart_251_raw':'smart_251_raw',
'sda_hdd_all.smart_252_normalized':'smart_252_normalized',
'sda_hdd_all.smart_252_raw':'smart_252_raw',
'sda_hdd_all.smart_254_normalized':'smart_254_normalized',
'sda_hdd_all.smart_254_raw':'smart_254_raw',
'sda_hdd_all.smart_255_normalized':'smart_255_normalized',
'sda_hdd_all.smart_255_raw':'smart_255_raw'}, inplace=True)


# In[13]:


# fill missing with zero 
dataset.fillna(0)
dataset.replace(np.nan, 0)
dataset.fillna(value = 0,
          inplace = True)
dataset.replace(np.NaN, 0)


# In[14]:


#Missing value Imputation 
dataset['smart_255_raw']=dataset['smart_255_raw'].fillna(dataset['smart_255_raw'].mean())
dataset['smart_15_normalized']=dataset['smart_15_normalized'].fillna(dataset['smart_15_normalized'].mean())
dataset['smart_255_normalized']=dataset['smart_255_normalized'].fillna(dataset['smart_255_normalized'].mean())
dataset['smart_252_raw']=dataset['smart_252_raw'].fillna(dataset['smart_252_raw'].mean())
dataset['smart_252_normalized']=dataset['smart_252_normalized'].fillna(dataset['smart_252_normalized'].mean())
dataset['smart_251_raw']=dataset['smart_251_raw'].fillna(dataset['smart_251_raw'].mean())
dataset['smart_251_normalized']=dataset['smart_251_normalized'].fillna(dataset['smart_251_normalized'].mean())
dataset['smart_250_raw']=dataset['smart_250_raw'].fillna(dataset['smart_250_raw'].mean())
dataset['smart_250_normalized']=dataset['smart_250_normalized'].fillna(dataset['smart_250_normalized'].mean())
dataset['smart_15_raw']=dataset['smart_15_raw'].fillna(dataset['smart_15_raw'].mean())
dataset['smart_234_normalized']=dataset['smart_234_normalized'].fillna(dataset['smart_234_normalized'].mean())
dataset['smart_234_raw']=dataset['smart_234_raw'].fillna(dataset['smart_234_raw'].mean())
dataset['smart_206_normalized']=dataset['smart_206_normalized'].fillna(dataset['smart_206_normalized'].mean())
dataset['smart_206_raw']=dataset['smart_206_raw'].fillna(dataset['smart_206_raw'].mean())
dataset['smart_210_raw']=dataset['smart_210_raw'].fillna(dataset['smart_210_raw'].mean())
dataset['smart_210_normalized']=dataset['smart_210_normalized'].fillna(dataset['smart_210_normalized'].mean())
dataset['smart_248_raw']=dataset['smart_248_raw'].fillna(dataset['smart_248_raw'].mean())
dataset['smart_248_normalized']=dataset['smart_248_normalized'].fillna(dataset['smart_248_normalized'].mean())
dataset['smart_247_normalized']=dataset['smart_247_normalized'].fillna(dataset['smart_247_normalized'].mean())
dataset['smart_247_raw']=dataset['smart_247_raw'].fillna(dataset['smart_247_raw'].mean())
dataset['smart_166_normalized']=dataset['smart_166_normalized'].fillna(dataset['smart_166_normalized'].mean())
dataset['smart_178_raw']=dataset['smart_178_raw'].fillna(dataset['smart_178_raw'].mean())
dataset['smart_160_normalized']=dataset['smart_160_normalized'].fillna(dataset['smart_160_normalized'].mean())
dataset['smart_161_normalized']=dataset['smart_161_normalized'].fillna(dataset['smart_161_normalized'].mean())
dataset['smart_161_raw']=dataset['smart_161_raw'].fillna(dataset['smart_161_raw'].mean())
dataset['smart_163_normalized']=dataset['smart_163_normalized'].fillna(dataset['smart_163_normalized'].mean())
dataset['smart_163_raw']=dataset['smart_163_raw'].fillna(dataset['smart_163_raw'].mean())
dataset['smart_164_normalized']=dataset['smart_164_normalized'].fillna(dataset['smart_164_normalized'].mean())
dataset['smart_164_raw']=dataset['smart_164_raw'].fillna(dataset['smart_164_raw'].mean())
dataset['smart_165_normalized']=dataset['smart_165_normalized'].fillna(dataset['smart_165_normalized'].mean())
dataset['smart_165_raw']=dataset['smart_165_raw'].fillna(dataset['smart_165_raw'].mean())
dataset['smart_160_raw']=dataset['smart_160_raw'].fillna(dataset['smart_160_raw'].mean())
dataset['smart_176_raw']=dataset['smart_176_raw'].fillna(dataset['smart_176_raw'].mean())
dataset['smart_176_normalized']=dataset['smart_176_normalized'].fillna(dataset['smart_176_normalized'].mean())
dataset['smart_178_normalized']=dataset['smart_178_normalized'].fillna(dataset['smart_178_normalized'].mean())
dataset['smart_167_normalized']=dataset['smart_167_normalized'].fillna(dataset['smart_167_normalized'].mean())
dataset['smart_166_raw']=dataset['smart_166_raw'].fillna(dataset['smart_166_raw'].mean())
dataset['smart_169_raw']=dataset['smart_169_raw'].fillna(dataset['smart_169_raw'].mean())
dataset['smart_169_normalized']=dataset['smart_169_normalized'].fillna(dataset['smart_169_normalized'].mean())
dataset['smart_167_raw']=dataset['smart_167_raw'].fillna(dataset['smart_167_raw'].mean())
dataset['smart_175_normalized']=dataset['smart_175_normalized'].fillna(dataset['smart_175_normalized'].mean())
dataset['smart_175_raw']=dataset['smart_175_raw'].fillna(dataset['smart_175_raw'].mean())
dataset['smart_180_raw']=dataset['smart_180_raw'].fillna(dataset['smart_180_raw'].mean())
dataset['smart_201_normalized']=dataset['smart_201_normalized'].fillna(dataset['smart_201_normalized'].mean())
dataset['smart_201_raw']=dataset['smart_201_raw'].fillna(dataset['smart_201_raw'].mean())
dataset['smart_202_normalized']=dataset['smart_202_normalized'].fillna(dataset['smart_202_normalized'].mean())
dataset['smart_180_normalized']=dataset['smart_180_normalized'].fillna(dataset['smart_180_normalized'].mean())
dataset['smart_179_raw']=dataset['smart_179_raw'].fillna(dataset['smart_179_raw'].mean())
dataset['smart_202_raw']=dataset['smart_202_raw'].fillna(dataset['smart_202_raw'].mean())
dataset['smart_179_normalized']=dataset['smart_179_normalized'].fillna(dataset['smart_179_normalized'].mean())
dataset['smart_13_normalized']=dataset['smart_13_normalized'].fillna(dataset['smart_13_normalized'].mean())
dataset['smart_13_raw']=dataset['smart_13_raw'].fillna(dataset['smart_13_raw'].mean())
dataset['smart_170_normalized']=dataset['smart_170_normalized'].fillna(dataset['smart_170_normalized'].mean())
dataset['smart_17_normalized']=dataset['smart_17_normalized'].fillna(dataset['smart_17_normalized'].mean())
dataset['smart_170_raw']=dataset['smart_170_raw'].fillna(dataset['smart_170_raw'].mean())
dataset['smart_218_normalized']=dataset['smart_218_normalized'].fillna(dataset['smart_218_normalized'].mean())
dataset['smart_231_raw']=dataset['smart_231_raw'].fillna(dataset['smart_231_raw'].mean())
dataset['smart_231_normalized']=dataset['smart_231_normalized'].fillna(dataset['smart_231_normalized'].mean())
dataset['smart_16_raw']=dataset['smart_16_raw'].fillna(dataset['smart_16_raw'].mean())
dataset['smart_245_raw']=dataset['smart_245_raw'].fillna(dataset['smart_245_raw'].mean())
dataset['smart_16_normalized']=dataset['smart_16_normalized'].fillna(dataset['smart_16_normalized'].mean())
dataset['smart_17_raw']=dataset['smart_17_raw'].fillna(dataset['smart_17_raw'].mean())
dataset['smart_245_normalized']=dataset['smart_245_normalized'].fillna(dataset['smart_245_normalized'].mean())
dataset['smart_182_normalized']=dataset['smart_182_normalized'].fillna(dataset['smart_182_normalized'].mean())
dataset['smart_181_raw']=dataset['smart_181_raw'].fillna(dataset['smart_181_raw'].mean())
dataset['smart_181_normalized']=dataset['smart_181_normalized'].fillna(dataset['smart_181_normalized'].mean())
dataset['smart_218_raw']=dataset['smart_218_raw'].fillna(dataset['smart_218_raw'].mean())
dataset['smart_182_raw']=dataset['smart_182_raw'].fillna(dataset['smart_182_raw'].mean())
dataset['smart_174_raw']=dataset['smart_174_raw'].fillna(dataset['smart_174_raw'].mean())
dataset['smart_174_normalized']=dataset['smart_174_normalized'].fillna(dataset['smart_174_normalized'].mean())
dataset['smart_173_raw']=dataset['smart_173_raw'].fillna(dataset['smart_173_raw'].mean())
dataset['smart_173_normalized']=dataset['smart_173_normalized'].fillna(dataset['smart_173_normalized'].mean())
dataset['smart_235_normalized']=dataset['smart_235_normalized'].fillna(dataset['smart_235_normalized'].mean())
dataset['smart_235_raw']=dataset['smart_235_raw'].fillna(dataset['smart_235_raw'].mean())
dataset['smart_232_normalized']=dataset['smart_232_normalized'].fillna(dataset['smart_232_normalized'].mean())
dataset['smart_168_raw']=dataset['smart_168_raw'].fillna(dataset['smart_168_raw'].mean())
dataset['smart_168_normalized']=dataset['smart_168_normalized'].fillna(dataset['smart_168_normalized'].mean())
dataset['smart_177_raw']=dataset['smart_177_raw'].fillna(dataset['smart_177_raw'].mean())
dataset['smart_177_normalized']=dataset['smart_177_normalized'].fillna(dataset['smart_177_normalized'].mean())
dataset['smart_232_raw']=dataset['smart_232_raw'].fillna(dataset['smart_232_raw'].mean())
dataset['smart_254_raw']=dataset['smart_254_raw'].fillna(dataset['smart_254_raw'].mean())
dataset['smart_254_normalized']=dataset['smart_254_normalized'].fillna(dataset['smart_254_normalized'].mean())
dataset['smart_233_raw']=dataset['smart_233_raw'].fillna(dataset['smart_233_raw'].mean())
dataset['smart_233_normalized']=dataset['smart_233_normalized'].fillna(dataset['smart_233_normalized'].mean())
dataset['smart_225_raw']=dataset['smart_225_raw'].fillna(dataset['smart_225_raw'].mean())
dataset['smart_225_normalized']=dataset['smart_225_normalized'].fillna(dataset['smart_225_normalized'].mean())
dataset['smart_11_raw']=dataset['smart_11_raw'].fillna(dataset['smart_11_raw'].mean())
dataset['smart_11_normalized']=dataset['smart_11_normalized'].fillna(dataset['smart_11_normalized'].mean())
dataset['smart_22_normalized']=dataset['smart_22_normalized'].fillna(dataset['smart_22_normalized'].mean())
dataset['smart_22_raw']=dataset['smart_22_raw'].fillna(dataset['smart_22_raw'].mean())
dataset['smart_23_normalized']=dataset['smart_23_normalized'].fillna(dataset['smart_23_normalized'].mean())
dataset['smart_23_raw']=dataset['smart_23_raw'].fillna(dataset['smart_23_raw'].mean())
dataset['smart_24_normalized']=dataset['smart_24_normalized'].fillna(dataset['smart_24_normalized'].mean())
dataset['smart_24_raw']=dataset['smart_24_raw'].fillna(dataset['smart_24_raw'].mean())
dataset['smart_183_raw']=dataset['smart_183_raw'].fillna(dataset['smart_183_raw'].mean())
dataset['smart_183_normalized']=dataset['smart_183_normalized'].fillna(dataset['smart_183_normalized'].mean())
dataset['smart_220_normalized']=dataset['smart_220_normalized'].fillna(dataset['smart_220_normalized'].mean())
dataset['smart_226_raw']=dataset['smart_226_raw'].fillna(dataset['smart_226_raw'].mean())
dataset['smart_226_normalized']=dataset['smart_226_normalized'].fillna(dataset['smart_226_normalized'].mean())
dataset['smart_224_raw']=dataset['smart_224_raw'].fillna(dataset['smart_224_raw'].mean())
dataset['smart_224_normalized']=dataset['smart_224_normalized'].fillna(dataset['smart_224_normalized'].mean())
dataset['smart_222_raw']=dataset['smart_222_raw'].fillna(dataset['smart_222_raw'].mean())
dataset['smart_222_normalized']=dataset['smart_222_normalized'].fillna(dataset['smart_222_normalized'].mean())
dataset['smart_220_raw']=dataset['smart_220_raw'].fillna(dataset['smart_220_raw'].mean())
dataset['smart_18_normalized']=dataset['smart_18_normalized'].fillna(dataset['smart_18_normalized'].mean())
dataset['smart_18_raw']=dataset['smart_18_raw'].fillna(dataset['smart_18_raw'].mean())
dataset['smart_223_raw']=dataset['smart_223_raw'].fillna(dataset['smart_223_raw'].mean())
dataset['smart_223_normalized']=dataset['smart_223_normalized'].fillna(dataset['smart_223_normalized'].mean())
dataset['smart_200_raw']=dataset['smart_200_raw'].fillna(dataset['smart_200_raw'].mean())
dataset['smart_200_normalized']=dataset['smart_200_normalized'].fillna(dataset['smart_200_normalized'].mean())
dataset['smart_242_raw']=dataset['smart_242_raw'].fillna(dataset['smart_242_raw'].mean())
dataset['smart_242_normalized']=dataset['smart_242_normalized'].fillna(dataset['smart_242_normalized'].mean())
dataset['smart_241_raw']=dataset['smart_241_raw'].fillna(dataset['smart_241_raw'].mean())
dataset['smart_241_normalized']=dataset['smart_241_normalized'].fillna(dataset['smart_241_normalized'].mean())
dataset['smart_240_raw']=dataset['smart_240_raw'].fillna(dataset['smart_240_raw'].mean())
dataset['smart_240_normalized']=dataset['smart_240_normalized'].fillna(dataset['smart_240_normalized'].mean())
dataset['smart_189_raw']=dataset['smart_189_raw'].fillna(dataset['smart_189_raw'].mean())
dataset['smart_189_normalized']=dataset['smart_189_normalized'].fillna(dataset['smart_189_normalized'].mean())
dataset['smart_184_raw']=dataset['smart_184_raw'].fillna(dataset['smart_184_raw'].mean())
dataset['smart_184_normalized']=dataset['smart_184_normalized'].fillna(dataset['smart_184_normalized'].mean())
dataset['smart_195_raw']=dataset['smart_195_raw'].fillna(dataset['smart_195_raw'].mean())
dataset['smart_195_normalized']=dataset['smart_195_normalized'].fillna(dataset['smart_195_normalized'].mean())
dataset['smart_191_raw']=dataset['smart_191_raw'].fillna(dataset['smart_191_raw'].mean())
dataset['smart_191_normalized']=dataset['smart_191_normalized'].fillna(dataset['smart_191_normalized'].mean())
dataset['smart_190_raw']=dataset['smart_190_raw'].fillna(dataset['smart_190_raw'].mean())
dataset['smart_190_normalized']=dataset['smart_190_normalized'].fillna(dataset['smart_190_normalized'].mean())
dataset['smart_187_normalized']=dataset['smart_187_normalized'].fillna(dataset['smart_187_normalized'].mean())
dataset['smart_187_raw']=dataset['smart_187_raw'].fillna(dataset['smart_187_raw'].mean())
dataset['smart_188_normalized']=dataset['smart_188_normalized'].fillna(dataset['smart_188_normalized'].mean())
dataset['smart_188_raw']=dataset['smart_188_raw'].fillna(dataset['smart_188_raw'].mean())
dataset['smart_8_normalized']=dataset['smart_8_normalized'].fillna(dataset['smart_8_normalized'].mean())
dataset['smart_2_normalized']=dataset['smart_2_normalized'].fillna(dataset['smart_2_normalized'].mean())
dataset['smart_2_raw']=dataset['smart_2_raw'].fillna(dataset['smart_2_raw'].mean())
dataset['smart_8_raw']=dataset['smart_8_raw'].fillna(dataset['smart_8_raw'].mean())
dataset['smart_196_normalized']=dataset['smart_196_normalized'].fillna(dataset['smart_196_normalized'].mean())
dataset['smart_196_raw']=dataset['smart_196_raw'].fillna(dataset['smart_196_raw'].mean())
dataset['smart_193_raw']=dataset['smart_193_raw'].fillna(dataset['smart_193_raw'].mean())
dataset['smart_193_normalized']=dataset['smart_193_normalized'].fillna(dataset['smart_193_normalized'].mean())
dataset['smart_197_normalized']=dataset['smart_197_normalized'].fillna(dataset['smart_197_normalized'].mean())
dataset['smart_197_raw']=dataset['smart_197_raw'].fillna(dataset['smart_197_raw'].mean())
dataset['smart_3_normalized']=dataset['smart_3_normalized'].fillna(dataset['smart_3_normalized'].mean())
dataset['smart_3_raw']=dataset['smart_3_raw'].fillna(dataset['smart_3_raw'].mean())
dataset['smart_4_normalized']=dataset['smart_4_normalized'].fillna(dataset['smart_4_normalized'].mean())
dataset['smart_4_raw']=dataset['smart_4_raw'].fillna(dataset['smart_4_raw'].mean())
dataset['smart_7_normalized']=dataset['smart_7_normalized'].fillna(dataset['smart_7_normalized'].mean())
dataset['smart_10_normalized']=dataset['smart_10_normalized'].fillna(dataset['smart_10_normalized'].mean())
dataset['smart_7_raw']=dataset['smart_7_raw'].fillna(dataset['smart_7_raw'].mean())
dataset['smart_10_raw']=dataset['smart_10_raw'].fillna(dataset['smart_10_raw'].mean())
dataset['smart_5_normalized']=dataset['smart_5_normalized'].fillna(dataset['smart_5_normalized'].mean())
dataset['smart_5_raw']=dataset['smart_5_raw'].fillna(dataset['smart_5_raw'].mean())
dataset['smart_199_raw']=dataset['smart_199_raw'].fillna(dataset['smart_199_raw'].mean())
dataset['smart_198_normalized']=dataset['smart_198_normalized'].fillna(dataset['smart_198_normalized'].mean())
dataset['smart_199_normalized']=dataset['smart_199_normalized'].fillna(dataset['smart_199_normalized'].mean())
dataset['smart_198_raw']=dataset['smart_198_raw'].fillna(dataset['smart_198_raw'].mean())
dataset['smart_192_normalized']=dataset['smart_192_normalized'].fillna(dataset['smart_192_normalized'].mean())
dataset['smart_192_raw']=dataset['smart_192_raw'].fillna(dataset['smart_192_raw'].mean())
dataset['smart_9_raw']=dataset['smart_9_raw'].fillna(dataset['smart_9_raw'].mean())
dataset['smart_1_normalized']=dataset['smart_1_normalized'].fillna(dataset['smart_1_normalized'].mean())
dataset['smart_1_raw']=dataset['smart_1_raw'].fillna(dataset['smart_1_raw'].mean())
dataset['smart_9_normalized']=dataset['smart_9_normalized'].fillna(dataset['smart_9_normalized'].mean())
dataset['smart_194_raw']=dataset['smart_194_raw'].fillna(dataset['smart_194_raw'].mean())
dataset['smart_194_normalized']=dataset['smart_194_normalized'].fillna(dataset['smart_194_normalized'].mean())
dataset['smart_12_raw']=dataset['smart_12_raw'].fillna(dataset['smart_12_raw'].mean())
dataset['smart_12_normalized']=dataset['smart_12_normalized'].fillna(dataset['smart_12_normalized'].mean())
#dataset['failure']=dataset['failure'].fillna(dataset['failure'].mean())
#dataset['capacity_bytes']=dataset['capacity_bytes'].fillna(dataset['capacity_bytes'].mean())
#dataset['model']=dataset['model'].fillna(dataset['model'].mean())
#dataset['serial_number']=dataset['serial_number'].fillna(dataset['serial_number'].mean())
#dataset['date']=dataset['date'].fillna(dataset['date'].mean())


# In[15]:


# Remove CORRILATION VARIYABLE 
# smart_226_raw - correlated with capacity_bytes
# smart_8_normalized - correlated withsmart_2_normalized
# smart_254_normalized - correlated with smart_3_normalized
# smart_12_raw correlated withsmart_4_raw
# smart_192_raw correlated with smart_4_raw
# smart_196_normalized correlated with smart_5_normalized
# smart_222_normalized correlated with smart_9_normalized
# smart_175_raw correlated with smart_9_normalized
# smart_190_normalized correlated with smart_11_raw
# smart_192_raw correlated with smart_12_raw
# smart_13_raw - correlated with capacity_bytes
# smart_177_normalized - correlated with smart_22_normalized
# smart_164_raw - correlated with smart_165_raw
# smart_167_raw - correlated withsmart_165_raw
# smart_175_raw - many
# smart_190_raw - correlated with smart_194_normalized and smart_196_normalized
# smart_202_raw - correlated with smart_180_raw and smart_194_normalized
# smart_226_raw - correlated with smart_3_raw
# smart_254_normalized - correlated with many


dataset = dataset.drop(['smart_226_raw'], 1)
dataset = dataset.drop(['smart_8_normalized'], 1)
dataset = dataset.drop(['smart_254_normalized'], 1)
dataset = dataset.drop(['smart_12_raw'], 1)
dataset = dataset.drop(['smart_192_raw'], 1)
dataset = dataset.drop(['smart_196_normalized'], 1)
dataset = dataset.drop(['smart_222_normalized'], 1)
dataset = dataset.drop(['smart_175_raw'], 1)
dataset = dataset.drop(['smart_190_normalized'], 1)
dataset = dataset.drop(['smart_13_raw'], 1)
dataset = dataset.drop(['smart_177_normalized'], 1)
#dataset = dataset.drop(['smart_164_raw'], 1)

#dataset = dataset.drop(['smart_167_raw'], 1)
dataset = dataset.drop(['smart_190_raw'], 1)
dataset = dataset.drop(['smart_202_raw'], 1)

 


# In[16]:


# drop date
dataset = dataset.drop(['date'], 1)


# In[17]:


# remove top 10 missing variables
dataset = dataset.drop(['smart_255_raw'], 1)
dataset = dataset.drop(['smart_15_normalized'], 1)
dataset = dataset.drop(['smart_234_raw'], 1)
dataset = dataset.drop(['smart_255_normalized'], 1)
dataset = dataset.drop(['smart_15_raw'], 1)
dataset = dataset.drop(['smart_234_normalized'], 1)
dataset = dataset.drop(['smart_206_normalized'], 1)
dataset = dataset.drop(['smart_206_raw'], 1)
dataset = dataset.drop(['smart_248_raw'], 1)
dataset = dataset.drop(['smart_248_normalized'], 1)


# In[18]:


# Remove below variable as it has only one value 

dataset = dataset.drop(['smart_210_raw'], 1)
dataset = dataset.drop(['smart_224_raw'], 1)
dataset = dataset.drop(['smart_18_raw'], 1)
dataset = dataset.drop(['smart_23_raw'], 1)
dataset = dataset.drop(['smart_24_raw'], 1)
dataset = dataset.drop(['smart_166_raw'], 1)
dataset = dataset.drop(['smart_176_raw'], 1)
dataset = dataset.drop(['smart_179_raw'], 1)
dataset = dataset.drop(['smart_181_raw'], 1)
dataset = dataset.drop(['smart_182_raw'], 1)
dataset = dataset.drop(['smart_251_normalized'], 1)
dataset = dataset.drop(['smart_250_normalized'], 1)
dataset = dataset.drop(['smart_254_raw'], 1)


# In[19]:


# normalize the data
dataset['normalized_capacity_bytes'] = (dataset['capacity_bytes'] - dataset['capacity_bytes'].min()) / (dataset['capacity_bytes'].max() - dataset['capacity_bytes'].min())
dataset['normalized_smart_1_raw'] = (dataset['smart_1_raw'] - dataset['smart_1_raw'].min()) / (dataset['smart_1_raw'].max() - dataset['smart_1_raw'].min())
dataset['normalized_smart_241_raw'] = (dataset['smart_241_raw'] - dataset['smart_241_raw'].min()) / (dataset['smart_241_raw'].max() - dataset['smart_241_raw'].min())
dataset['normalized_smart_242_raw'] = (dataset['smart_242_raw'] - dataset['smart_242_raw'].min()) / (dataset['smart_242_raw'].max() - dataset['smart_242_raw'].min())
dataset['normalized_smart_7_raw'] = (dataset['smart_7_raw'] - dataset['smart_7_raw'].min()) / (dataset['smart_7_raw'].max() - dataset['smart_7_raw'].min())
dataset['normalized_smart_9_raw'] = (dataset['smart_9_raw'] - dataset['smart_9_raw'].min()) / (dataset['smart_9_raw'].max() - dataset['smart_9_raw'].min())
dataset['normalized_smart_11_raw'] = (dataset['smart_11_raw'] - dataset['smart_11_raw'].min()) / (dataset['smart_11_raw'].max() - dataset['smart_11_raw'].min())
dataset['normalized_smart_173_raw'] = (dataset['smart_173_raw'] - dataset['smart_173_raw'].min()) / (dataset['smart_173_raw'].max() - dataset['smart_173_raw'].min())
dataset['normalized_smart_188_raw'] = (dataset['smart_188_raw'] - dataset['smart_188_raw'].min()) / (dataset['smart_188_raw'].max() - dataset['smart_188_raw'].min())
dataset['normalized_smart_193_raw'] = (dataset['smart_193_raw'] - dataset['smart_193_raw'].min()) / (dataset['smart_193_raw'].max() - dataset['smart_193_raw'].min())
dataset['normalized_smart_195_raw'] = (dataset['smart_195_raw'] - dataset['smart_195_raw'].min()) / (dataset['smart_195_raw'].max() - dataset['smart_195_raw'].min())
dataset['normalized_smart_225_raw'] = (dataset['smart_225_raw'] - dataset['smart_225_raw'].min()) / (dataset['smart_225_raw'].max() - dataset['smart_225_raw'].min())
dataset['normalized_smart_232_raw'] = (dataset['smart_232_raw'] - dataset['smart_232_raw'].min()) / (dataset['smart_232_raw'].max() - dataset['smart_232_raw'].min())
dataset['normalized_smart_233_raw'] = (dataset['smart_233_raw'] - dataset['smart_233_raw'].min()) / (dataset['smart_233_raw'].max() - dataset['smart_233_raw'].min())
dataset['normalized_smart_235_raw'] = (dataset['smart_235_raw'] - dataset['smart_235_raw'].min()) / (dataset['smart_235_raw'].max() - dataset['smart_235_raw'].min())
dataset['normalized_smart_240_raw'] = (dataset['smart_240_raw'] - dataset['smart_240_raw'].min()) / (dataset['smart_240_raw'].max() - dataset['smart_240_raw'].min())
dataset['normalized_smart_247_raw'] = (dataset['smart_247_raw'] - dataset['smart_247_raw'].min()) / (dataset['smart_247_raw'].max() - dataset['smart_247_raw'].min())
dataset['normalized_smart_251_raw'] = (dataset['smart_251_raw'] - dataset['smart_251_raw'].min()) / (dataset['smart_251_raw'].max() - dataset['smart_251_raw'].min())
dataset['normalized_smart_252_raw'] = (dataset['smart_252_raw'] - dataset['smart_252_raw'].min()) / (dataset['smart_252_raw'].max() - dataset['smart_252_raw'].min())
dataset['normalized_smart_5_raw'] = (dataset['smart_5_raw'] - dataset['smart_5_raw'].min()) / (dataset['smart_5_raw'].max() - dataset['smart_5_raw'].min())
dataset['normalized_smart_197_raw'] = (dataset['smart_197_raw'] - dataset['smart_197_raw'].min()) / (dataset['smart_197_raw'].max() - dataset['smart_197_raw'].min())
dataset['normalized_smart_10_raw'] = (dataset['smart_10_raw'] - dataset['smart_10_raw'].min()) / (dataset['smart_10_raw'].max() - dataset['smart_10_raw'].min())
dataset['normalized_smart_198_raw'] = (dataset['smart_198_raw'] - dataset['smart_198_raw'].min()) / (dataset['smart_198_raw'].max() - dataset['smart_198_raw'].min())



# In[20]:


# drop after normalization 
dataset = dataset.drop(['capacity_bytes'], 1)
dataset = dataset.drop(['smart_1_raw'], 1)
dataset = dataset.drop(['smart_241_raw'], 1)
dataset = dataset.drop(['smart_242_raw'], 1)
dataset = dataset.drop(['smart_7_raw'], 1)
dataset = dataset.drop(['smart_9_raw'], 1)
dataset = dataset.drop(['smart_11_raw'], 1)
dataset = dataset.drop(['smart_173_raw'], 1)
dataset = dataset.drop(['smart_188_raw'], 1)
dataset = dataset.drop(['smart_193_raw'], 1)
dataset = dataset.drop(['smart_195_raw'], 1)
dataset = dataset.drop(['smart_225_raw'], 1)
dataset = dataset.drop(['smart_232_raw'], 1)
dataset = dataset.drop(['smart_233_raw'], 1)
dataset = dataset.drop(['smart_235_raw'], 1)
dataset = dataset.drop(['smart_240_raw'], 1)
dataset = dataset.drop(['smart_247_raw'], 1)
dataset = dataset.drop(['smart_251_raw'], 1)
dataset = dataset.drop(['smart_252_raw'], 1)
dataset = dataset.drop(['smart_5_raw'], 1)
dataset = dataset.drop(['smart_197_raw'], 1)
dataset = dataset.drop(['smart_10_raw'], 1)
dataset = dataset.drop(['smart_198_raw'], 1)


# In[21]:


# one hot encoding or label encoder or creating dummies for all the categorical data
# Label Encoding model

dataset.model = pd.Categorical(dataset.model)
dataset["model"] = dataset["model"].cat.codes


# In[22]:


# define the y variable 
target = 'failure'   #defining a global variable


# In[23]:


# get only serial number form test data
df_test_serial_number=dataset.serial_number
#df_test_failure=df_test.failure


# In[24]:


# fill missing with zero 
dataset.fillna(0)
dataset.replace(np.nan, 0)
dataset.fillna(value = 0,
          inplace = True)
dataset.replace(np.NaN, 0)


# In[25]:


# Machine learning algorithms require that the distribution of our data is uniform or normal
qtrans = QuantileTransformer(output_distribution='normal', random_state=0)
dataset['smart_1_normalized'] = qtrans.fit_transform(dataset[['smart_1_normalized']])
dataset['smart_2_normalized'] = qtrans.fit_transform(dataset[['smart_2_normalized']])
dataset['smart_2_raw'] = qtrans.fit_transform(dataset[['smart_2_raw']])
dataset['smart_3_normalized'] = qtrans.fit_transform(dataset[['smart_3_normalized']])
dataset['smart_3_raw'] = qtrans.fit_transform(dataset[['smart_3_raw']])
dataset['smart_4_normalized'] = qtrans.fit_transform(dataset[['smart_4_normalized']])
dataset['smart_4_raw'] = qtrans.fit_transform(dataset[['smart_4_raw']])
dataset['smart_5_normalized'] = qtrans.fit_transform(dataset[['smart_5_normalized']])
dataset['smart_7_normalized'] = qtrans.fit_transform(dataset[['smart_7_normalized']])
dataset['smart_8_raw'] = qtrans.fit_transform(dataset[['smart_8_raw']])
dataset['smart_9_normalized'] = qtrans.fit_transform(dataset[['smart_9_normalized']])
dataset['smart_10_normalized'] = qtrans.fit_transform(dataset[['smart_10_normalized']])
dataset['smart_11_normalized'] = qtrans.fit_transform(dataset[['smart_11_normalized']])
dataset['smart_12_normalized'] = qtrans.fit_transform(dataset[['smart_12_normalized']])
dataset['smart_13_normalized'] = qtrans.fit_transform(dataset[['smart_13_normalized']])
dataset['smart_16_normalized'] = qtrans.fit_transform(dataset[['smart_16_normalized']])
dataset['smart_16_raw'] = qtrans.fit_transform(dataset[['smart_16_raw']])
dataset['smart_17_normalized'] = qtrans.fit_transform(dataset[['smart_17_normalized']])
dataset['smart_17_raw'] = qtrans.fit_transform(dataset[['smart_17_raw']])
dataset['smart_18_normalized'] = qtrans.fit_transform(dataset[['smart_18_normalized']])
dataset['smart_22_normalized'] = qtrans.fit_transform(dataset[['smart_22_normalized']])
dataset['smart_22_raw'] = qtrans.fit_transform(dataset[['smart_22_raw']])
dataset['smart_23_normalized'] = qtrans.fit_transform(dataset[['smart_23_normalized']])
dataset['smart_24_normalized'] = qtrans.fit_transform(dataset[['smart_24_normalized']])
dataset['smart_160_normalized'] = qtrans.fit_transform(dataset[['smart_160_normalized']])
dataset['smart_160_raw'] = qtrans.fit_transform(dataset[['smart_160_raw']])
dataset['smart_161_normalized'] = qtrans.fit_transform(dataset[['smart_161_normalized']])
dataset['smart_161_raw'] = qtrans.fit_transform(dataset[['smart_161_raw']])
dataset['smart_163_normalized'] = qtrans.fit_transform(dataset[['smart_163_normalized']])
dataset['smart_163_raw'] = qtrans.fit_transform(dataset[['smart_163_raw']])
dataset['smart_164_normalized'] = qtrans.fit_transform(dataset[['smart_164_normalized']])
dataset['smart_165_normalized'] = qtrans.fit_transform(dataset[['smart_165_normalized']])
dataset['smart_165_raw'] = qtrans.fit_transform(dataset[['smart_165_raw']])
#dataset['smart_166_normalized'] = qtrans.fit_transform(dataset[['smart_166_normalized']])
dataset['smart_167_normalized'] = qtrans.fit_transform(dataset[['smart_167_normalized']])
dataset['smart_168_normalized'] = qtrans.fit_transform(dataset[['smart_168_normalized']])
dataset['smart_168_raw'] = qtrans.fit_transform(dataset[['smart_168_raw']])
dataset['smart_169_normalized'] = qtrans.fit_transform(dataset[['smart_169_normalized']])
dataset['smart_169_raw'] = qtrans.fit_transform(dataset[['smart_169_raw']])
dataset['smart_170_normalized'] = qtrans.fit_transform(dataset[['smart_170_normalized']])
dataset['smart_170_raw'] = qtrans.fit_transform(dataset[['smart_170_raw']])
dataset['smart_173_normalized'] = qtrans.fit_transform(dataset[['smart_173_normalized']])
dataset['smart_174_normalized'] = qtrans.fit_transform(dataset[['smart_174_normalized']])
dataset['smart_174_raw'] = qtrans.fit_transform(dataset[['smart_174_raw']])
dataset['smart_175_normalized'] = qtrans.fit_transform(dataset[['smart_175_normalized']])
dataset['smart_176_normalized'] = qtrans.fit_transform(dataset[['smart_176_normalized']])
dataset['smart_177_raw'] = qtrans.fit_transform(dataset[['smart_177_raw']])
dataset['smart_178_normalized'] = qtrans.fit_transform(dataset[['smart_178_normalized']])
dataset['smart_178_raw'] = qtrans.fit_transform(dataset[['smart_178_raw']])
dataset['smart_179_normalized'] = qtrans.fit_transform(dataset[['smart_179_normalized']])
dataset['smart_180_normalized'] = qtrans.fit_transform(dataset[['smart_180_normalized']])
dataset['smart_180_raw'] = qtrans.fit_transform(dataset[['smart_180_raw']])
dataset['smart_181_normalized'] = qtrans.fit_transform(dataset[['smart_181_normalized']])
dataset['smart_182_normalized'] = qtrans.fit_transform(dataset[['smart_182_normalized']])
dataset['smart_183_normalized'] = qtrans.fit_transform(dataset[['smart_183_normalized']])
dataset['smart_183_raw'] = qtrans.fit_transform(dataset[['smart_183_raw']])
dataset['smart_184_normalized'] = qtrans.fit_transform(dataset[['smart_184_normalized']])
dataset['smart_184_raw'] = qtrans.fit_transform(dataset[['smart_184_raw']])
dataset['smart_187_normalized'] = qtrans.fit_transform(dataset[['smart_187_normalized']])
dataset['smart_187_raw'] = qtrans.fit_transform(dataset[['smart_187_raw']])
dataset['smart_188_normalized'] = qtrans.fit_transform(dataset[['smart_188_normalized']])
dataset['smart_189_normalized'] = qtrans.fit_transform(dataset[['smart_189_normalized']])
dataset['smart_189_raw'] = qtrans.fit_transform(dataset[['smart_189_raw']])
dataset['smart_191_normalized'] = qtrans.fit_transform(dataset[['smart_191_normalized']])
dataset['smart_191_raw'] = qtrans.fit_transform(dataset[['smart_191_raw']])
dataset['smart_192_normalized'] = qtrans.fit_transform(dataset[['smart_192_normalized']])
dataset['smart_193_normalized'] = qtrans.fit_transform(dataset[['smart_193_normalized']])
dataset['smart_194_normalized'] = qtrans.fit_transform(dataset[['smart_194_normalized']])
dataset['smart_194_raw'] = qtrans.fit_transform(dataset[['smart_194_raw']])
dataset['smart_195_normalized'] = qtrans.fit_transform(dataset[['smart_195_normalized']])
dataset['smart_196_raw'] = qtrans.fit_transform(dataset[['smart_196_raw']])
dataset['smart_197_normalized'] = qtrans.fit_transform(dataset[['smart_197_normalized']])
dataset['smart_198_normalized'] = qtrans.fit_transform(dataset[['smart_198_normalized']])
dataset['smart_199_normalized'] = qtrans.fit_transform(dataset[['smart_199_normalized']])
dataset['smart_199_raw'] = qtrans.fit_transform(dataset[['smart_199_raw']])
dataset['smart_200_normalized'] = qtrans.fit_transform(dataset[['smart_200_normalized']])
dataset['smart_200_raw'] = qtrans.fit_transform(dataset[['smart_200_raw']])
dataset['smart_201_normalized'] = qtrans.fit_transform(dataset[['smart_201_normalized']])
dataset['smart_201_raw'] = qtrans.fit_transform(dataset[['smart_201_raw']])
dataset['smart_202_normalized'] = qtrans.fit_transform(dataset[['smart_202_normalized']])
dataset['smart_210_normalized'] = qtrans.fit_transform(dataset[['smart_210_normalized']])
dataset['smart_218_normalized'] = qtrans.fit_transform(dataset[['smart_218_normalized']])
dataset['smart_218_raw'] = qtrans.fit_transform(dataset[['smart_218_raw']])
dataset['smart_220_normalized'] = qtrans.fit_transform(dataset[['smart_220_normalized']])
dataset['smart_220_raw'] = qtrans.fit_transform(dataset[['smart_220_raw']])
dataset['smart_222_raw'] = qtrans.fit_transform(dataset[['smart_222_raw']])
dataset['smart_223_normalized'] = qtrans.fit_transform(dataset[['smart_223_normalized']])
dataset['smart_223_raw'] = qtrans.fit_transform(dataset[['smart_223_raw']])
dataset['smart_224_normalized'] = qtrans.fit_transform(dataset[['smart_224_normalized']])
dataset['smart_225_normalized'] = qtrans.fit_transform(dataset[['smart_225_normalized']])
dataset['smart_226_normalized'] = qtrans.fit_transform(dataset[['smart_226_normalized']])
dataset['smart_231_normalized'] = qtrans.fit_transform(dataset[['smart_231_normalized']])
dataset['smart_231_raw'] = qtrans.fit_transform(dataset[['smart_231_raw']])
dataset['smart_232_normalized'] = qtrans.fit_transform(dataset[['smart_232_normalized']])
dataset['smart_233_normalized'] = qtrans.fit_transform(dataset[['smart_233_normalized']])
dataset['smart_235_normalized'] = qtrans.fit_transform(dataset[['smart_235_normalized']])
dataset['smart_240_normalized'] = qtrans.fit_transform(dataset[['smart_240_normalized']])
dataset['smart_241_normalized'] = qtrans.fit_transform(dataset[['smart_241_normalized']])
dataset['smart_242_normalized'] = qtrans.fit_transform(dataset[['smart_242_normalized']])
dataset['smart_245_normalized'] = qtrans.fit_transform(dataset[['smart_245_normalized']])
dataset['smart_245_raw'] = qtrans.fit_transform(dataset[['smart_245_raw']])
dataset['smart_247_normalized'] = qtrans.fit_transform(dataset[['smart_247_normalized']])
dataset['smart_250_raw'] = qtrans.fit_transform(dataset[['smart_250_raw']])
dataset['smart_252_normalized'] = qtrans.fit_transform(dataset[['smart_252_normalized']])


# In[26]:


# remove max VIF variables 
dataset = dataset.drop(['smart_170_raw'], 1)
dataset = dataset.drop(['smart_180_normalized'], 1)
dataset = dataset.drop(['smart_180_raw'], 1)
dataset = dataset.drop(['smart_201_normalized'], 1)
dataset = dataset.drop(['smart_201_raw'], 1)
dataset = dataset.drop(['smart_218_raw'], 1)
dataset = dataset.drop(['smart_16_raw'], 1)
dataset = dataset.drop(['smart_17_raw'], 1)
dataset = dataset.drop(['smart_245_raw'], 1)
dataset = dataset.drop(['smart_169_raw'], 1)


# In[27]:


# delete
sns.countplot(dataset['failure'])


# In[28]:


# Up-sampling the minority class for test data 
valid_test = dataset[dataset['failure'] == 0]    #data of HDDs which do not indicate failure
failed_test = dataset[dataset['failure'] == 1]   #data of HDDs likely to fail


# In[29]:


print("valid hdds:",len(valid_test))      #storing the total number of valid HDDs
print("failing hdds:",len(failed_test))


# In[30]:


#features selection
features = ['serial_number',
'model',
'failure',
'smart_1_normalized',
'smart_2_normalized',
'smart_2_raw',
'smart_3_normalized',
'smart_3_raw',
'smart_4_normalized',
'smart_4_raw',
'smart_5_normalized',
'smart_7_normalized',
'smart_8_raw',
'smart_9_normalized',
'smart_10_normalized',
'smart_11_normalized',
'smart_12_normalized',
'smart_13_normalized',
'smart_16_normalized',
'smart_17_normalized',
'smart_18_normalized',
'smart_22_normalized',
'smart_22_raw',
'smart_23_normalized',
'smart_24_normalized',
'smart_160_normalized',
'smart_160_raw',
'smart_161_normalized',
'smart_161_raw',
'smart_163_normalized',
'smart_163_raw',
'smart_164_normalized',
'smart_165_normalized',
'smart_165_raw',
'smart_166_normalized',
'smart_167_normalized',
'smart_168_normalized',
'smart_168_raw',
'smart_169_normalized',
'smart_170_normalized',
'smart_173_normalized',
'smart_174_normalized',
'smart_174_raw',
'smart_175_normalized',
'smart_176_normalized',
'smart_177_raw',
'smart_178_normalized',
'smart_178_raw',
'smart_179_normalized',
'smart_181_normalized',
'smart_182_normalized',
'smart_183_normalized',
'smart_183_raw',
'smart_184_normalized',
'smart_184_raw',
'smart_187_normalized',
'smart_187_raw',
'smart_188_normalized',
'smart_189_normalized',
'smart_189_raw',
'smart_191_normalized',
'smart_191_raw',
'smart_192_normalized',
'smart_193_normalized',
'smart_194_normalized',
'smart_194_raw',
'smart_195_normalized',
'smart_196_raw',
'smart_197_normalized',
'smart_198_normalized',
'smart_199_normalized',
'smart_199_raw',
'smart_200_normalized',
'smart_200_raw',
'smart_202_normalized',
'smart_210_normalized',
'smart_218_raw',
'smart_220_normalized',
'smart_220_raw',
'smart_222_raw',
'smart_223_normalized',
'smart_223_raw',
'smart_224_normalized',
'smart_225_normalized',
'smart_226_normalized',
'smart_231_normalized',
'smart_231_raw',
'smart_232_normalized',
'smart_233_normalized',
'smart_235_normalized',
'smart_240_normalized',
'smart_241_normalized',
'smart_242_normalized',
'smart_245_normalized',
'smart_247_normalized',
'smart_250_raw',
'smart_252_normalized',
'normalized_capacity_bytes',
'normalized_smart_1_raw',
'normalized_smart_241_raw',
'normalized_smart_242_raw',
'normalized_smart_7_raw',
'normalized_smart_9_raw',
'normalized_smart_11_raw',
'normalized_smart_173_raw',
'normalized_smart_188_raw',
'normalized_smart_193_raw',
'normalized_smart_195_raw',
'normalized_smart_225_raw',
'normalized_smart_232_raw',
'normalized_smart_233_raw',
'normalized_smart_235_raw',
'normalized_smart_240_raw',
'normalized_smart_247_raw',
'normalized_smart_251_raw',
'normalized_smart_252_raw',
'normalized_smart_5_raw',
'normalized_smart_197_raw',
'normalized_smart_10_raw',
'normalized_smart_198_raw'
]


# In[31]:


# drop the remaining features
misc_feat = [fname for fname in dataset if fname not in features]  #misc features to be dropped 


# In[32]:


# drop the remaining features
dataset.drop(misc_feat,inplace=True,axis=1)


# In[33]:


# model cannot process string values, we remove the columns # to avoid errors 
obj = dataset.dtypes[dataset.dtypes == object ].index  


# In[34]:


# drop the sting data 
dataset = dataset.drop(obj,axis=1)


# In[35]:


dataset['failure'].value_counts()


# In[36]:


#Splitting the values for X_test and Y_test
X_test = dataset.drop('failure',axis=1)
Y_test = dataset['failure']


# In[37]:


# fill missing with zero 
X_test.fillna(0)
X_test.replace(np.nan, 0)
X_test.fillna(value = 0,
          inplace = True)
X_test.replace(np.NaN, 0)


# In[38]:


import pickle
# events1.disk_smartdata_dtls
model_filename = "My_rfc_model.sav"
my_knn_model = pickle.load(open(model_filename, 'rb'))
result = my_knn_model.predict(X_test)
result
Model_Prediction = pd.DataFrame(result, columns =['Model_Prediction'])
#e_dataframe = pd.DataFrame(e) 
#Probability_of_failure = pd.DataFrame(y_pred_prob)
df_test_serial_number=pd.DataFrame(df_test_serial_number)
Model_Prediction['row_num'] = np.arange(len(Model_Prediction))
df_test_serial_number['row_num'] = np.arange(len(df_test_serial_number))
#result = pd.concat([Model_Prediction, df_test_serial_number], axis=1)
# Stack the Data Frames on top of each other
Model_Prediction.Model_Prediction.replace((0, 1), ('not_fail', 'Predicted_to_be_fail'), inplace=True)
RandomForest_Model_Prediction=df_test_serial_number.merge(Model_Prediction, on='row_num', how='left')
RandomForest_Model_Prediction = RandomForest_Model_Prediction.drop(['row_num'], 1)
RandomForest_Model_Prediction
#3 DELETE


# In[39]:


# Download descriptive analysis
#final_colums1=dataset.describe()
RandomForest_Model_Prediction.to_csv(r'E:\work\Supermicro\modeling\data_Q2_2019\RandomForest_Model_Prediction.csv')


# In[147]:


###
from sklearn.ensemble import RandomForestClassifier 
import pickle
#  model creation 
rfc = RandomForestClassifier() 
y_pred_prob = rfc.predict_proba(X_test)
Probability_of_failure = pd.DataFrame(y_pred_prob, columns =['Active_Probability','Failure_Probability'])
#e_dataframe = pd.DataFrame(e) 
#Probability_of_failure = pd.DataFrame(y_pred_prob)
Probability_of_failure
#3 yPred_df_yPred_validation2
# validation2
result = pd.concat([RandomForest_Model_Prediction, Probability_of_failure], axis=1)
result


# In[43]:


from sklearn.ensemble import RandomForestClassifier 

import pickle
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

#  model creation 
rfc = RandomForestClassifier() 
best_rfc = rfc
best_rfc = grid_search
best_rfc = grid_search.best_estimator_


# In[148]:


#1 finalized_model.sav
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

import pickle
#  model creation 
rfc = RandomForestClassifier() 
# save the model to disk
#filename = 'finalized_model.sav'
###pickle.dump(rfc, open(filename, 'wb'))
#import pickle
#loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

# accuraccy score with test data
######accuraccy_result = loaded_model.score(X_train, Y_train)
#print(result)
# final prediction results 
prediction_result=loaded_model.predict(X_test)
prediction_result
#Model_Prediction dataframe
Model_Prediction = pd.DataFrame(prediction_result, columns =['Model_Prediction'])
#print("\nPandas DataFrame: ")
Model_Prediction


df_test_serial_number=pd.DataFrame(df_test_serial_number)
Model_Prediction['row_num'] = np.arange(len(Model_Prediction))
df_test_serial_number['row_num'] = np.arange(len(df_test_serial_number))
#result = pd.concat([Model_Prediction, df_test_serial_number], axis=1)
# Stack the Data Frames on top of each other
Model_Prediction.Model_Prediction.replace((0, 1), ('not_fail', 'Predicted_to_be_fail'), inplace=True)
RandomForest_Model_Prediction=df_test_serial_number.merge(Model_Prediction, on='row_num', how='left')
RandomForest_Model_Prediction = RandomForest_Model_Prediction.drop(['row_num'], 1)
RandomForest_Model_Prediction
#3 DELETE


# In[41]:


#  model creation 
rfc = RandomForestClassifier() 
#rfc.fit(X_train, Y_train) 
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(rfc, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)


# In[ ]:




