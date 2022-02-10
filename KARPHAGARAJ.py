#!/usr/bin/env python
# coding: utf-8

# In[1]:


#STEP-1 Import libraries 
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
import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql 
from pyspark.ml.classification import MultilayerPerceptronClassifier
sc =SparkContext.getOrCreate()
sqlContext = SQLContext(sc)


# In[2]:


#STEP-2 Read data from HIVE
from pyspark.sql import HiveContext
from pyspark import SparkConf
spark = SparkSession(sc)
hive_context = HiveContext(sc)
sqlContext = SQLContext(sc)
#bank1 = sqlContext.table("sda_hdd_db.sda_hdd_all")
df = sqlContext.sql("SELECT * FROM sda_hdd_db.ml_smart_data_view")


# In[3]:


#STEP-3 fill missing with zero 
df=df.na.fill(value=0)


# In[4]:


df.printSchema()


# In[5]:


df.count()


# In[6]:


#1. data1= create diffrent dataset for 1 failure from df (TABLE ml_smart_data_view)

df1 = df.where(df.failure == 1)


# In[9]:


df1.count()


# In[17]:


sqlContext.sql("select cast(process_date as date) FROM sda_hdd_db.ml_smart_data_view limit 1").show()


# In[32]:


#3. data2= create diffrent dataset for 0 failure from df (TABLE ml_smart_data_view) where time line is 0 for each hourse 1 entry where failure = 0

df2 = sqlContext.sql("select process_date,serial_number,model,capacity_bytes,failure,smart_1_normalized,smart_1_raw,smart_2_normalized,smart_2_raw,smart_3_normalized,smart_3_raw,smart_4_normalized,smart_4_raw,smart_5_normalized,smart_5_raw,smart_7_normalized,smart_7_raw,smart_8_normalized,smart_8_raw,smart_9_normalized,smart_9_raw,smart_10_normalized,smart_10_raw,smart_11_normalized,smart_11_raw,smart_12_normalized,smart_12_raw,smart_13_normalized,smart_13_raw,smart_15_normalized,smart_15_raw,smart_16_normalized,smart_16_raw,smart_17_normalized,smart_17_raw,smart_18_normalized,smart_18_raw,smart_22_normalized,smart_22_raw,smart_23_normalized,smart_23_raw,smart_24_normalized,smart_24_raw,smart_160_normalized,smart_160_raw,smart_161_normalized,smart_161_raw,smart_163_normalized,smart_163_raw,smart_164_normalized,smart_164_raw,smart_165_normalized,smart_165_raw,smart_166_normalized,smart_166_raw,smart_167_normalized,smart_167_raw,smart_168_normalized,smart_168_raw,smart_169_normalized,smart_169_raw,smart_170_normalized,smart_170_raw,smart_173_normalized,smart_173_raw,smart_174_normalized,smart_174_raw,smart_175_normalized,smart_175_raw,smart_176_normalized,smart_176_raw,smart_177_normalized,smart_177_raw,smart_178_normalized,smart_178_raw,smart_179_normalized,smart_179_raw,smart_180_normalized,smart_180_raw,smart_181_normalized,smart_181_raw,smart_182_normalized,smart_182_raw,smart_183_normalized,smart_183_raw,smart_184_normalized,smart_184_raw,smart_187_normalized,smart_187_raw,smart_188_normalized,smart_188_raw,smart_189_normalized,smart_189_raw,smart_190_normalized,smart_190_raw,smart_191_normalized,smart_191_raw,smart_192_normalized,smart_192_raw,smart_193_normalized,smart_193_raw,smart_194_normalized,smart_194_raw,smart_195_normalized,smart_195_raw,smart_196_normalized,smart_196_raw,smart_197_normalized,smart_197_raw,smart_198_normalized,smart_198_raw,smart_199_normalized,smart_199_raw,smart_200_normalized,smart_200_raw,smart_201_normalized,smart_201_raw,smart_202_normalized,smart_202_raw,smart_206_normalized,smart_206_raw,smart_210_normalized,smart_210_raw,smart_218_normalized,smart_218_raw,smart_220_normalized,smart_220_raw,smart_222_normalized,smart_222_raw,smart_223_normalized,smart_223_raw,smart_224_normalized,smart_224_raw,smart_225_normalized,smart_225_raw,smart_226_normalized,smart_226_raw,smart_231_normalized,smart_231_raw,smart_232_normalized,smart_232_raw,smart_233_normalized,smart_233_raw,smart_234_normalized,smart_234_raw,smart_235_normalized,smart_235_raw,smart_240_normalized,smart_240_raw,smart_241_normalized,smart_241_raw,smart_242_normalized,smart_242_raw,smart_245_normalized,smart_245_raw,smart_247_normalized,smart_247_raw,smart_248_normalized,smart_248_raw,smart_250_normalized,smart_250_raw,smart_251_normalized,smart_251_raw,smart_252_normalized,smart_252_raw,smart_254_normalized,smart_254_raw,smart_255_normalized,smart_255_raw from (select *, row_number() OVER (PARTITION BY cast(process_date as date) ORDER BY model DESC) as rn  FROM sda_hdd_db.ml_smart_data_view) tmp where rn = 1 and failure=0")


# In[33]:


df2.count()


# In[26]:


df2.show()


# In[34]:


#4. data3=merge data1 & data2

df3 = df1.union(df2)


# In[31]:


df3.count()


# In[ ]:


#1. data1= create diffrent dataset for 1 failure from df (TABLE ml_smart_data_view)
#2. check the timeframe for Filter 1 is all are added in 0 time line of each hourse LET ME KNOW THE TIME FRAME 
#3. data2= create diffrent dataset for 0 failure from df (TABLE ml_smart_data_view) where time line is 0 for each hourse 1 entry where failure = 0
#4. data3=merge data1 & data2
#5. data3 should be in deffrent table it shout stream automaticaly (data3 will be used for ML model)


# In[ ]:


# sample code to filter 0 time frame data in each houre

#dataset10 = pd.read_sql('''select process_date, serial_number, model from events1.disk_smartdata_dtls where extract (hour from process_date) || ":" || extract (minute from process_date) in ('1:0', '2:0', '3:0')''', conn)
#dataset10 = pd.read_sql('''select process_date, serial_number, model from events1.disk_smartdata_dtls where extract (minute from process_date) in ('0')  limit 10''', conn)
#dataset10 = pd.read_sql('''select process_date, serial_number, model from events1.disk_smartdata_dtls where extract (minute from process_date) in ('0')''', conn)
#dataset10


# In[ ]:


#### order by df in decenting oder using process_date column
df

