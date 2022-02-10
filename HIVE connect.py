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
import sasl
import thrift_sasl
from pyhive import hive
#from impala.dbapi import connect
from hdfs import InsecureClient
from pyhive import hive
from sqlalchemy import Column, String, Integer
import serial
import time
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from passlib.apps import custom_app_context as pwd_context
from tempfile import gettempdir


# In[3]:


#cursor = conn.cursor()
#cursor.execute('SELECT * FROM test_parquet')
#print(cursor.execute) 
# prints the result set's schema
#results = cursor.fetchall() # default.client_failure1_data_hive
conn = hive.Connection(host='172.27.27.60', port=10000, password='Hive#Pa55', username='smicro', auth='CUSTOM')
#dataframe = pd.read_sql("SELECT * FROM failure1_data_hive", conn)
dataframe = pd.read_sql("SELECT * FROM demo_client_data_hive", conn)
dataframe


# In[4]:


dataframe.columns


# In[2]:


################################## 2nd methord usig spark
from pyspark import SparkContext
sc =SparkContext()
from pyspark.sql import HiveContext
hive_context = HiveContext(sc)
data = hive_context.table("demo_client_data_hive")
data.show()


# In[6]:


data.name


# In[62]:


data.dtypes


# In[3]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.master('yarn').appName('pythonSpark').enableHiveSupport().getOrCreate()


# In[ ]:


df1 = spark.read.load('/data/employees1.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
df1.write.format("PARQUET").saveAsTable("default.test_parquet")
df2 = spark.read.load('/data/employees2.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
df2.write.mode("append").saveAsTable("default.test_parquet")

