#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import psycopg2

  


# In[ ]:


# get the final results
RandomForest_Model_Prediction


# In[ ]:


# Delete the existing table in SQL DB
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
psqlCon         = psycopg2.connect("dbname=rawData user=data_user password=kgtopg8932");
psqlCon.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT);
psqlCursor      = psqlCon.cursor();
tableName       = "ml_prediction";
dropTableStmt   = "DROP TABLE %s;"%tableName;
psqlCursor.execute(dropTableStmt);


# In[ ]:


#upload new table in to SQL DB
from sqlalchemy import create_engine
engine = create_engine('postgresql://data_user:kgtopg8932@localhost:5432/rawData')
RandomForest_Model_Prediction.to_sql('ml_prediction', engine)


# In[ ]:


# CHECK THE TABLE 
query1 = "select * from ml_prediction" 
dataset = sqlio.read_sql_query(query1,conn)
dataset


# In[ ]:


# CHECK THE TABLE 
query1 = "select * from ml_prediction" 


# In[ ]:


dataset = sqlio.read_sql_query(query1,conn)
dataset

