#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from findspark import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark import SparkContext
import pyspark.sql 

sc =SparkContext.getOrCreate()
#sc =SparkContext()
sqlContext = SQLContext(sc)
#from pyspark.sql import SQLContext
#sqlContext = SQLContext(sc)
#spark = SparkSession.builder.appName('ml-bank').getOrCreate()
#df = spark.read.csv('bank.csv', header = True, inferSchema = True)
#df.printSchema()


# In[10]:


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


sqlctx = SQLContext(sc)
df = sqlctx.load(
    url = "jdbc:postgresql://[hostname]/[metastore]",
    dbtable = "(SELECT * FROM sda_hdd_db.sda_hdd_all LIMIT 10) as blah",
    password = "Hive#Pa55",
    user =  "smicro",
    source = "jdbc",
    driver = "org.postgresql.Driver"
)


# In[16]:


sqlctx = SQLContext(sc)
df = sqlctx.load(
    url = "jdbc:postgresql://['172.27.27.60']/[metastore]",
    dbtable = "(SELECT * FROM sda_hdd_db.sda_hdd_all LIMIT 10) as blah",
    password = "Hive#Pa55",
    user =  "smicro",
    source = "jdbc",
    driver = "org.postgresql.Driver"
)


# In[14]:




from pyspark.sql import SQLContext
abc = SQLContext(sc)
jdbcurl = "jdbc:mysql://nn01.itversity.com:3306/retail-db?username=smicro&password=Hive#Pa55"
df=abc.load(source=“jdbc”, url=jdbcurl, dbtable=“metastore”)


# In[6]:


import pyspark

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.sql("select 'spark' as hello ")

df = spark.sql("SELECT * FROM sda_hdd_db.sda_hdd_all LIMIT 10")

df.show()


# In[ ]:


### HBASE 


# In[7]:


table = connection.table('employees')
table1 = connection.table('failure1rec')
table2 = connection.table('client_data')


# In[11]:


table


# In[12]:


table1


# In[13]:


table2


# In[7]:


#table.put('2',{'f1','hey'})
rows = table2.rows([b'10A0A08SF97G',b'7QT024SR'])


# In[13]:


connection = hive.Connection(host='172.27.27.60', port=10000, password='Hive#Pa55', username='smicro', auth='CUSTOM')
#dataset = pd.read_sql("SELECT * FROM sda_hdd_db.ml_smart_data_view limit 10", conn)
connection.open()
table = connection.table('employees')
rows = table.rows(['1'])
for key, data in rows:
    print(key, data)
    


# In[14]:


print(table)


# In[34]:


print(table.row('1'))


# In[38]:


rows = table2.rows(['1'])
for key, data in rows:
    print(key, data)


# In[35]:


############## https://www.cdata.com/kb/tech/hbase-python-petl.rst
#!pip install connect
import petl as etl
import pandas as pd
import cdata as mod
import connect as connect


# In[37]:


import cdata.apachehbase as mod


# In[84]:


cnxn = mod.connect("Server=172.27.27.60;Port=60000;")
#cnxn = happybase.Connection(host='172.27.27.60', port=60000, autoconnect=True)

#host='172.27.27.60', port=60000, autoconnect=True)


# In[36]:


from operator import mod


# In[80]:


sql = "SELECT 'name','age' FROM employees"


# In[81]:


sql


# In[82]:


table1 = etl.fromdb(cnxn,sql)


# In[83]:


table1


# In[11]:


get_ipython().system('pip3 install starbase')


# In[57]:


from starbase import Connection


# In[58]:


cnxn = Connection("Server=172.27.27.60;Port=60000;")


# In[60]:


sql = "SELECT * FROM client_data"


# In[61]:


sql = "SELECT model, failure FROM client_data WHERE model = 'MK0271YGJP8LVA'"


# In[62]:


sql


# In[64]:


table1 = etl.fromdb(cnxn,sql)
table1


# In[11]:


get_ipython().system('pip install logger')


# In[1]:


import happybase as hp


# In[6]:


import logging


# In[7]:


from logging.handlers import RotatingFileHandler


# In[12]:


from logging.handlers import logging


# In[14]:


import csv


# In[15]:


def create_hbase_connection():
    try:
        conn = hp.connection()
        conn.open()
        return conn
    except Exception as e:
        logger.info[e]


# In[5]:


def scan_table():
    try:
        connection = create_hbase_connection()
        table=connection.table('client_data')
        for key.data in table.scan():
        no=key.decode('smart_1_normalized')
        for value1, value2 in data.items():
            cf1=value1.decode['smart_1_normalized']
            cf2=value2.decode['smart_1_raw']
            print (no,cf1,cf2)
    except Exception as e:
        logger.info[e]


# In[2]:




connection = happybase.Connection('172.27.27.60')
connection = happybase.Connection('172.27.27.60, autoconnect=False)

# before first use:
connection.open()
print connection.tables()


# In[8]:


import happybase
Connection = happybase.Connection('172.27.27.60', table_prefix='client_data')


# In[14]:


print Connection.table()


# In[21]:


import happybase

connection = happybase.Connection('172.27.27.60')
table = connection.table('client_data')


# In[56]:


table = connection.table('client_data')

#table.put(b'row-key', {b'family:qual1': b'value1', b'family:qual2': b'value2'})
#row = table.row(b'row-key')
#print(row[b'family:qual1']) 


row = table.row(b'HBASE_ROW_KEY')


# In[34]:


row = table.row(b'row-key')


# In[27]:


print(row[b'smart_1_raw:qual1'])


# In[26]:


for key, data in table.rows([b'row-key-1', b'row-key-2']):
    print(key, data)


# In[28]:


for key, data in table.scan(row_prefix=b'row'):
    print(key, data)  # prints 'value1' and 'value2'


# In[38]:


jdbc:apachehbase:Server='172.27.27.60';Port=60000;


# In[ ]:


table = connection.table('table-name')

table.put(b'row-key', {b'family:qual1': b'value1',
                       b'family:qual2': b'value2'})


# In[1]:


import happybase
connection = happybase.Connection(host='172.27.27.60', port=9090, autoconnect=True)
connection.open()
print(connection.tables())

#tables = connection.table('failure1rec')
#rows = table.rows(['1','2'])
#for key, data in rows:
#   print(key, data)
#rows = table.scan('client_data')
#rows = table.scan(connection.table('client_data'))
#rows = table.scan(filter="HBASE_ROW_KEY", row_start="MK0271YGJP8LVA", row_stop="Z1Y45RA5")
#rows = table.scan(filter="serial_number", row_start="MK0271YGJP8LVA", row_stop="Z1Y45RA5")
#rows = table.scan(filter="serial_number", row_start="1", row_stop="10000")
#c = happybase.Connection('172.27.27.60',9090 )
#print(tables)
#one=print(rows)
#one


# In[6]:


#import happybase

#c = happybase.Connection('172.27.27.60',60000, autoconnect=False)
#c.open()
#print(c.tables())

import happybase

c = happybase.Connection('172.27.27.60',9090 )
print(c.tables())


# In[2]:


print(connection.tables())


# In[3]:


import happybase
server = "server-address"
connection = happybase.Connection(server)
print connection.tables()


# In[94]:


one.to_csv(r'E:\work\Supermicro\modeling\data_Q2_2019\one.csv')


# In[4]:


import happybase
connection = happybase.Connection(host='172.27.27.60', port=9090, autoconnect=True)
connection.open()
print(connection.tables())


# In[5]:


tables = connection.table(b'client_data')


# In[6]:


print(tables)


# In[65]:


rows = table1.scan(filter="HBASE_ROW_KEY", row_start="MK0271YGJP8LVA", row_stop="Z1Y45RA5")
#[print(n) for n in rows]
print(rows)


# In[ ]:





# In[53]:


import happybase

connection = happybase.Connection('172.27.27.60')
table = connection.table('sample1')

table.put(b'row-key', {b'family:qual1': b'value1',
                       b'family:qual2': b'value2'})


# In[4]:


import happybase
c = happybase.Connection('127.0.0.1',9090, autoconnect=False)
c.open()
print(c.tables())


# In[35]:


table = connection.table('client_data')


# In[36]:


rows = table.rows(['row-key-1', 'row-key-2'])
for key, data in rows:
print key, data


# In[59]:


import happybase
import numpy as np
import pandas as pd
import pdhbase as pdh
connection = None
try:
    connection = happybase.Connection('127.0.0.1')
    connection.open()
    df = pdh.read_hbase(connection, 'sample_table', 'df_key', cf='cf')
    print df
finally:
    if connection:
        connection.close()


# In[61]:


import happybase
import numpy as np
import pandas as pd
import pdhbase as pdh
connection = None
try:
    connection = happybase.Connection('127.0.0.1')
    connection.open()
    df = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    df['f'] = 'hello world'
    pdh.to_hbase(df, connection, 'sample_table', 'df_key', cf='cf')
    finally:
        if connection:
            connection.close()


# In[42]:


pool = happybase.ConnectionPool(size=3, host='172.27.27.60')


# In[43]:


with pool.connection() as connection:
    print(connection.tables())


# In[47]:


with pool.connection() as connection:
    table = connection.table('client_data')
    row = table.row(b'serial_number')


# In[48]:


process_data(row)


# In[11]:


import happybase
cn = happybase.Connection('172.27.27.60')
v = cn.table('client_data')
v


# In[23]:


#n = v.scan(row_prefix='0001')
n = v.scan(row_prefix='1')


# In[24]:


n


# In[25]:


g_to_list = list(n) # where g is a generator


# In[26]:


for key,data in n:
    print key,data


# In[8]:


v.despine()


# In[ ]:




