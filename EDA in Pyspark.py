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
sc =SparkContext.getOrCreate()
sqlContext = SQLContext(sc)


# In[ ]:


#STEP-2 Read data from HIVE
from pyspark.sql import HiveContext
from pyspark import SparkConf
spark = SparkSession(sc)
hive_context = HiveContext(sc)
sqlContext = SQLContext(sc)
#bank1 = sqlContext.table("sda_hdd_db.sda_hdd_all")
dataset = sqlContext.sql("SELECT * FROM sda_hdd_db.ml_smart_data_view")


# In[4]:


# convert catogorical colum as dable 
for col in dataset.columns:
    dataset = dataset.withColumn(col,dataset[col].cast('double'))


# In[11]:


#STEP-4 select columns based on 1. less missing value 2. VIF 4. Correlation
# 5. Outliers, skewness, kurtosis 6.EDA 7.Descriptive analysis ect... 
# columns_to_drop = ["smart_201_normalized","smart_17_raw","smart_245_raw","smart_201_raw","smart_218_raw","smart_16_raw","smart_170_raw","smart_180_normalized","smart_180_raw","model","serial_number","process_date","date","smart_226_raw","smart_8_normalized","smart_254_normalized","smart_12_raw","smart_192_raw","smart_196_normalized","smart_222_normalized","smart_175_raw","smart_190_normalized","smart_13_raw","smart_177_normalized","smart_190_raw","smart_202_raw","smart_255_raw", "smart_15_normalized", "smart_234_raw", "smart_255_normalized", "smart_15_raw", "smart_234_normalized", "smart_206_normalized", "smart_206_raw", "smart_248_raw", "smart_248_normalized","smart_210_raw", "smart_224_raw", "smart_18_raw", "smart_23_raw", "smart_24_raw", "smart_179_raw", "smart_181_raw", "smart_182_raw", "smart_251_normalized", "smart_250_normalized", "smart_254_raw"]

input_cols = dataset.select("capacity_bytes",
"failure",
"smart_1_normalized",
"smart_1_raw",
"smart_2_normalized",
"smart_2_raw",
"smart_3_normalized",
"smart_3_raw",
"smart_4_normalized",
"smart_4_raw",
"smart_5_normalized",
"smart_5_raw",
"smart_7_normalized",
"smart_7_raw",
"smart_8_raw",
"smart_9_normalized",
"smart_9_raw",
"smart_10_normalized",
"smart_10_raw",
"smart_11_normalized",
"smart_11_raw",
"smart_12_normalized",
"smart_13_normalized",
"smart_16_normalized",
"smart_17_normalized",
"smart_18_normalized",
"smart_22_normalized",
"smart_22_raw",
"smart_23_normalized",
"smart_24_normalized",
"smart_168_normalized",
"smart_168_raw",
"smart_170_normalized",
"smart_173_normalized",
"smart_173_raw",
"smart_174_normalized",
"smart_174_raw",
"smart_175_normalized",
"smart_177_raw",
"smart_179_normalized",
"smart_181_normalized",
"smart_182_normalized",
"smart_183_normalized",
"smart_183_raw",
"smart_184_normalized",
"smart_184_raw",
"smart_187_normalized",
"smart_187_raw",
"smart_188_normalized",
"smart_188_raw",
"smart_189_normalized",
"smart_189_raw",
"smart_191_normalized",
"smart_191_raw",
"smart_192_normalized",
"smart_193_normalized",
"smart_193_raw",
"smart_194_normalized",
"smart_194_raw",
"smart_195_normalized",
"smart_195_raw",
"smart_196_raw",
"smart_197_normalized",
"smart_197_raw",
"smart_198_normalized",
"smart_198_raw",
"smart_199_normalized",
"smart_199_raw",
"smart_200_normalized",
"smart_200_raw",
"smart_202_normalized",
"smart_210_normalized",
"smart_218_normalized",
"smart_220_normalized",
"smart_220_raw",
"smart_222_raw",
"smart_223_normalized",
"smart_223_raw",
"smart_224_normalized",
"smart_225_normalized",
"smart_225_raw",
"smart_226_normalized",
"smart_231_normalized",
"smart_231_raw",
"smart_232_normalized",
"smart_232_raw",
"smart_233_normalized",
"smart_233_raw",
"smart_235_normalized",
"smart_235_raw",
"smart_240_normalized",
"smart_240_raw",
"smart_241_normalized",
"smart_241_raw",
"smart_242_normalized",
"smart_242_raw",
"smart_245_normalized",
"smart_247_normalized",
"smart_247_raw",
"smart_250_raw",
"smart_251_raw",
"smart_252_normalized",
"smart_252_raw",
"smart_160_normalized",
"smart_160_raw",
"smart_161_normalized",
"smart_161_raw",
"smart_163_normalized",
"smart_163_raw",
"smart_164_normalized",
"smart_164_raw",
"smart_165_normalized",
"smart_165_raw",
"smart_166_normalized",
"smart_166_raw",
"smart_167_normalized",
"smart_167_raw",
"smart_169_normalized",
"smart_169_raw",
"smart_176_normalized",
"smart_176_raw",
"smart_178_normalized",
"smart_178_raw")
cols = df.columns
#df.printSchema()


# In[12]:


# impute the null value with mean
imputed_col = ['f_{}'.format(i+1) for i in range(len(input_cols))]
model = Imputer(strategy='mean',missingValue=None,inputCols=input_cols,outputCols=imputed_col).fit(dataset)
impute_data = model.transform(dataset)


# In[4]:


# count missing 
dataset=df
import pyspark.sql.functions as F
def count_missings(spark_df,sort=True):
    """
    Counts number of nulls and nans in each column
    """
    dataset = spark_df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'string', 'date')]).toPandas()

    if len(dataset) == 0:
        print("There are no any missing values!")
        return None

    if sort:
        return dataset.rename(index={0: 'count'}).T.sort_values("count",ascending=False)

    return dataset
missings=count_missings(dataset)
# download missing 
#missings
import numpy as np
import pandas as pd
from openpyxl import Workbook
writer = pd.ExcelWriter("missings_info_pyspark.xlsx")
missings.to_excel(excel_writer=writer, sheet_name='Sheet1', na_rep="")
writer.save()


# In[12]:


#STEP-3 fill missing with zero 
df=df.na.fill(value=0)


# In[13]:


get_ipython().system('pip install hist')


# In[15]:


# histogram 
import pandas as pd
import pyspark.sql as sparksql
from pyspark_dist_explore import hist
import matplotlib.pyplot as plt
#df.hist(column = 'smart_12_normalized')

from pyspark_dist_explore import hist
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
hist(ax, df.select('smart_12_normalized'), bins = 20, color=['red'])


# In[6]:


dataset.smart_1_normalized.plot.density(color='green')
plt.title('smart_12_normalized')


# In[16]:


#STEP-4 select columns based on 1. less missing value 2. VIF 4. Correlation
# 5. Outliers, skewness, kurtosis 6.EDA 7.Descriptive analysis ect... 
# columns_to_drop = ["smart_201_normalized","smart_17_raw","smart_245_raw","smart_201_raw","smart_218_raw","smart_16_raw","smart_170_raw","smart_180_normalized","smart_180_raw","model","serial_number","process_date","date","smart_226_raw","smart_8_normalized","smart_254_normalized","smart_12_raw","smart_192_raw","smart_196_normalized","smart_222_normalized","smart_175_raw","smart_190_normalized","smart_13_raw","smart_177_normalized","smart_190_raw","smart_202_raw","smart_255_raw", "smart_15_normalized", "smart_234_raw", "smart_255_normalized", "smart_15_raw", "smart_234_normalized", "smart_206_normalized", "smart_206_raw", "smart_248_raw", "smart_248_normalized","smart_210_raw", "smart_224_raw", "smart_18_raw", "smart_23_raw", "smart_24_raw", "smart_179_raw", "smart_181_raw", "smart_182_raw", "smart_251_normalized", "smart_250_normalized", "smart_254_raw"]

df = df.select("capacity_bytes",
"failure",
"smart_1_normalized",
"smart_1_raw",
"smart_2_normalized",
"smart_2_raw",
"smart_3_normalized",
"smart_3_raw",
"smart_4_normalized",
"smart_4_raw",
"smart_5_normalized",
"smart_5_raw",
"smart_7_normalized",
"smart_7_raw",
"smart_8_raw",
"smart_9_normalized",
"smart_9_raw",
"smart_10_normalized",
"smart_10_raw",
"smart_11_normalized",
"smart_11_raw",
"smart_12_normalized",
"smart_13_normalized",
"smart_16_normalized",
"smart_17_normalized",
"smart_18_normalized",
"smart_22_normalized",
"smart_22_raw",
"smart_23_normalized",
"smart_24_normalized",
"smart_168_normalized",
"smart_168_raw",
"smart_170_normalized",
"smart_173_normalized",
"smart_173_raw",
"smart_174_normalized",
"smart_174_raw",
"smart_175_normalized",
"smart_177_raw",
"smart_179_normalized",
"smart_181_normalized",
"smart_182_normalized",
"smart_183_normalized",
"smart_183_raw",
"smart_184_normalized",
"smart_184_raw",
"smart_187_normalized",
"smart_187_raw",
"smart_188_normalized",
"smart_188_raw",
"smart_189_normalized",
"smart_189_raw",
"smart_191_normalized",
"smart_191_raw",
"smart_192_normalized",
"smart_193_normalized",
"smart_193_raw",
"smart_194_normalized",
"smart_194_raw",
"smart_195_normalized",
"smart_195_raw",
"smart_196_raw",
"smart_197_normalized",
"smart_197_raw",
"smart_198_normalized",
"smart_198_raw",
"smart_199_normalized",
"smart_199_raw",
"smart_200_normalized",
"smart_200_raw",
"smart_202_normalized",
"smart_210_normalized",
"smart_218_normalized",
"smart_220_normalized",
"smart_220_raw",
"smart_222_raw",
"smart_223_normalized",
"smart_223_raw",
"smart_224_normalized",
"smart_225_normalized",
"smart_225_raw",
"smart_226_normalized",
"smart_231_normalized",
"smart_231_raw",
"smart_232_normalized",
"smart_232_raw",
"smart_233_normalized",
"smart_233_raw",
"smart_235_normalized",
"smart_235_raw",
"smart_240_normalized",
"smart_240_raw",
"smart_241_normalized",
"smart_241_raw",
"smart_242_normalized",
"smart_242_raw",
"smart_245_normalized",
"smart_247_normalized",
"smart_247_raw",
"smart_250_raw",
"smart_251_raw",
"smart_252_normalized",
"smart_252_raw",
"smart_160_normalized",
"smart_160_raw",
"smart_161_normalized",
"smart_161_raw",
"smart_163_normalized",
"smart_163_raw",
"smart_164_normalized",
"smart_164_raw",
"smart_165_normalized",
"smart_165_raw",
"smart_166_normalized",
"smart_166_raw",
"smart_167_normalized",
"smart_167_raw",
"smart_169_normalized",
"smart_169_raw",
"smart_176_normalized",
"smart_176_raw",
"smart_178_normalized",
"smart_178_raw")
cols = df.columns
#df.printSchema()


# In[17]:


# scatter_matrix
pdf = df.toPandas()
from pandas.tools.plotting import scatter_matrix
stuff = scatter_matrix(pdf, alpha=0.7, figsize=(6, 6), diagonal='kde', color=pdf.col)


# In[19]:


# OneHotEncoder  OneHotEncoderEstimator, 
from pyspark.ml.feature import OneHotEncoder,StringIndexer, VectorAssembler
# OneHotEncoder  OneHotEncoderEstimator, 
categoricalColumns = ['model']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
#label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
#stages += [label_stringIdx]
numericCols = ["capacity_bytes",
"failure",
"smart_1_normalized",
"smart_1_raw",
"smart_2_normalized",
"smart_2_raw",
"smart_3_normalized",
"smart_3_raw",
"smart_4_normalized",
"smart_4_raw",
"smart_5_normalized",
"smart_5_raw",
"smart_7_normalized",
"smart_7_raw",
"smart_8_raw",
"smart_9_normalized",
"smart_9_raw",
"smart_10_normalized",
"smart_10_raw",
"smart_11_normalized",
"smart_11_raw",
"smart_12_normalized",
"smart_13_normalized",
"smart_16_normalized",
"smart_17_normalized",
"smart_18_normalized",
"smart_22_normalized",
"smart_22_raw",
"smart_23_normalized",
"smart_24_normalized",
"smart_168_normalized",
"smart_168_raw",
"smart_170_normalized",
"smart_173_normalized",
"smart_173_raw",
"smart_174_normalized",
"smart_174_raw",
"smart_175_normalized",
"smart_177_raw",
"smart_179_normalized",
"smart_181_normalized",
"smart_182_normalized",
"smart_183_normalized",
"smart_183_raw",
"smart_184_normalized",
"smart_184_raw",
"smart_187_normalized",
"smart_187_raw",
"smart_188_normalized",
"smart_188_raw",
"smart_189_normalized",
"smart_189_raw",
"smart_191_normalized",
"smart_191_raw",
"smart_192_normalized",
"smart_193_normalized",
"smart_193_raw",
"smart_194_normalized",
"smart_194_raw",
"smart_195_normalized",
"smart_195_raw",
"smart_196_raw",
"smart_197_normalized",
"smart_197_raw",
"smart_198_normalized",
"smart_198_raw",
"smart_199_normalized",
"smart_199_raw",
"smart_200_normalized",
"smart_200_raw",
"smart_202_normalized",
"smart_210_normalized",
"smart_218_normalized",
"smart_220_normalized",
"smart_220_raw",
"smart_222_raw",
"smart_223_normalized",
"smart_223_raw",
"smart_224_normalized",
"smart_225_normalized",
"smart_225_raw",
"smart_226_normalized",
"smart_231_normalized",
"smart_231_raw",
"smart_232_normalized",
"smart_232_raw",
"smart_233_normalized",
"smart_233_raw",
"smart_235_normalized",
"smart_235_raw",
"smart_240_normalized",
"smart_240_raw",
"smart_241_normalized",
"smart_241_raw",
"smart_242_normalized",
"smart_242_raw",
"smart_245_normalized",
"smart_247_normalized",
"smart_247_raw",
"smart_250_raw",
"smart_251_raw",
"smart_252_normalized",
"smart_252_raw",
"smart_160_normalized",
"smart_160_raw",
"smart_161_normalized",
"smart_161_raw",
"smart_163_normalized",
"smart_163_raw",
"smart_164_normalized",
"smart_164_raw",
"smart_165_normalized",
"smart_165_raw",
"smart_166_normalized",
"smart_166_raw",
"smart_167_normalized",
"smart_167_raw",
"smart_169_normalized",
"smart_169_raw",
"smart_176_normalized",
"smart_176_raw",
"smart_178_normalized",
"smart_178_raw"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# In[23]:


# Correlation matrix
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)


# In[40]:


matrix = Correlation.corr(df_vector, vector_col)
cor_np = matrix.collect()[0][matrix.columns[0]].toArray()


# In[38]:


def correlation_matrix(df, corr_columns, method='pearson'):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)
    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns)


# In[43]:


#describe
describe=df.describe()
#missings
import numpy as np
import pandas as pd
from openpyxl import Workbook
writer = pd.ExcelWriter("describe_pyspark.xlsx")
describe.to_excel(excel_writer=writer, sheet_name='Sheet1', na_rep="")
writer.save()


# In[20]:


# scalling
#print("Before Scaling :")
#dataset.show(5)

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# UDF for converting column type from vector to double type
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

# Iterating over columns to be scaled
for i in ["capacity_bytes",
"smart_1_normalized",
"smart_1_raw",
"smart_2_normalized",
"smart_2_raw",
"smart_3_normalized",
"smart_3_raw",
"smart_4_normalized",
"smart_4_raw",
"smart_5_normalized",
"smart_5_raw",
"smart_7_normalized",
"smart_7_raw",
"smart_8_raw",
"smart_9_normalized",
"smart_9_raw",
"smart_10_normalized",
"smart_10_raw",
"smart_11_normalized",
"smart_11_raw",
"smart_12_normalized",
"smart_13_normalized",
"smart_16_normalized",
"smart_17_normalized",
"smart_18_normalized",
"smart_22_normalized",
"smart_22_raw",
"smart_23_normalized",
"smart_24_normalized",
"smart_168_normalized",
"smart_168_raw",
"smart_170_normalized",
"smart_173_normalized",
"smart_173_raw",
"smart_174_normalized",
"smart_174_raw",
"smart_175_normalized",
"smart_177_raw",
"smart_179_normalized",
"smart_181_normalized",
"smart_182_normalized",
"smart_183_normalized",
"smart_183_raw",
"smart_184_normalized",
"smart_184_raw",
"smart_187_normalized",
"smart_187_raw",
"smart_188_normalized",
"smart_188_raw",
"smart_189_normalized",
"smart_189_raw",
"smart_191_normalized",
"smart_191_raw",
"smart_192_normalized",
"smart_193_normalized",
"smart_193_raw",
"smart_194_normalized",
"smart_194_raw",
"smart_195_normalized",
"smart_195_raw",
"smart_196_raw",
"smart_197_normalized",
"smart_197_raw",
"smart_198_normalized",
"smart_198_raw",
"smart_199_normalized",
"smart_199_raw",
"smart_200_normalized",
"smart_200_raw",
"smart_202_normalized",
"smart_210_normalized",
"smart_218_normalized",
"smart_220_normalized",
"smart_220_raw",
"smart_222_raw",
"smart_223_normalized",
"smart_223_raw",
"smart_224_normalized",
"smart_225_normalized",
"smart_225_raw",
"smart_226_normalized",
"smart_231_normalized",
"smart_231_raw",
"smart_232_normalized",
"smart_232_raw",
"smart_233_normalized",
"smart_233_raw",
"smart_235_normalized",
"smart_235_raw",
"smart_240_normalized",
"smart_240_raw",
"smart_241_normalized",
"smart_241_raw",
"smart_242_normalized",
"smart_242_raw",
"smart_245_normalized",
"smart_247_normalized",
"smart_247_raw",
"smart_250_raw",
"smart_251_raw",
"smart_252_normalized",
"smart_252_raw",
"smart_160_normalized",
"smart_160_raw",
"smart_161_normalized",
"smart_161_raw",
"smart_163_normalized",
"smart_163_raw",
"smart_164_normalized",
"smart_164_raw",
"smart_165_normalized",
"smart_165_raw",
"smart_166_normalized",
"smart_166_raw",
"smart_167_normalized",
"smart_167_raw",
"smart_169_normalized",
"smart_169_raw",
"smart_176_normalized",
"smart_176_raw",
"smart_178_normalized",
"smart_178_raw"]:
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    dataset = pipeline.fit(dataset).transform(dataset).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")

#print("After Scaling :")
#dataset.show(5)


# In[ ]:


# drop after scalling 
columns_to_drop = ["capacity_bytes",
"smart_1_normalized",
"smart_1_raw",
"smart_2_normalized",
"smart_2_raw",
"smart_3_normalized",
"smart_3_raw",
"smart_4_normalized",
"smart_4_raw",
"smart_5_normalized",
"smart_5_raw",
"smart_7_normalized",
"smart_7_raw",
"smart_8_raw",
"smart_9_normalized",
"smart_9_raw",
"smart_10_normalized",
"smart_10_raw",
"smart_11_normalized",
"smart_11_raw",
"smart_12_normalized",
"smart_13_normalized",
"smart_16_normalized",
"smart_17_normalized",
"smart_18_normalized",
"smart_22_normalized",
"smart_22_raw",
"smart_23_normalized",
"smart_24_normalized",
"smart_168_normalized",
"smart_168_raw",
"smart_170_normalized",
"smart_173_normalized",
"smart_173_raw",
"smart_174_normalized",
"smart_174_raw",
"smart_175_normalized",
"smart_177_raw",
"smart_179_normalized",
"smart_181_normalized",
"smart_182_normalized",
"smart_183_normalized",
"smart_183_raw",
"smart_184_normalized",
"smart_184_raw",
"smart_187_normalized",
"smart_187_raw",
"smart_188_normalized",
"smart_188_raw",
"smart_189_normalized",
"smart_189_raw",
"smart_191_normalized",
"smart_191_raw",
"smart_192_normalized",
"smart_193_normalized",
"smart_193_raw",
"smart_194_normalized",
"smart_194_raw",
"smart_195_normalized",
"smart_195_raw",
"smart_196_raw",
"smart_197_normalized",
"smart_197_raw",
"smart_198_normalized",
"smart_198_raw",
"smart_199_normalized",
"smart_199_raw",
"smart_200_normalized",
"smart_200_raw",
"smart_202_normalized",
"smart_210_normalized",
"smart_218_normalized",
"smart_220_normalized",
"smart_220_raw",
"smart_222_raw",
"smart_223_normalized",
"smart_223_raw",
"smart_224_normalized",
"smart_225_normalized",
"smart_225_raw",
"smart_226_normalized",
"smart_231_normalized",
"smart_231_raw",
"smart_232_normalized",
"smart_232_raw",
"smart_233_normalized",
"smart_233_raw",
"smart_235_normalized",
"smart_235_raw",
"smart_240_normalized",
"smart_240_raw",
"smart_241_normalized",
"smart_241_raw",
"smart_242_normalized",
"smart_242_raw",
"smart_245_normalized",
"smart_247_normalized",
"smart_247_raw",
"smart_250_raw",
"smart_251_raw",
"smart_252_normalized",
"smart_252_raw",
"smart_160_normalized",
"smart_160_raw",
"smart_161_normalized",
"smart_161_raw",
"smart_163_normalized",
"smart_163_raw",
"smart_164_normalized",
"smart_164_raw",
"smart_165_normalized",
"smart_165_raw",
"smart_166_normalized",
"smart_166_raw",
"smart_167_normalized",
"smart_167_raw",
"smart_169_normalized",
"smart_169_raw",
"smart_176_normalized",
"smart_176_raw",
"smart_178_normalized",
"smart_178_raw"]
dataset = dataset.drop(*columns_to_drop)


# In[ ]:


#dataset.printSchema()
df.dtypes


# In[ ]:


#1 count missing value# import sql function pyspark
import pyspark.sql.functions as f

# null values in each column
data_agg = dataset.agg(*[f.count(f.when(f.isnull(c), c)).alias(c) for c in dataset.columns])
data_agg.show()


# In[46]:


import numpy as np
import pandas as pd
from openpyxl import Workbook
writer = pd.ExcelWriter("describe_pyspark.xlsx")
describe.to_excel(excel_writer=writer, sheet_name='Sheet1', na_rep="")
writer.save()


# In[ ]:


df.select('model').describe().show()


# In[47]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[48]:


X = df.iloc[:,:-1]
calc_vif(X)


# In[ ]:




