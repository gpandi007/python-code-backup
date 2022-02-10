#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Modeling approach 1 run the model in live database and run with test data - take 3 min time to load all the data
# Modeling approach 2 save the model and run with test data - it does not take time 

# save the model to disk
import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(rfc, open(filename, 'wb'))
# some time later...######### preprocess test data alone to test the prediction results
# load the model from disk

# read testdata #########################################################################
# read testdata, #Missing value Imputation, # Remove CORRILATION VARIYABLE 
# drop date # remove top 10 missing variables # Remove below variable as it has only one value # normalize the data 
# drop after normalization # Label Encoding model
# data conversion  # define the y variable #QuantileTransformer 
# remove max VIF variables #features selection #misc_feat - drop the remaining features 
# drop the sting data #Splitting the values for X_train and Y_train 
# get only serial number form test data # pass x test to get the prediction


# In[2]:


import pickle
loaded_model = pickle.load(open(filename, 'rb'))
# accuraccy score with test data
accuraccy_result = loaded_model.score(X_test, Y_test)
print(result)
# final prediction results 
prediction_result=loaded_model.predict(X_test)
prediction_result
#Model_Prediction dataframe
Model_Prediction = pd.DataFrame(prediction_result, columns =['Model_Prediction'])
print("\nPandas DataFrame: ")
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


# In[3]:


loaded_model


# In[ ]:




