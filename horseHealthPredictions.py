#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.preprocessing import StandardScaler


# In[2]:


import warnings
warnings.simplefilter('ignore')


# In[3]:


horseData=pd.read_csv("train.csv")
print(horseData)


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


horseData["outcome"].hist(bins=3)


# In[6]:


horseData["surgery"].replace(["yes","no"],[0,1],inplace=True)
horseData["age"].replace(["adult","young"],[0,1],inplace=True)
horseData["temp_of_extremities"].replace(["cool","cold","normal","None"],[0,1,2,3],inplace=True)
horseData["peripheral_pulse"].replace(["reduced","normal","absent","None"],[0,1,2,3],inplace=True)
horseData["mucous_membrane"].replace(["dark_cyanotic","pale_cyanotic","pale_pink","normal_pink","bright_red","bright_pink","None"],[0,1,2,3,4,5,6],inplace=True)
horseData["capillary_refill_time"].replace(["less_3_sec","more_3_sec"],[0,1],inplace=True)
horseData["pain"].replace(["mild_pain","depressed","severe_pain","extreme_pain","moderate","alert","None"],[0,1,2,3,4,5,6],inplace=True)
horseData["peristalsis"].replace(["hypomotile","absent"],[0,1],inplace=True)
horseData["abdominal_distention"].replace(["slight","moderate","severe","None","none"],[0,1,2,3,3],inplace=True)
horseData["nasogastric_tube"].replace(["slight","None","none"],[0,1,1],inplace=True)
horseData["nasogastric_reflux"].replace(["more_1_liter","less_1_liter","None","none"],[0,1,2,2],inplace=True)
horseData["rectal_exam_feces"].replace(["normal","decreased","absent","increased","None"],[0,1,2,3,4],inplace=True)
horseData["abdomen"].replace(["distend_small","distend_large","other","None"],[0,1,2,3],inplace=True)
horseData["abdomo_appearance"].replace(["clear","serosanguious","cloudy","None"],[0,1,2,3],inplace=True)
horseData["surgical_lesion"].replace(["yes","no"],[0,1],inplace=True)
horseData["cp_data"].replace(["yes","no"],[0,1],inplace=True)
horseData["outcome"].replace(["lived","died","euthanized"],[0,1,2],inplace=True)
horseData.drop("id",axis="columns", inplace=True)
horseData.drop("hospital_number",axis="columns", inplace=True)
print(horseData)


# In[7]:


df=pd.DataFrame(horseData)


# In[8]:


corrMatrix=df.corr()
corrMatrix["outcome"]


# In[9]:


horseData.describe().transpose()


# In[10]:


horseData.groupby("outcome").mean()


# In[11]:


cols=['outcome','age','pulse','respiratory_rate','surgery','rectal_temp','mucous_membrane','abdominal_distention','packed_cell_volume','total_protein','abdomo_appearance','abdomo_protein']


# In[12]:


sns.pairplot(horseData[cols],hue='outcome',palette='RdBu')


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test=train_test_split(horseData[cols],horseData['outcome'],random_state=0)


# In[14]:


print("X_train shape:",X_train.shape)
print("y_train shape:",y_train.shape)


# In[15]:


print("X_test shape:",X_test.shape)
print("y_test shape:",y_test.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)


# In[22]:


knn.fit(X_train,y_train)


# In[23]:


prediction=knn.predict(X_test)
print("Prediction:",prediction)


# In[24]:


print("Test set score:{:.2f}".format(np.mean(prediction==y_test)))


# In[ ]:




