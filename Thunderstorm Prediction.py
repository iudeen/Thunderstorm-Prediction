#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bankdata = pd.read_csv(r"C:\Users\Aditya\Documents\MASTERS\IRFAN\img_pixels.csv", header= None)  


# In[3]:


bankdata.head()


# In[4]:


df= pd.read_csv(r"C:\Users\Aditya\Documents\MASTERS\IRFAN\data.csv", header= None)
col= ['Temp', 'Prec', 'Dew', 'Air', 'pres','class']
df.columns=col
df.head()


# In[5]:


df.head()


# In[6]:


new_df = pd.concat([df,bankdata], axis = 1)


# In[7]:


new_df


# In[8]:


x=new_df.drop('class',axis=1)
y= new_df['class']


# In[9]:


from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(y)
LabelEncoder()
le.classes_
y=le.transform(y)


# In[10]:


from sklearn.model_selection import train_test_split
# Create training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state=42, stratify = y)


# In[11]:


print(y_test)


# In[13]:


from sklearn.svm import SVC  
clf = SVC()
clf.fit(x_train, y_train)  


# In[14]:


#from sklearn.svm import LinearSVC
#clf = LinearSVC(random_state=0)
#clf.fit(x_train, y_train)


# In[15]:


y_pred = clf.predict(x_test)


# In[16]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[17]:


clf.score(x_test, y_test)


# In[18]:


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train)


# In[19]:


y_pred = svclassifier.predict(x_test)


# In[ ]:





# In[ ]:





# In[ ]:




