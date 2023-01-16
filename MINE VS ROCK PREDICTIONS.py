#!/usr/bin/env python
# coding: utf-8

# #  MINE VS ROCK PREDICTION

# In[38]:


#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[9]:


#importing the data
sonar = pd.read_csv("copy of sonar data.csv",header=None)   #here we put header = None because we dot have column name


# In[10]:


sonar.head()


# In[11]:


sonar.describe()  #getting the statistical values of each data


# In[13]:


sonar.shape  #getting the number of rows and colums


# In[15]:


sonar[60].value_counts()  #to check the counts of the distinct values


# In[25]:


sonar.groupby(60).mean()


# In[44]:


sns.countplot(x= 60, data = sonar)  #using sns to get the plot


# In[26]:


#STARTING THE MODEL SELECTION

#Seperating the data and labels

X = sonar.drop(columns =60, axis=1)
Y = sonar[60]


# In[28]:


X


# In[29]:


Y


# In[32]:


#Training and Test Data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.3, stratify =Y, random_state =101) 


# In[33]:


print(X,X_train.shape,X_test.shape)  #Checking the size of the X 


# In[34]:


#Model Training --> Logistic Regression

model = LogisticRegression()


# In[36]:


#trainig the model with training data

model.fit  #fitting the model


# In[37]:


model.fit(X_train,Y_train)


# In[50]:


#Checking the accuracy
from sklearn.metrics import classification_report


# In[52]:


#prediction on x train , (but remember we use the X_test to predict, X_train is already aware of the answers )

X_train_predict = model.predict(X_train)


# In[53]:


train_accuracy = accuracy_score(X_train_predict,Y_train)


# In[54]:


train_accuracy


# In[55]:


#Predicting on x test

X_test_predict= model.predict(X_test)
test_accuracy = classification_report(X_test_predict,Y_test)


# In[57]:


print(X_test_predict)
print(test_accuracy)


# In[61]:


from sklearn.metrics import confusion_matrix


# In[ ]:





# In[62]:


test_accuracy = confusion_matrix(X_test_predict,Y_test)


# In[63]:


test_accuracy  #using the confusion matix to determine false negative and false postives 


# In[65]:


#Making a Predictive System

input_data = (0.0090,0.0062,0.0253,0.0489,0.1197,0.1589,0.1392,0.0987,0.0955,0.1895,0.1896,0.2547,0.4073,0.2988,0.2901,0.5326,0.4022,0.1571,0.3024,0.3907,0.3542,0.4438,0.6414,0.4601,0.6009,0.8690,0.8345,0.7669,0.5081,0.4620,0.5380,0.5375,0.3844,0.3601,0.7402,0.7761,0.3858,0.0667,0.3684,0.6114,0.3510,0.2312,0.2195,0.3051,0.1937,0.1570,0.0479,0.0538,0.0146,0.0068,0.0187,0.0059,0.0095,0.0194,0.0080,0.0152,0.0158,0.0053,0.0189,0.0102)

#Changing the input data to numpy array
inputaray = np.asarray(input_data)

inputreshape = inputaray.reshape(1,-1)

prediction = model.predict(inputreshape)


# In[66]:


prediction


# In[67]:


if prediction =='R':
    print( 'this is a rock')
else:
    print ('this is a mine')


# In[68]:


plt.scatter(X_test_predict,Y_test)  #just trying to something out


# In[ ]:




