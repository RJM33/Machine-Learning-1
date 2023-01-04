#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np


# In[48]:


#Import data set
data=pd.read_csv(r"C:\Users\PRIYANKA\OneDrive\Desktop\Priyanka\Datasets\Heart_Dis.csv")
data


# In[49]:


#correlation coefficient : For selecting the features 
data.corr()


# In[50]:


#check Null Values 
data.isnull().sum()


# In[51]:


#Selecting depepndent and independent variable 


# In[52]:


x = data.iloc[:,:-1]
y=data.iloc[:,-1]
print(y)


# In[53]:


#checking the count of Categorical dependent varibale
data["target"].value_counts()


# In[54]:


# For convert Minority (0) to majority (1) , for that we use here Over sampling technique and it balance the count 


# In[55]:


from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler()
x_data,y_data = sampler.fit_resample(x,y)


# In[56]:


print(x_data)


# In[57]:


#checking Count 
from collections import Counter
print(Counter(y_data))


# In[58]:


#Standardiztion for imporve accuracy 
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_scaled = scaler.fit_transform(x_data)


# In[59]:


x_scaled


# In[60]:


#Splitting data into train and test data ,where test data is 25% and train is 75% .


# In[61]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x_scaled,y_data,test_size = 0.25,random_state = 50)


# In[62]:


print(Counter(y_test))


# In[63]:


x_train


# In[64]:


#Fitting the KNN Model 


# In[65]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=1)
knn.fit(x_train,y_train)


# In[66]:


y_pred = knn.predict(x_test)


# In[67]:


y_pred  # Predicted values 


# In[68]:


##predicting a new observation in knn


# In[69]:


print(knn.predict(scaler.transform([[72,0,1,135,360,1,0,289,1,3.1,2,3,3]]))) 


# In[70]:


##Accuracy Checking 
from sklearn.metrics import accuracy_score
a=accuracy_score(y_test,y_pred)
print("Accuracy of  the model is :",a.round(4)*100,"%")


# In[71]:


##accuracy if done normalization instead of standardization is 0.878 i.e less 


# In[72]:


## Support vector machine model 


# In[73]:


from sklearn.svm import SVC
svm = SVC(kernel= "linear",random_state = 0)
svm.fit(x_train,y_train)


# In[74]:


y_pred1 = svm.predict(x_test)
print("predicted value :",y_pred1)


# In[75]:


acc= accuracy_score(y_test,y_pred1)
print("Accuracy of  the model is :",acc.round(4)*100,"%")


# In[76]:


##predicting a new observation in svm


# In[77]:


print(svm.predict(scaler.transform([[72,0,1,135,360,1,0,289,1,3.1,2,3,3]])))


# In[78]:


##both svm and knn predicted different classes for the same observation


# In[79]:


#Fitting Logistic Regression 


# In[80]:


from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression ()
logistic.fit(x_train,y_train)


# In[81]:


y_pred2=logistic.predict(x_test)
y_pred


# In[82]:


acc2=accuracy_score(y_test,y_pred2)
print("Accuracy of  the model is :",acc2.round(4)*100,"%")


# In[83]:


#Predicting new observation 


# In[84]:


print(logistic.predict(scaler.transform([[72,0,1,135,360,1,0,289,1,3.1,2,3,3]])))

