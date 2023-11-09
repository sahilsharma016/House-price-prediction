#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("housing.csv")


# In[3]:


housing


# ### Importing files

# In[5]:


#import the necessary libraries required 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


housing.info()


# In[8]:


#display scatter_matrix also
fig = plt.figure()
scatter_matrix(housing,figsize =(25,25),alpha=0.9,diagonal="kde",marker="o");


# ### checking co-relations

# In[9]:


hcorr = housing.corr()
hcorr


# In[10]:


#heatmap using seaborn
#set the context for plotting 
sns.set(context="paper",font="monospace")
housing_corr_matrix = housing.corr()
#set the matplotlib figure
fig, axe = plt.subplots(figsize=(12,8))
#Generate color palettes 
cmap = sns.diverging_palette(220,10,center = "light", as_cmap=True)
#draw the heatmap
sns.heatmap(housing_corr_matrix,vmax=1,square =True, cmap=cmap,annot=True );


# ### Remove outliers

# In[11]:


def getOutliers(dataframe,column):
    column = "total_rooms" 
    #housing[column].plot.box(figsize=(8,8))
    des = dataframe[column].describe()
    desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}
    Q1 = des[desPairs['25']]
    Q3 = des[desPairs['75']]
    IQR = Q3-Q1
    lowerBound = Q1-1.5*IQR
    upperBound = Q3+1.5*IQR
    print("(IQR = {})Outlier are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))
    #b = df[(df['a'] > 1) & (df['a'] < 5)]
    data = dataframe[(dataframe [column] < lowerBound) | (dataframe [column] > upperBound)]

    print("Outliers out of total = {} are \n {}".format(housing[column].size,len(data[column])))
    #remove the outliers from the dataframe
    outlierRemoved = housing[~housing[column].isin(data[column])]
    return outlierRemoved


# In[12]:


#get the outlier
df_outliersRemoved = getOutliers(housing,"total_rooms")


# ### Impute missing values

# In[13]:


housing.isnull().sum()


# In[14]:


print(housing["total_bedrooms"].describe())


# In[15]:


total_bedroms = housing[housing["total_bedrooms"].notnull()]["total_bedrooms"]#["total_bedrooms"]
total_bedroms.hist(figsize=(12,8),bins=50)


# In[20]:


print(housing.iloc[:, 4:5].head())

# Create a SimpleImputer instance with strategy="median"
imputer = SimpleImputer(strategy="median")

# Fit the imputer on the 'total_bedrooms' column
imputer.fit(housing.iloc[:, 4:5])

# Transform and replace missing values in the 'total_bedrooms' column
housing.iloc[:, 4:5] = imputer.transform(housing.iloc[:, 4:5])

# Check for missing values after imputation
print(housing.isnull().sum())


# ### Handling categorical values

# In[21]:


labelEncoder = LabelEncoder()
print(housing["ocean_proximity"].value_counts())
housing["ocean_proximity"] = labelEncoder.fit_transform(housing["ocean_proximity"])
housing["ocean_proximity"].value_counts()
housing.describe()


# ### Split data into train and test

# In[23]:


housing_ind = housing.drop("median_house_value",axis=1)
print(housing_ind.head())
housing_dep = housing["median_house_value"]
print("Medain Housing Values")
housing_dep.head()


# In[24]:


#check for rand_state
X_train,X_test,y_train,y_test = train_test_split(housing_ind,housing_dep,test_size=0.2,random_state=42)
#print(X_train.head())
#print(X_test.head())
#print(y_train.head())
#print(y_test.head())
print("X_train shape {} and size {}".format(X_train.shape,X_train.size))
print("X_test shape {} and size {}".format(X_test.shape,X_test.size))
print("y_train shape {} and size {}".format(y_train.shape,y_train.size))
print("y_test shape {} and size {}".format(y_test.shape,y_test.size))


# In[25]:


X_train.head()


# ## Standardize the data

# In[26]:


independent_scaler = StandardScaler()
X_train = independent_scaler.fit_transform(X_train)
X_test = independent_scaler.transform(X_test)
print(X_train[0:5,:])
print("test data")
print(X_test[0:5,:])


# # Linear regression

# In[27]:


#initantiate the linear regression
linearRegModel = LinearRegression(n_jobs=-1)
#fit the model to the training data (learn the coefficients)
linearRegModel.fit(X_train,y_train)
#print the intercept and coefficients 
print("Intercept is "+str(linearRegModel.intercept_))
print("coefficients  is "+str(linearRegModel.coef_))


# ##  Predictions

# In[28]:


#predict on the test data
y_pred = linearRegModel.predict(X_test)


# In[29]:


print(len(y_pred))
print(len(y_test))
print(y_pred[0:5])
print(y_test[0:5])


# In[30]:


test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',);


# ## Root mean sq error

# In[31]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print(np.sqrt(metrics.mean_squared_error(y_train,linearRegModel.predict(X_train))))


# # now with decision tree

# In[32]:


dtReg = DecisionTreeRegressor(max_depth=9)
dtReg.fit(X_train,y_train)


# In[33]:


dtReg_y_pred = dtReg.predict(X_test)
dtReg_y_pred


# In[34]:


print(np.sqrt(metrics.mean_squared_error(y_test,dtReg_y_pred)))


# In[35]:


test = pd.DataFrame({'Predicted':dtReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# In[ ]:




