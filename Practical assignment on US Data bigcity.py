#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# # Dataset

# Population of U.S. Cities

# # Description

# The bigcity data frame has 49 rows and 2 columns.
# The measurements are the population (in 1000's) of 49 U.S. cities in 1920 and 1930. The 49 cities are a random sample taken
# from the 196 largest cities in 1920.

# # Format

# This data frame contains the following columns:
# 
# u The 1920 population.
# 
# x The 1930 population.
# 
# Source:
# 
# The data were obtained from
# 
# Cochran, W.G. (1977) Sampling Techniques. Third edition. John Wiley
# 
# References:
# 
# Davison, A.C. and Hinkley, D.V. (1997) Bootstrap Methods and Their Application. Cambridge University Press

# # 1. Read the dataset given in file named 'bigcity.csv'.

# In[8]:


import pandas as pd
data=pd.read_csv("C:/Users/Shivji Tiwari/Desktop/Applied Data Science IBM/Practicle Assignment1/bigcity.csv")
data.head()


# # 2. Explore the shape of dataset (0.5 points)
# Find the number of rows in given dataset and separate the input and target variables into X and Y. Hint: You can shape function 
# to get the size of the dataframe

# In[10]:


rows=data.shape[0]
print("no of rows:%d"%(rows))

X=data.u
Y=data.x
X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)


# # 3. Check the dataset for any missing values and also print out the correlation matrix (0.5 points)
# You can use .isna() and .corr() functions to check NA's and correlation in the dataframe respectively

# In[12]:


data.isna().sum()


# In[14]:


data.corr()


# ###### The high correlation betwwen u and x indicates that the variable u is a good predictor of variable x

# # 4. Split data into train, test sets (0.5 points)
# Divide the data into training and test sets with 80-20 split using scikit-learn. Print the shapes of training and test feature 
# sets.*
# Check: train_test_split function

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=1)


# # 5. Find coefficients & intercept (0.5 points)
# Estimate the coefficients b0 and b1 using scikit-learn.
# Check: coef_ and intercept_ functions can help you get coefficients & intercept

# In[16]:


from sklearn.linear_model import LinearRegression
# invoke the LinearRegression function and find the bestfit model on training data
regression_model=LinearRegression()
regression_model.fit(X_train, Y_train)
# Let us explore the coefficients for each of the independent attributes
b1=regression_model.coef_
b0=regression_model.intercept_
print("b1 is:{} and b0 is:{}".format(b1,b0))


# # 6.  Linear Relationship between feature and target (0.5 points)
# Plot the line with b1 and b0 as slope and y-intercept.

# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X_train ,b1*X_train+b0)


# # 7. Evaluation of model with scikit-learn (0.5 points)
# Validate the model with Root Mean Squares error and R^2 score using scikit-learn. RMSE and R2 for test data and prediction
# 
# Hint: You can import mean_squared_error function & r2 (R square) from sklearn.metrics. Performing root operation over mean 
# square error over mean square error gives you root mean square error

# In[25]:


y_pred=regression_model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
mse=mean_squared_error(Y_test, y_pred)
rms=sqrt(mse)
r2_score=r2_score(Y_test,y_pred)
print("the root mean square error is:{} and R2 error is {}".format(rms,r2_score))


# # 8. Calculate the accuracy of the model for both training and test data set (0.5 points)
# 
# Hint: .score() function

# In[26]:


regression_model.score(X_train,Y_train)


# In[27]:


regression_model.score(X_test,Y_test)


# In[ ]:




