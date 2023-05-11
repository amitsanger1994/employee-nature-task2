#!/usr/bin/env python
# coding: utf-8

# In[9]:


#  Ideaspice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats


# In[10]:


#importing the Dataset
data = pd.read_csv("task2.csv")


# In[11]:


data.head()


# In[15]:


#convert the target value into numerical
data['Turnover']=data['Turnover'].str.replace('Yes','0')
data['Turnover']=data['Turnover'].str.replace('No','1')
data['Turnover']=pd.to_numeric(data['Turnover'])


# In[17]:


data.head()


# In[25]:


#SPLIT THE FEATURES INTO NUMERIC DATASETS

numeric = data.select_dtypes("number")


# In[26]:


numeric


# In[27]:


numericdata= numeric
numericdata


# In[31]:


numericdata['Turnover'].mean()


# In[30]:


numericdata.describe()


# In[33]:


numericdata.columns


# In[37]:


numericdata.duplicated().sum()


# In[43]:


numericdata.isnull().sum()


# In[46]:


numericdata


# In[49]:


numericdata=numericdata.drop(['Hours','StockOptionLevel','YearsAtCompany'], axis=1)
numericdata()


# In[48]:


numericdata=numericdata.drop(['EmployeId'], axis=1)


# In[50]:


numericdata


# In[56]:


numericdata.columns


# In[57]:


import seaborn as sns
sns.boxplot(numericdata['Age'])


# In[58]:


import seaborn as sns
sns.boxplot(numericdata['Turnover'])


# In[59]:


import seaborn as sns
sns.boxplot(numericdata['Qualifications'])


# In[60]:


import seaborn as sns
sns.boxplot(numericdata['EmployeSatisfaction'])


# In[61]:


import seaborn as sns
sns.boxplot(numericdata['JobEngagement'])


# In[62]:


import seaborn as sns
sns.boxplot(numericdata['JobSatisfaction'])


# In[63]:


import seaborn as sns
sns.boxplot(numericdata['DailyBilling'])


# In[65]:


import seaborn as sns
sns.boxplot(numericdata['HourBilling'])


# In[66]:


import seaborn as sns
sns.boxplot(numericdata['MonthlyBilling'])


# In[68]:


Q1 = np.percentile(numericdata['MonthlyBilling'], 25, interpolation = 'midpoint') 
Q2 = np.percentile(numericdata['MonthlyBilling'], 50, interpolation = 'midpoint') 
Q3 = np.percentile(numericdata['MonthlyBilling'], 75, interpolation = 'midpoint') 
 
print('Q1 25 percentile of the given data is, ', Q1)
print('Q1 50 percentile of the given data is, ', Q2)
print('Q1 75 percentile of the given data is, ', Q3)

IQR = Q3-Q1
print('Interquartile range is', IQR)
 


# In[69]:


low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR
print('low_limit is', low_lim)
print('up_limit is', up_lim)


# In[70]:


numericdata[numericdata['MonthlyBilling'] < up_lim]
numericdata[numericdata['MonthlyBilling'] > low_lim]


# In[71]:


new_df = numericdata[numericdata['MonthlyBilling'] < up_lim]
new_df.shape


# In[72]:


import seaborn as sns
sns.boxplot(new_df['MonthlyBilling'])


# In[73]:


print(numericdata.iloc[0])


# In[74]:


import seaborn as sns
sns.boxplot(numericdata['MonthlyRate'])


# In[75]:


import seaborn as sns
sns.boxplot(numericdata['Work Experience'])


# In[76]:


import seaborn as sns
sns.boxplot(numericdata['Last Rating'])


# In[77]:


import seaborn as sns
sns.boxplot(numericdata['RelationshipSatisfaction'])


# In[78]:


import seaborn as sns
sns.boxplot(numericdata['TrainingTimesLastYear'])


# In[79]:


import seaborn as sns
sns.boxplot(numericdata['Work&Life'])


# In[84]:


import seaborn as sns
sns.boxplot(numericdata['YearsInCurrentRole'])


# In[83]:


numericdata.head()


# In[85]:


import seaborn as sns
sns.boxplot(numericdata['YearsSinceLastPromotion'])


# In[86]:


import seaborn as sns
sns.boxplot(numericdata['YearsWithCurrentManager'])


# In[87]:


Q1 = np.percentile(numericdata['YearsWithCurrentManager'], 25,
                   interpolation = 'midpoint')
 
Q3 = np.percentile(numericdata['YearsWithCurrentManager'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1
 
print("Old Shape: ", numericdata.shape)
 
# Upper bound
upper = np.where(numericdata['YearsWithCurrentManager'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(numericdata['YearsWithCurrentManager'] <= (Q1-1.5*IQR))
 
''' Removing the Outliers '''
numericdata.drop(upper[0], inplace = True)
numericdata.drop(lower[0], inplace = True)
 
print("New Shape: ", numericdata.shape)


# In[88]:


numericdata.shape


# In[89]:


import seaborn as sns
sns.boxplot(numericdata['YearsWithCurrentManager'])


# In[90]:


import seaborn as sns
sns.boxplot(numericdata['DistanceFromHome'])


# In[91]:


numericdata.shape


# In[92]:


# the target variable, finding the patterns and insights from data using visualizations.


#Analyse of rating features
fig=plt.figure()
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)


E_Satisfaction=numericdata.groupby('EmployeSatisfaction')['Turnover'].sum()
E_Satisfaction.plot(kind='pie',title='EmployeeSatisfaction vs Turnover',ylabel='Turnover', xlabel='EmployeSatisfaction', figsize=(15, 15),autopct="%1.1f%%",label=None,ax=ax1)

J_Satisfaction=numericdata.groupby('JobSatisfaction')['Turnover'].sum()
J_Satisfaction.plot(kind='pie',title='JobSatisfaction vs Turnover',ylabel='Turnover', xlabel='JobSatisfaction', figsize=(15, 15),autopct="%1.1f%%",label=None,ax=ax2)

Jb_Engagement=numericdata.groupby('JobEngagement')['Turnover'].sum()
Jb_Engagement.plot(kind='pie',title='JobEngagement vs Turnover',ylabel='Turnover', xlabel='JobEngagement', figsize=(15, 15),autopct="%1.1f%%",label=None, ax=ax3)

R_Satisfaction=numericdata.groupby('RelationshipSatisfaction')['Turnover'].sum()
R_Satisfaction.plot(kind='pie',title='RelationshipSatisfaction vs Turnover',ylabel='Turnover', xlabel='RelationshipSatisfaction',figsize=(15, 15),autopct="%1.1f%%",label=None,ax=ax4)

labels = "low","medium","high","veryhigh"
fig.legend(labels=labels, loc="center")

plt.show()


# In[93]:


fig=plt.figure()
ax5=fig.add_subplot(221)
ax6=fig.add_subplot(222)


L_Rating=numericdata.groupby('Last Rating')['Turnover'].sum()
L_Rating.plot(kind='pie',title='Last Rating vs Turnover',ylabel='Turnover', xlabel='Last Rating', figsize=(15, 15),autopct="%1.1f%%",label=None,ax=ax5)

W_Life=numericdata.groupby('Work&Life')['Turnover'].sum()
W_Life.plot(kind='pie',title='Work&Life vs Turnover',ylabel='Turnover', xlabel='Work&Life', figsize=(15, 15),autopct="%1.1f%%",label=None,ax=ax6)


labels = "low","medium","high","veryhigh"
fig.legend(labels=labels, loc="center")

plt.show()


# In[95]:


#Evaluate the target variable, find patterns and insights from data using visualizations.
#'DailyBilling','HourBilling','MonthlyBilling'
mi = numericdata[numericdata["Turnover"]== 1]["MonthlyBilling"]
mi = mi.reset_index()
mi.drop(['index'],axis=1, inplace=True)

mn = numericdata[numericdata["Turnover"]== 0]["MonthlyBilling"]
mn = mn.reset_index()
mn.drop(['index'],axis=1, inplace=True)

mi['mn']=mn
mi.rename(columns={'MonthlyBilling':1,'mn':0}, inplace = True)


# In[96]:


mi.plot(kind='box',figsize=(10,7))

plt.title('Boxplot of monthly Billing Vs Turnover')
plt.xlabel('Monthly Billing')
plt.ylabel('Turnover')
plt.show()


# In[97]:


mi = numericdata[numericdata["Turnover"]== 1]["MonthlyRate"]
mi = mi.reset_index()
mi.drop(['index'],axis=1, inplace=True)

mn = numericdata[numericdata["Turnover"]== 0]["MonthlyRate"]
mn = mn.reset_index()
mn.drop(['index'],axis=1, inplace=True)

mi['mn']=mn
mi.rename(columns={"MonthlyRate":1,'mn':0}, inplace = True)


# In[98]:


mi.plot(kind='box',figsize=(10,7))

plt.title('Boxplot of MonthlyRating Vs Turnover')
plt.xlabel('MonthlyRate')
plt.ylabel('Turnover')
plt.show()


# In[99]:


mi = numericdata[numericdata["Turnover"]== 1]["HourBilling"]
mi = mi.reset_index()
mi.drop(['index'],axis=1, inplace=True)

mn = numericdata[numericdata["Turnover"]== 0]["HourBilling"]
mn = mn.reset_index()
mn.drop(['index'],axis=1, inplace=True)

mi['mn']=mn
mi.rename(columns={'HourBilling':1,'mn':0}, inplace = True)


# In[100]:


mi.plot(kind='box',figsize=(10,7))

plt.title('Boxplot of Hour Billing Vs Turnover')
plt.xlabel('Hour Billing')
plt.ylabel('Turnover')
plt.show()


# In[101]:


mi = numericdata[numericdata["Turnover"]== 1]["DailyBilling"]
mi = mi.reset_index()
mi.drop(['index'],axis=1, inplace=True)

mn = numericdata[numericdata["Turnover"]== 0]["DailyBilling"]
mn = mn.reset_index()
mn.drop(['index'],axis=1, inplace=True)

mi['mn']=mn
mi.rename(columns={'DailyBilling':1,'mn':0}, inplace = True)


# In[102]:


mi.plot(kind='box',figsize=(10,7))

plt.title('Boxplot of DailyBilling Vs Turnover')
plt.xlabel('DailyBilling')
plt.ylabel('Turnover')
plt.show()


# #    Analysis of Work Experience
# #Nature of employees who stay, nature of employees who leave
#     

# In[104]:


numericdata.columns


# In[106]:


col = numericdata[['YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrentManager',
       'DistanceFromHome']]
col


# In[ ]:




