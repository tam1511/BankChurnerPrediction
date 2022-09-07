#!/usr/bin/env python
# coding: utf-8

# # Data Analysis

# * Before doing predicting analytics, we first need to explore the data 

# In[3]:


# load libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

import matplotlib
from matplotlib import rcParams
rcParams['figure.figsize'] = 12,5


# In[29]:


# read data
churn = pd.read_csv('E:/ThanhTam_DA/Project/Prediction/Bank Churners/BankChurners.csv')


# In[3]:


churn.head()


# In[30]:


# drop the first index client number and two last columns which are not relevant
churn = churn.drop(["CLIENTNUM","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],axis=1)


# In[25]:


print("No. of rows :",churn.shape[0])
print("No. of columns :",churn.shape[1])


# In[11]:


# check missing values
churn.isna().sum().sum() # no missing values


# ### 1. Churned Customer Percentage

# In[13]:


churn['Attrition_Flag'].value_counts(normalize=True)


# In[31]:


# Replace 'Existing Customer' with 0 and 'Attrited Customer' with 1
churn['Attrition_Flag'].replace({'Existing Customer':0, 'Attrited Customer':1}, inplace=True)
churn['Attrition_Flag'].value_counts(normalize=True)


# In[32]:


lab = ["Attrited Customer","Existing Customer"]
fig, ax = plt.subplots(nrows=1, ncols=1)
a = sns.barplot(x=churn["Attrition_Flag"].value_counts(), y = churn["Attrition_Flag"].value_counts())

from IPython.core.pylabtools import figsize

#Removing top and Right borders
sns.despine(bottom = False, left = False)

spots = churn["Attrition_Flag"].value_counts()
#Putting % on the bar plot
for p in ax.patches:
    ax.text(p.get_x() + 0.2, p.get_height()+4.5, '{:.2f}%'.format((p.get_height()/10127)*100))
    
#Beautifying the plot
plt.title('\n Attrited Customer Percentage \n', size=16, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.xlabel('\n Attrition_Flag \n', fontsize=13, color='black')
plt.ylabel('\n Count \n', fontsize=13, color='black')
patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=j) for i,j in zip(range(0,10),lab)]
plt.show()


# * **Attrition_Flag** is the target variable which indicates if a customer is attrited (churned or left)
# 
# * **16%** of the customers churned which I think is high for the bank need to investigate and try to understand why these customers churned

# ### 2. Genearal distribution of all numerical variables

# * Before go into details let's have a look of the distribution of all numerical variables

# In[32]:


# Lets change the Income Category to be useful for this analysis.

churn['Income_Category'] = churn['Income_Category'].replace(['$60K - $80K','Less than $40K',
                                                             '$80K - $120K','$40K - $60K','$120K +',
                                                             'Unknown'],[3,1,4,2,5,9999])

# Lets see the general distrubition of the numerical variables

axList = churn.hist(bins=20, figsize = (15, 15))


# ### 3. Churn Rate based on Gender and Marital Status

# In[33]:


churn[['Attrition_Flag','Gender','Marital_Status']].groupby(['Gender','Marital_Status']).mean().round(2)


# * Females are slightly more likely to churn than males in average
# * Males and Females got married have likely less churned than Who are single or divorced.

# ### 4. Average Age and Number of dependents by each group Attiriton Flag column

# In[35]:


churn[['Attrition_Flag','Customer_Age','Dependent_count']].groupby(['Attrition_Flag']).mean().round(2)


# * The values are quite close so there seems to be **no significant difference** between churned and not-churned based on Customer Demographic age and dependents

# In[36]:


sns.displot(data=churn, kind='hist',x='Customer_Age', hue='Attrition_Flag',
           height=7, aspect=1.2)


# * Again the ditribution of customer age is similar between churned and ot churned customers. (Customer age does not seem to bean important factor in customer churn)

# ### 5. Credit Limit Feature and Attrition Flag 

# In[40]:


plt.figure(figsize=(10,6))

sns.boxplot(data=churn, y='Credit_Limit', x='Attrition_Flag', width=0.5)


# * The credit-limit distribution of churned and not-churned customers seem pretty similar

# ### 6. The number of inactive months for customers 

# In[41]:


churn[['Attrition_Flag','Months_Inactive_12_mon']].groupby(['Months_Inactive_12_mon']).agg(['mean','count']).round(2)


# * Churn rate increased as the number of inactive months increased
# 

# ### 7. Correlation Heatmap

# In[42]:


corr = churn.corr().round(2)

plt.figure(figsize=(12,8))

sns.heatmap(corr, annot=True, cmap="YlGnBu")


# * The first row/column is what we are mosly interested in. It shows the correlation coefficients between the target variable (Attrition_Flag) and the remained variables
# 
# * Total transaction count and change are higher correlated to the target variable
# 
# * We also see the high correlation (0.81) between total transaction amount and total transaction count. We can visualize these two variable for further analysis

# In[47]:


sns.relplot(data=churn, kind='scatter', x='Total_Trans_Amt', y='Total_Trans_Ct',
           hue='Attrition_Flag', height=7)


# * One interesting finding is that customer who have done more than 100 transactions do not churn !

# In[48]:


# Let's create a boxplot of the total transaction amount 
plt.figure(figsize=(10,6))

sns.boxplot(data=churn, y='Total_Trans_Amt', x='Attrition_Flag', width=0.5)


# * We clearly see that the total transaction amount is higher for not-churned customers

# ### Variable Selection for Modelling

# In[10]:


churn.head()


# In[34]:


# Replace 'Female' with 0 and 'Male' with 1
churn['Gender'].replace({'F':0, 'M':1}, inplace=True)
churn['Gender'].value_counts(normalize=True)


# In[35]:


# Education level
churn['Education_Level'].replace({'Uneducated':0,'High School':12, 'College':15, 'Graduate':16, 'Post-Graduate':18, 'Doctorate':22, 'Unknown':99}, inplace=True)
churn.groupby('Education_Level')['Education_Level'].count()


# In[36]:


# Marital Status
churn['Marital_Status'].replace({'Single':0, 'Married':1, 'Divorced':3,'Unknown':9}, inplace=True)
churn.groupby('Marital_Status')['Marital_Status'].count()


# In[37]:


# Income category
churn.groupby('Income_Category')['Income_Category'].count()


# 
# * Our dependent variable is **Attrition_Flag** (churn or not churn)
# * Our other independent varables: 
#     * **Customer Characteristics**: Customer_Age, Gender, Dependent_Count, Education_Level, Marital_Status, Income_Category, Card_Category.
#     * **Behavioral Characteristics**: Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio having the correlation with the target variable.

# In[43]:


model = churn.drop(['Months_on_book', 'Avg_Open_To_Buy', 'Total_Trans_Ct'], axis=1)


# In[45]:


model.to_csv('E:/ThanhTam_DA/Project/Prediction/Bank Churners/churn.csv') # save to csv file


# Then I will use R for modelling

# In[ ]:




