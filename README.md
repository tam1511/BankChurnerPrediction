# Bank Churner Prediction : Project Overview
* For this analytics, I will look into a bank customer data to predict whether the customer will leave the credit card services of the bank. By predicting which customer is a high risk of churning (leave) is valuable for the bank with returning customers.
* Data was prepared by using **Python** and apply modelling by **R** (R is one of the predominant languages in data science ecosystem and makes it simple to efficiently implement statistical techniques and thus it is excellent choice for machine learning tasks)

# Code and Resources used
**Python** Jupyter notebook ver 3.10
**R Studio** RMarkdown
**EDA article** : https://towardsdatascience.com/practical-data-analysis-with-pandas-and-seaborn-8fec3cb9cd16

# Data Description

The data set contains information about the customers of a bank. 
* Attrition Flag : Existing Customer (0) and Attrited Customer (1)
* Demographic Variables : Customer Age, Gender, Marital Status, number of dependents, Education level, Income
* Behavior Variables : Type of Cards, Months on book, The amount transaction, etc

# EDA 
Some highlight findings focusing on the target variable below :
![Data Scientist's Salary EDA - Jupyter Notebook - AVG Secure Browser 9_8_2022 12_20_22 AM](https://user-images.githubusercontent.com/99704273/188916319-c33e8ed0-f29f-43e7-8af9-4f9fdd5f15b5.png)
![image](https://user-images.githubusercontent.com/99704273/188916060-7c60d038-d40c-4411-92b0-59a3f0095229.png)
![image](https://user-images.githubusercontent.com/99704273/188916406-0afefa81-7c10-41d1-876f-03ce07f1d1f2.png)

# Model Building

First I transformed some categorical variable into numeric ( i.e income) and readable code. I also split the data into train and tests sets with a test size of 20%.
I tried two different models and evaluated the accuracy based on the test error rate

* **Logistic Regression** the small p-values associated with almost all of the predictors 
* However **RandomForest** with the lower test error rate 

