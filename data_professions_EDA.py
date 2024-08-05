import pandas as pd
data = pd.read_csv(r"C:\Users\shabr\OneDrive\Documents\Project Python DQLb\Salary Prediction of Data Professions.csv")
pd.set_option('display.max_columns', None)

print('FIRST 5 DATA')
print(data.head())
print('DATASET INFORMATION')
print(data.info())
print('DATASET SIZE:', data.shape)

# Finding missing value
data.isnull().sum()[(data.isnull().sum().values) > 0].sort_values(ascending=False)
print('Total missing value:', data.isnull().sum().sum())
null_columns = data.isnull().sum()[(data.isnull().sum().values) > 0].sort_values(ascending=False).index.to_list()
print('Columns with missing value:', null_columns)

# Handling missing value and recheck
data.dropna(axis = 0, how = 'any', inplace=True)

print('DATASET INFORMATION')
print(data.info())
print(data.describe())
print('DATASET SIZE:', data.shape)

# Finding duplicate data
print(data.duplicated().sum())
detect_duplicate = data.copy()
detect_duplicate['is_duplicate'] = data.duplicated(keep = False)
detect_duplicate['original_row'] = data.duplicated(keep = 'first')

from sklearn.metrics import confusion_matrix
confusion_matrix(detect_duplicate['is_duplicate'], detect_duplicate['original_row'] ) ## After drop duplicate, length must be 2470
len(data.drop_duplicates())
len(data.drop_duplicates(subset=['FIRST NAME', 'LAST NAME'])) # Just in case
data.drop_duplicates(inplace=True)

print('DATASET SIZE:', data.shape)

# Reset Index
data = data.reset_index()

# New column: "FULL NAME"
data.insert(2, 'FULL NAME', data['FIRST NAME'] + ' ' + data['LAST NAME'])
print('FIRST 5 DATA')
print(data.head())

# Datetime and New Columns (DSJ (Day Since Join))
import datetime
data['DOJ'] = pd.to_datetime(data['DOJ'])
data['CURRENT DATE'] = pd.to_datetime(data['CURRENT DATE'])
data['CURRENT DATE'].unique() ## Only 01-07-2016
data.insert(5, 'DSJ', data['CURRENT DATE'] - data['DOJ'])
data['DSJ'].max() #2540 days
data['DSJ'].min() #341 days
def date(x):
    if len(x) == 8:
        val = x[0:3]
    elif len(x) == 9:
        val = x[0:4]
    return val
data['DSJ'] = data['DSJ'].astype('str')
data['DSJ'] = data['DSJ'].apply(date).astype('int')
del data['CURRENT DATE']
data.info()

#Delete Unnecessary Columns
del data['index']

# Swap columns index
col = data.columns.to_list()
full_name_index = col.index('FULL NAME')
last_name_index = col.index('LAST NAME')
dsj_index = col.index('DSJ')
doj_index = col.index('DOJ')

col[full_name_index], col[last_name_index] = col[last_name_index], col[full_name_index]
col[dsj_index], col[doj_index] = col[doj_index], col[dsj_index]

data = data[col]

print('FIRST 5 DATA')
print(data.head())
print('DATASET INFORMATION')
print(data.info())
print('DATASET SIZE:', data.shape)


#EDA 
data_eda = data.copy()

import numpy as np
np.sort(data['AGE'].unique())
def age_interval(age):
    if (age>=21) & (age <=25):
        val = '21-25'
    elif (age>=26) & (age <=30):
        val = '26-30'
    elif (age>=31) & (age <=35):
        val = '31-35'
    elif (age>=36) & (age <=40):
        val = '36-40'
    else:
        val = '41-45'
    return val
data_eda['AGE INTERVAL'] = data_eda['AGE'].apply(age_interval)

import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------------------------------------------------------------------#
# Distribution of Age Interval
sns.histplot(data_eda['AGE INTERVAL'].sort_values())
plt.title('Distribution of Age Interval')
plt.ylabel('Frequency')
plt.show() ## Age 21-25 are dominating
print('Age 21-25 ratio:', len(data_eda[data_eda['AGE INTERVAL'] == '21-25'])/len(data_eda))
# 78% of employee's age are 21-25
#___________________________________________________________________________________________#
# Age 21-25 Designation
data_eda_21_25 = data_eda[data_eda['AGE INTERVAL'] == '21-25']
data_eda_21_25['DESIGNATION'].unique()
data_eda_21_25.groupby('DESIGNATION')['FULL NAME'].count().plot(kind = 'pie', autopct='%1.0f%%')
plt.show() 
# 96% of them are Analyst, and 4% of them are Senior Analyst
#___________________________________________________________________________________________#
# Descriptive Statistics of 21-25 Age's Salary
data_eda_21_25[data_eda_21_25['DESIGNATION'] == 'Analyst']['SALARY'].describe()
data_eda_21_25[data_eda_21_25['DESIGNATION'] == 'Senior Analyst']['SALARY'].describe()
#___________________________________________________________________________________________#
# Employee's Leaves Allowance
data_eda['LEAVES ALLOWANCE'] = data_eda['LEAVES USED'] + data_eda['LEAVES REMAINING']
print(data_eda['LEAVES ALLOWANCE'].std()) 
# The leave allowance for all employees is the same (30 Leaves Allowance)
#___________________________________________________________________________________________#
# The number of leaves for each employee based on age
data_eda.groupby(['AGE INTERVAL'])[['LEAVES USED', 'LEAVES REMAINING']].mean().plot(kind = 'bar')
plt.show() 
# Employee with 21-25 age interval have a relatively same number of LEAVES USED with the older ages
#___________________________________________________________________________________________#
# Minimum, Average, and Maximum RATINGS of each age interval
fig, axes = plt.subplots(1,3, figsize = (10,5))
sns.barplot(data_eda.groupby('AGE INTERVAL')['RATINGS'].min(), ax=axes[0], color='orange')
axes[0].set_title('Minimum Ratings')
sns.barplot(data_eda.groupby('AGE INTERVAL')['RATINGS'].mean(), ax=axes[1], color='red')
axes[1].set_title('Average Ratings')
sns.barplot(data_eda.groupby('AGE INTERVAL')['RATINGS'].max(), ax=axes[2], color='brown')
axes[2].set_title('Maximum Ratings')
plt.tight_layout()
plt.show() 
# The maximum and minimum ratings of each age interval is relatively same, 
# but the average ratings in the 36-40 age interval is slightly higher than the others
#___________________________________________________________________________________________#
# Salary of each age interval and designations
fig, axes = plt.subplots(1, 3, figsize = (20,5))
sns.barplot(data_eda.groupby('AGE INTERVAL')['SALARY'].mean().sort_index(), palette='inferno', ax=axes[0])
data_eda_piv = data_eda.pivot_table(columns='DESIGNATION',values='SALARY',index='AGE INTERVAL', aggfunc='mean').stack().reset_index()
sns.barplot(x=data_eda_piv['AGE INTERVAL'],y=data_eda_piv[0],hue=data_eda_piv['DESIGNATION'], ax=axes[1])
axes[1].set_ylabel('SALARY')
sns.barplot(data_eda.groupby('DESIGNATION')['SALARY'].mean().sort_values(), palette='flare', ax=axes[2])
plt.tight_layout()
plt.show() 
# Asumption: There is a positive correlation between AGE, DESIGNATION, and SALARY
#___________________________________________________________________________________________#
# Number of Male and Female employee in each DESIGNATION
sex_designation_piv = data_eda.pivot_table(index='SEX',
                                           columns='DESIGNATION',
                                           values='FULL NAME',
                                           aggfunc='nunique').unstack().reset_index().sort_values(by=0, ascending=False)
sns.barplot(data=sex_designation_piv,x='DESIGNATION',y=0,hue='SEX',palette='inferno')
plt.title('Designation Frequency by Gender')
plt.ylabel('Frequency')
plt.show()
# The number of Male and Female in each DESIGNATION is balance
#___________________________________________________________________________________________#
# Salary of Male and Female employee in each DESIGNATION
sex_salary_piv = data_eda.pivot_table(index='SEX',
                                           columns='DESIGNATION',
                                           values='SALARY',
                                           aggfunc='mean').unstack().reset_index().sort_values(by=0)
sns.barplot(data=sex_salary_piv,x='DESIGNATION',y=0,hue='SEX',palette='flare')
plt.title('Designation Salary by Gender')
plt.ylabel('SALARY')
plt.tight_layout()
plt.show()
# The salary of Male and Female in each DESIGNATION is balance
#___________________________________________________________________________________________#
# Designation average ratings by gender
sex_ratings_piv = data_eda.pivot_table(index='SEX',
                                       columns='DESIGNATION',
                                       values='RATINGS',
                                       aggfunc='mean').unstack().reset_index()
sex_ratings_piv['SALARY'] = sex_salary_piv[0]
sex_ratings_piv = sex_ratings_piv.sort_values(by='SALARY')
sns.barplot(data=sex_ratings_piv, x='DESIGNATION',y=0,hue='SEX')
plt.title('Average Ratings each Designation by Gender')
plt.show()
# Female employees have a slightly higher ratings than male employees, except in manager designation
#___________________________________________________________________________________________#

# EDA 2 (Correlation)

# Encode SEX and DESIGNATION columns 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data_eda['SEX'] = LE.fit_transform(data_eda['SEX'])
np.sort(data['SEX'].unique())
LE.classes_
data_eda['DESIGNATION'] = LE.fit_transform(data_eda['DESIGNATION'])
LE.classes_
np.sort(data['DESIGNATION'].unique())

# Correlation
data_eda.corr(numeric_only = True)

#df is your dataframe
df[["FirstColumnName","SecondColumnName"]]
#df is your dataframe
df.plot.barh(stacked=True, x='ColumnOne', y='ColumnTwo')