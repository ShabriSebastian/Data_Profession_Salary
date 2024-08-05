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

# New Column (DSJ (Day Since Join))
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

# Machine Learning Modelling: Is age and past experience affect salary?
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

data_modelling = data.copy()
LE = LabelEncoder()
data_modelling['DESIGNATION'] = LE.fit_transform(data_modelling['DESIGNATION'])
import numpy as np
np.sort(data_modelling['DESIGNATION'].unique())
LE.classes_
scaler = MinMaxScaler(feature_range=(0,1))
data_modelling[['AGE','PAST EXP', 'SALARY']] = scaler.fit_transform(data_modelling[['AGE','PAST EXP','SALARY']])
x = data_modelling[['AGE', 'PAST EXP']]
y = data_modelling['SALARY']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

reg = LinearRegression()
reg = reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mse_evaluation = mean_squared_error(y_test, y_pred)
print('MSE SCORE:', mse_evaluation)
mae_evaluation = mean_absolute_error(y_test, y_pred)
print('MAE SCORE:', mae_evaluation)
rmse_evaluation = np.sqrt(mse_evaluation)
print('RMSE SCORE:', rmse_evaluation)

