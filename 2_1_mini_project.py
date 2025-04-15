#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("D:/Users/tonyn/Desktop/da_sci_4th/py_test/trip.csv")
data.head()
# %%
data.info()
# %%
data.describe()
# %%
data[data.duplicated( )]
# %%
data = data.drop_duplicates()
#%%
data.isna().sum()
# %%
data.isna().mean()
# %%
data = data.dropna()
# %%
data.isna().mean()
# %%
data['passenger_count'].sort_values()
# %%
sns.scatterplot(x = data.index, y = data['passenger_count'])
# %%
data = data[data['passenger_count'] <= 6]
# %%
len(data[data['passenger_count'] == 0])
# %%
data = data[data['passenger_count'] != 0]
# %%
sns.scatterplot(x = data.index, y = data['passenger_count'])
# %%
data['trip_distance'].sort_values()
# sns.scatterplot(x = data.index, y = data['trip_distance'])
# %%
data = data[data['trip_distance'] != 0]
# %%
sns.histplot(data['trip_distance'])
# %%
data.describe()
# %%
(data['fare_amount'] <= 0).sum()
# %%
data = data[data['fare_amount'] > 0]
# %%
data.sort_values('fare_amount')
sns.histplot(data['fare_amount'])
sns.scatterplot(x = data.index, y = data['fare_amount'])
# %%
data = data[data['fare_amount'] < 250]
# %%
def fare_func(x):
    if x > 150:
        return 150
    else:
        return x
    
data['fare_amount'] = data['fare_amount'].apply(fare_func)
# or
data['fare_amount'] = data['fare_amount'].apply(lambda x: 150 if x > 150 else x)
# %%
data.sort_values('fare_amount')
# %%
sns.scatterplot(x = data.index, y = data['tip_amount'])
# %%
data.sort_values('tip_amount')
# %%
data = data[data['tip_amount'] > 0]
# %%
len(data)
# %%
sns.scatterplot(x = data.index, y = data['tip_amount'])
# %%
data.head(30)
# %%
data['payment_method'].unique()
# %%
data['payment_method'].nunique()
# %%
data['payment_method'].value_counts()
# %%
data['payment_method'] = data['payment_method'].replace({
    'Debit Card': 'Card',
    'Credit Card': 'Card'
})
# %%
data['payment_method'].value_counts()
# %%
example = 'Susan Robinson'
# %%
example.split()
# %%
data['passenger_first_name'] = data['passenger_name'].str.split().str[0]
# %%
data.head()
# %%
data.info()
# %%
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
# %%
data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])
# %%
data.info()
# %%
data['travel_time'] = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']
# %%
data.head()
# %%
data.info()
# %%
data['travel_time'] = data['travel_time'].dt.total_seconds()
# %%
data.head()
# %%
data['total_amount'] = (
    data['fare_amount'] +
    data['extra'] +
    data['mta_tax'] +
    data['tip_amount'] +
    data['tolls_amount'] +
    data['improvement_surcharge']
)
# %%
sns.scatterplot(x = data['fare_amount'], y = data['trip_distance'])

# %%
sns.scatterplot(x = data['travel_time'], y = data['trip_distance'])
# %%
data['travel_time'].sort_values()
# %%
data = data[data['travel_time'] < 20000]
# %%
sns.scatterplot(x = data['travel_time'], y = data['trip_distance'])

# %%
data.describe()
# %%
data.info()