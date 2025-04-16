#%%
import pandas as pd
import numpy as np
import seaborn as sns

car_df = pd.read_csv("D:/Users/tonyn/Desktop/da_sci_4th/py_test/data_file/cars.csv")
brand_df = pd.read_csv("D:/Users/tonyn/Desktop/da_sci_4th/py_test/data_file/brand.csv")
# %%
car_df.head()
# %%
brand_df.head()
# %%
car_df['brand'] = car_df['title'].str.split().str[0]
# %%
car_df.head()
# %%
brand_df['title'] = brand_df['title'].str.upper()
# %%
brand_df.head()
# %%
# car_df['brand']=car_df['title'].apply(lambda x: x.lower().split()[0])
# car_df.merge(car_df,brand_df, on='brand')
car_df = car_df.merge(brand_df, left_on='brand', right_on='title', how='left')
# %%
car_df.head()
# %%
car_df.drop(['title_y'],axis=1, inplace=True)
# %%
car_df.rename({'title_x':'title'}, axis=1, inplace=True)
# %%
car_df.head()
#%%
car_df.duplicated()
# %%
car_df.drop_duplicates()
# %%
bonus_df = car_df.copy()
# %%
car_df.info()
# %%
car_df.head()
# %%
car_df['Engine'] = car_df['Engine'].str.replace('L', '')
# %%
car_df['Emission Class'] = car_df['Emission Class'].str.split().str[1]
# %%
car_df['Engine'] = pd.to_numeric(car_df['Engine'])
car_df['Emission Class'] = pd.to_numeric(car_df['Emission Class'])
# %%
car_df.info()
# %%
car_df.describe()
# %%
car_df.isna().mean()
# %%
car_df['Service history'].unique()
# %%
car_df.groupby('Service history')['Price'].mean()
# %%
car_df['Service history'] = car_df['Service history'].fillna('Unknown')
# %%
car_df.groupby('Service history')['Price'].mean()
# %%
car_df[car_df['Engine'].isna()]
# %%
car_df['na_values'] = car_df.isna().sum(axis = 1)
# %%
car_df.head()
# %%
len(car_df[car_df['na_values'] >= 4])
# %%
car_df = car_df[car_df['na_values'] < 4]
# %%
car_df = car_df.drop('na_values', axis=1)
# %%
car_df.isna().mean()
# %%
sns.displot(car_df['Previous Owners'])
# %%
car_df['Previous Owners'].median()
# %%
sns.displot(car_df['Engine'])
# %%
car_df['Engine'].mean()
# %%
car_df['Engine'].median()
# %%
sns.displot(car_df['Doors'])
# %%
car_df['Doors'].median()
# %%
sns.displot(car_df['Seats'])
# %%
car_df['Seats'].median()
# %%
sns.displot(car_df['Emission Class'])
# %%
car_df['Emission Class'].median()
# %%
car_df['Previous Owners'] = car_df['Previous Owners'].fillna(3)
car_df['Engine'] = car_df['Engine'].fillna(1.6)
car_df['Doors'] = car_df['Doors'].fillna(5)
car_df['Seats'] = car_df['Seats'].fillna(5)
car_df['Emission Class'] = car_df['Emission Class'].fillna(5)
# %%
car_df.isna().mean()
# %%
car_df.describe()
# %%
car_df['Price'].sort_values()
# %%
car_df['Mileage(miles)'].sort_values()
# %%
car_df[car_df['Mileage(miles)'] < 1000]
# %%
car_df = car_df[car_df['Mileage(miles)'] >= 1000]
# %%
car_df['Registration_Year'].sort_values()
# %%
car_df[car_df['Registration_Year'] >= 2025]
# %%
car_df = car_df[car_df['Registration_Year'] < 2025]
# %%
car_df['Previous Owners'].sort_values()
# %%
car_df[car_df['Previous Owners'] == 9]
# %%
car_df.groupby('brand')['Price'].agg(['mean', 'std'])
# %%
pd.pivot_table(car_df, index='brand',columns='Fuel type', values='Price')
# %%
car_df.head()
# %%
sns.scatterplot( x= car_df['Previous Owners'], y = car_df['Price'])
# %%
sns.scatterplot( x= car_df['Registration_Year'], y = car_df['Price'])
# %%
sns.scatterplot( x= car_df['Registration_Year'], y = np.log(car_df['Price']))
# %%
car_df.head()
# %%
car_df[['title','Fuel type','Body type','Gearbox','Emission Class','Service history','brand','country']].nunique()
# %%
car_df.drop('title', axis = 1, inplace = True)
# %%
brand_counts = pd.DataFrame(car_df['brand'].value_counts())
brand_counts
# %%
price_mean_brand = pd.DataFrame(car_df.groupby('brand')['Price'].mean())
price_mean_brand
# %%
brand_counts.join(price_mean_brand)
# %%
car_df
# %%
car_df = pd.get_dummies(car_df, drop_first = True)
# %%
car_df
# %%
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
# %%
car_df = pd.DataFrame(rs.fit_transform(car_df), columns=car_df.columns)
# %%
car_df.head()
# %%
from sklearn.decomposition import PCA
# %%
pca = PCA(5)
# %%
pd.DataFrame(pca.fit_transform(car_df))
# %%
(pca.explained_variance_ratio_).sum()
# %%
for i in range(2, 11):
    pca = PCA(i)
    pca.fit(car_df)
    print(i, round(pca.explained_variance_ratio_.sum(), 2))
# %%
pca = PCA(7)
pd.DataFrame(pca.fit_transform(car_df), columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7'])
# %%
bonus_df
# %%
bonus_df.groupby('country')['brand'].nunique()
# %%
num_df = bonus_df.select_dtypes(include='number')
num_df.corr()
# %%
