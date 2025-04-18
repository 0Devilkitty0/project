#%%
import pandas as pd
import numpy as np
import seaborn as sns

cc_df = pd.read_csv("D:/Users/tonyn/Desktop/da_sci_4th/py_test/data_file/fraud.csv")
# %%
cc_df[cc_df.duplicated()]
# %%
pd.set_option('display.max_columns', 50)
cc_df.head()
# %%
cc_df.info()
# %%
cc_df.describe()
# %%
# 불필요 칼럼 제거

cc_df.head(3)
# %%
cc_df['merchant'].nunique()
# %%
cc_df['job'].nunique()
# %%
cc_df['cc_num'].nunique() # 124
# %%
cc_df.drop(['merchant','first','last','street','city','state','zip','job','trans_num','unix_time'], axis = 1, inplace= True)
# %%
cc_df.sort_values('cc_num')
cc_df
# %%

# %%
# 구매금액의 z-score 계산하기
###########################################################
# 예시)
temp = pd.DataFrame({'a': [10,20,30,20,10,200], 
                     'b': [100,300,200,150,250,200], 
                     'c': [10, 500, 20, 250, 25, 200]})
# %%
temp
# %%
temp.mean() 
# %%
temp.std()
# %%
# 각각의 데이터에 대해 a 컬럼의 z-score를 계산 
# (data - mean) / std

(temp['a'] - 48.33) / 74.67
# %%
# 각각의 데이터에 대해 b 컬럼의 z-score를 계산 
(temp['b'] - 200) / 70.71

# %%
# 각각의 데이터에 대해 c 컬럼의 z-score를 계산 
(temp['c'] - 167.5) / 192.50

###########################################################
# %%
# 카드 별 결제 금액의 z-score
cc_df['cc_num'].value_counts()
# %%
amt_info = cc_df.groupby('cc_num')['amt'].agg(['mean','std']).reset_index()
# %%
amt_info
# %%
amt_info.to_pickle('D:/Users/tonyn/Desktop/da_sci_4th/py_test/py_file/amt_info.pkl')
# %%
cc_df = cc_df.merge(amt_info, on='cc_num', how='left')
# %%
cc_df
# %%
cc_df['amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']
# %%
cc_df.head()
# %%
cc_df[cc_df['is_fraud'] == 1]
# %%
cc_df.drop(['mean','std'], axis = 1, inplace = True)
# %%
# 카드, 사용 카테고리 별 결제 금액의 z-score
cat_info = cc_df.groupby(['cc_num','category'])['amt'].agg(['mean','std']).reset_index()
# %%
cat_info
# %%
cat_info.to_pickle('D:/Users/tonyn/Desktop/da_sci_4th/py_test/py_file/cat_info.pkl')
# %%
cc_df = cc_df.merge(cat_info, on=['cc_num','category'], how='left')
# %%
cc_df
# %%
cc_df['cat_amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']
# %%
cc_df
# %%
cc_df.drop(['mean','std'], axis =1 , inplace = True)
# %%
cc_df.head()
# %%
# 결제 시간 관련 분석

cc_df.info()
# %%
cc_df.head()
# %%
cc_df['hour'] = pd.to_datetime(cc_df['trans_date_trans_time']).dt.hour
# %%
cc_df.head(20)
# %%
def hour_func(x):
    if (x >= 6) & (x < 12):
        return 'morning'
    elif (x >= 12) & (x < 18):
        return 'afternoon'
    elif (x >= 18) & (x < 23):
        return 'night'
    else:
        return 'evening'
# %%
cc_df['hour_cat'] = cc_df['hour'].apply(lambda x : hour_func(x))
# %%
cc_df.head()
# %%
cc_df['hour_cat'].value_counts()
# %%
all_cnt = cc_df.groupby('cc_num')['amt'].count().reset_index()
# %%
hour_cnt = cc_df.groupby(['cc_num','hour_cat'])['amt'].count().reset_index()
# %%
all_cnt.head()
# %%
hour_cnt.head()
# %%
hour_cnt = hour_cnt.merge(all_cnt, on='cc_num', how='left')
# %%
hour_cnt.head()
# %%
hour_cnt.rename({'amt_x' : 'hour_cnt', 'amt_y' : 'total_cnt'}, axis=1, inplace=True)
# %%
hour_cnt
# %%
hour_cnt.head()
# %%
hour_cnt['hour_perc'] = hour_cnt['hour_cnt'] / hour_cnt['total_cnt'] * 100
# %%
hour_cnt.head(10)
# %%
hour_cnt.tail(10)
# %%
hour_cnt = hour_cnt[['cc_num','hour_cat','hour_perc']]
# %%
hour_cnt.to_pickle('D:/Users/tonyn/Desktop/da_sci_4th/py_test/py_file/hour_cnt.pkl')
# %%
cc_df = cc_df.merge(hour_cnt, on=['cc_num', 'hour_cat'], how='left')
# %%
cc_df.head()
# %%
cc_df.drop(['trans_date_trans_time', 'hour', 'hour_cat'], axis =1 , inplace = True)
# %%
# 거리 관련 분석
from geopy.distance import distance

#ex)
distance((48.8878, -118.2105), (49.159047, -118.186462)).km
# %%
cc_df['distance'] = cc_df.apply(lambda x: distance((x['lat'], x['long']), (x['merch_lat'], x['merch_long'])).km, axis = 1)
# %%
cc_df.head()
# %%
dist_info = cc_df.groupby('cc_num')['distance'].agg(['mean','std']).reset_index()
# %%
dist_info
# %%
dist_info.to_pickle('D:/Users/tonyn/Desktop/da_sci_4th/py_test/py_file/dist_info.pkl')
# %%
cc_df = cc_df.merge(dist_info, on='cc_num', how='left')
# %%
cc_df.head()
# %%
cc_df['dist_z'] = (cc_df['distance'] - cc_df['mean']) / cc_df['std']
# %%
cc_df.head()
# %%
cc_df.drop(['lat','long','merch_lat','merch_long','mean','std'], axis = 1, inplace = True)
# %%
cc_df.head()
# %%
cc_df.info()
# %%
cc_df['dob'] = pd.to_datetime(cc_df['dob']).dt.year
# %%
cc_df.head()
# %%
cc_df['category'].nunique()

#%%
#################################################
cc_df
# %%
gen_amt = cc_df.groupby('gender')['amt'].agg(['mean','std']).reset_index()
# %%
gen_amt
# %%
cc_df = cc_df.merge(gen_amt, on='gender', how='left')

#################################################
# # %%
cc_df
# %%
cc_df['gen_amt_z'] = (cc_df['distance'] - cc_df['mean']) / cc_df['std']

# %%
cc_df = pd.get_dummies(cc_df, columns=['category'], drop_first=True)
# %%
cc_df.head()
# %%
cc_df.drop('cc_num', axis = 1, inplace = True)
# %%
cc_df
# %%
###################################################
# %%

# %%
# from datetime import datetime
# pd.to_datetime(1325376044, unit='s')
# datetime.date.fromtimestamp(1388534355)
# %%

# %%

# %%

# %%

# %%
