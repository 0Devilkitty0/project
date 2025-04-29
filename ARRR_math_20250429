#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
retail_df = pd.ExcelFile("D:/Users/tonyn/Desktop/da_sci_4th/math_test/online_retail_II.xlsx")

# %%
df_1 = pd.read_excel(retail_df, sheet_name='Year 2009-2010') 
df_2 = pd.read_excel(retail_df, sheet_name='Year 2010-2011') 
df = pd.concat([df_1, df_2])


# %%
# 문제 1: 데이터 로드 및 탐색
df.head(5)
# %%
df.info()


# %%
# 문제 2: 데이터 전처리
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['Customer ID'])
# %%
df['Customer ID'] = df['Customer ID'].astype(int)
# %%
df.sort_values(by='InvoiceDate', inplace=True)
# %%
df['Total'] = df['Quantity'] * df['Price']


# %%
# 문제 3: 나라별 구매 인원 시각화
customer_country = (
    df
    .groupby('Country')['Customer ID']
    .nunique()
    .sort_values(ascending=False)
    .reset_index()
)
customer_country.columns = ['Country', 'UniqueCustomers']
# %%
sns.barplot(data=customer_country, x='Country', y='UniqueCustomers')
plt.title('Unique Customers by Country')
plt.xlabel('Country')
plt.xticks(rotation=90)
plt.ylabel('Unique Customers')
for i, row in customer_country.iterrows():
    plt.text(i, row['UniqueCustomers'] + 5, int(row['UniqueCustomers']), ha='center', va='bottom', fontsize=8)
plt.show()


# %%
# 문제 4: Acquisition (고객 유입 분석)
first_purchase = (
    df
    .groupby('Customer ID')['InvoiceDate']
    .min()
    .reset_index()
)
# %%
first_purchase.columns = ['Customer ID', 'first_purchase']
first_purchase['first_purchase'] = first_purchase['first_purchase'].dt.to_period('M')
# %%
month_new = (
    first_purchase
    .groupby('first_purchase')['Customer ID']
    .nunique()
    .reset_index()
)
month_new.columns = ['month', 'count']
# %%
sns.barplot(data=month_new, x='month', y='count')
plt.title('First purchase by month')
plt.xlabel('month')
plt.xticks(rotation=90)
plt.ylabel('First purchase')
plt.show()


# %%
# 문제 5: Activation (고객 활성화 분석)
total_purchase = df.groupby('Customer ID')['Total'].sum()
# %%
active = total_purchase[total_purchase >= 50]
# %%
total_customers  = total_purchase.count()
active_customers = active.count()
active_per  = active_customers / total_customers
# %%
print(f"전체 고객 수: {total_customers}명")
print(f"활성화된 고객 수 (첫 구매 후 £50 이상 지출한 고객): {active_customers}명")
print(f"활성화율: {active_per:.2%}")


# %%
# 문제 6: Retention:코호트 분석(고객)
df_from_2010 = df[df['InvoiceDate'] >= '2010-01-01']
df_from_2010['Quarter'] = df_from_2010['InvoiceDate'].dt.to_period('Q')
# %%
df_from_2010.head()
# %%
df_from_2010['Cohort_Q'] = df_from_2010.groupby('Customer ID')['Quarter'].transform('min')
# %%
df_from_2010['Cohort_Index'] = (df_from_2010['Quarter'].astype(int) - df_from_2010['Cohort_Q'].astype(int)) + 1
# %%
cohort_data = (df_from_2010
               .groupby(['Cohort_Q', 'Cohort_Index'])['Customer ID']
               .nunique()
               .reset_index())


cohort_counts = cohort_data.pivot(index='Cohort_Q', columns='Cohort_Index', values='Customer ID')
cohort_sizes = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
# %%
plt.figure(figsize=(12, 6))
sns.heatmap(
    data = retention,
    annot=True,                # 셀 내부에 값 표시
    fmt='.2f',                 # 텍스트 형식 (소수점 2자리)
    cmap='Greens',             # 색상 팔레트
    cbar_kws={'label': 'Retention Rate, %'},  # 컬러바 제목
    linewidths=0.5,            # 셀 간격
    linecolor='gray',          # 셀 경계 색상
    vmin=0, 
    vmax=100)           # Retention Rate의 범위 설정
plt.title('Cohort Analysis')
plt.ylabel('Quarter')
plt.show()


# %%
# 문제 7: Retention: 코호트 분석(평균 구매수량)
purchase_data = (
    df_from_2010
    .groupby(['Cohort_Q', 'Cohort_Index'])['Quantity']
    .mean()
    .reset_index()
)


avg_quantity = purchase_data.pivot(
    index='Cohort_Q',
    columns='Cohort_Index',
    values='Quantity'
)
plt.figure(figsize=(12, 8))
sns.heatmap(avg_quantity, 
            annot=True,                # 셀 내부에 값 표시
            fmt='.1f',                 # 텍스트 형식 (소수점 1자리)
            cmap='Blues',              # 색상 팔레트
            cbar_kws={'label': 'Average Quantity'},  # 컬러바 제목
            linewidths=0.5,            # 셀 간격
            linecolor='gray',          # 셀 경계 색상
            vmin=0                     # 최소값 설정 (필요 시 조정 가능)
        )
plt.title("Cohort Analysis: Average Quantity")
plt.ylabel("Quarter")
plt.show()


# %%
# 문제 8: Revenue : ARPU
df.head()
# %%
df.info()
# %%
df[df['Quantity'] <= 0]
# %%
df = df[df['Quantity'] > 0]  
df = df[df['Price'] > 0]  
# %%
df['Revenue'] = df['Quantity'] * df['Price']
# %%
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
# %%
monthly_user_revenue = df.groupby(['InvoiceMonth', 'Customer ID'])['Revenue'].sum().reset_index()
# %%
arpu = (monthly_user_revenue
        .groupby('InvoiceMonth')['Revenue']
        .mean())
# %%
plt.figure(figsize=(12, 6))
arpu.plot(kind='line', marker='o', color='skyblue')
plt.title("Monthly ARPU (Average Revenue Per User)")
plt.xlabel("Year-Month")
plt.ylabel("ARPU (£)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# %%
# 문제 9: Revenue(CLV)
clv_data = df.groupby('Customer ID').agg(
    Total_Revenue=('Revenue', 'sum'),
    Total_Orders=('Invoice', 'nunique'),
    first_purchase=('InvoiceDate', 'min'),
    last_purchase=('InvoiceDate', 'max')
).reset_index()

#%%
clv_data['avg_order_value'] = clv_data['Total_Revenue'] / clv_data['Total_Orders']
clv_data['Frequency'] = clv_data['Total_Orders'] / (
    df.groupby('Customer ID')['InvoiceDate'].nunique().values
)
clv_data['retention_time'] = (clv_data['last_purchase'] - clv_data['first_purchase']).dt.days + 1

# %%
clv_data['CLV'] = clv_data['avg_order_value'] * clv_data['Frequency'] * clv_data['retention_time']
# %%
clv_data = clv_data.sort_values(by='CLV', ascending=False)
# %%
clv_data


# %%
# 기초통계 문제

# 문제10
# 어떤 회사의 고객 대기시간은 5분에서 15분 사이의 균등분포를 따릅니다. 
# 고객 100명이 대기한 시간을 시뮬레이션하고, 평균 대기시간과 표준편차를 계산하세요.
waiting_time = np.random.uniform(low=5, high=15, size=100)

mean_waiting_time = np.mean(waiting_time)
std_waiting_time = np.std(waiting_time)

print(f"평균 대기시간: {mean_waiting_time:.2f}")
print(f"표준편차: {std_waiting_time:.2f}")


#%%
# 문제11
# 한 신제품의 초기 성공 확률이 0.3이라고 가정합니다. 
# 10회의 시뮬레이션에서 성공한 횟수를 구하고, 각 성공 여부를 출력하세요.
outcomes = np.random.binomial(n=1, p=0.3, size=10)

success_count = np.sum(outcomes)

print(f"각 시도 결과: {outcomes}")
print(f"성공 횟수: {success_count}")

# %%
# 문제12
# 한 수업에서 학생 20명이 5문제로 구성된 퀴즈를 치릅니다. 각 문제의 정답 확률은 0.7이라고 가정할 때, 각 학생이 맞힌 점수를 시뮬레이션하고, 전체 학생의 평균 점수를 계산하세요.
scores = np.random.binomial(n=5, p=0.7, size=20)

average_score = np.mean(scores)

print(f"학생별 점수: {scores}")
print(f"평균 점수: {average_score:.2f}")


# %%
# 문제13
# 한 공장에서 생산되는 제품의 무게는 평균 50g, 표준편차 5g의 정규분포를 따릅니다.
# 1000개의 제품 무게를 시뮬레이션하고, 무게가 45g 이상 55g 이하인 제품의 비율을 계산 (변수명: `within_range`)하세요. 
# 무게 분포의 히스토그램을 그리세요.
from scipy.stats import norm

weights = np.random.normal(loc=50, scale=5, size=1000)

within_range = np.mean((weights >= 45) & (weights <= 55))

print(f"45g 이상 55g 이하 비율: {within_range:.2%}")


plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(weights, bins=30, density=True, alpha=0.6, label='Weights')

x = np.linspace(min(bins), max(bins), 1000)
pdf = norm.pdf(x, loc=50, scale=5)
plt.plot(x, pdf, 'r-', lw=2, label='PDF')

plt.title("Product Weight Distribution")
plt.xlabel("Weight(g)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
