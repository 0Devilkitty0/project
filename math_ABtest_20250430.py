#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t

# %%
# ### 문제 1)
# 빵집에서는 매일 아침에 구워지는 식빵 한 개의 평균 무게가 500g이 되도록 맞추고자 합니다. 
# 빵집 주인은 오늘 아침에 구운 식빵 중에서 랜덤하게 25개의 식빵을 샘플링하여 무게를 측정했습니다. 
# 그 결과, 표본 평균은 495g, 표준편차는 10g으로 나왔습니다. 
# 빵집 주인이 목표한 500g의 무게를 충족하고 있는지(다시말해 목표 무게를 넘는지 안 넘는지) 
# 5% 유의수준에서 검정해보세요

average_weight = 495
std_weight = 10
pop_mean = 500
n = 25
alpha = 0.05

t_statistic = (average_weight - pop_mean) / (std_weight / (n ** 0.5))

p_value = 2 * stats.t.cdf(t_statistic, df=n-1) if t_statistic < 0 else 2 * (1 - stats.t.cdf(t_statistic, df=n-1))

# 결과 출력
if p_value < alpha:
    print(f"t-값: {t_statistic:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 기각합니다. 빵의 평균 무게는 목표와 다릅니다.")
else:
    print(f"t-값: {t_statistic:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 채택합니다. 빵의 평균 무게는 목표와 통계적으로 차이가 없습니다.")

# %%
# 문제2) 분포 시각화
df = 24
alpha = 0.05
t_statistic = -2.5
#%%
t_crit = t.ppf(1 - alpha / 2, df)
#%%
x = np.linspace(-4, 4, 1000)
y = t.pdf(x, df)

plt.figure(figsize=(10, 6))
plt.plot(x, y)

plt.fill_between(x, y, where=(x <= -t_crit) | (x >= t_crit), color='green', alpha=0.3, label='Rejection Region')

plt.axvline(-t_crit, color='green', linestyle='--')
plt.axvline(t_crit, color='green', linestyle='--')

plt.axvline(t_statistic, color='red', linestyle='--')

plt.title("t-distribution")
plt.show()

# %%
# 문제 3) 단일 t 표본 검정
# 어느 학교에서 새로운 교육 프로그램을 도입한 후 학생들의 수학 성적이 향상되었는지 확인하려고 합니다. 
# 프로그램 도입 후 무작위로 선택한 16명의 학생들의 수학 성적 평균은 78점이고, 모집단의 평균은 75점입니다. 
# 모집단의 표준편차는 알 수 없다고 합니다.
# 유의수준 0.05에서 이 교육 프로그램이 성적 향상에 효과가 있는지 단일 표본 t-검정을 실시하세요.

sample_scores = [79, 77, 80, 76, 78, 81, 75, 79, 77, 80, 78, 76, 82, 77, 79, 78]

all_average = 75
alpha = 0.05

t_stat, p_value = stats.ttest_1samp(sample_scores, all_average)

# 결과 출력
if p_value < alpha:
    print(f"t-값: {t_stat:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 기각합니다. 교육프로그램은 효과가 있습니다.")
else:
    print(f"t-값: {t_stat:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 채택합니다. 교육프로그램은 효과가 없습니다.")
# %%
# 문제 4) 독립 표본 t 검정
# 한 연구소에서 두 가지 새로운 다이어트 프로그램의 효과를 비교하려고 합니다. 
# 연구소는 두 그룹의 참가자들을 대상으로 12주간 다이어트 프로그램을 진행한 후 체중 감소량을 측정했습니다.
# 유의수준 5% 에서 두 그룹 간 평균 체중 감소량에 유의미한 차이가 있는지 독립 표본 t-검정을 실시하세요.

# 그룹 A와 B의 체중 감소량 데이터
group_A = [5.1, 4.7, 6.2, 4.9, 5.3, 6.1, 5.0, 5.8, 4.8, 5.2]
group_B = [4.3, 4.1, 3.8, 4.6, 4.0, 4.5, 3.7, 4.2, 3.9, 4.4, 3.5, 4.3]

alpha = 0.05

levene_stat, levene_p = stats.levene(group_A, group_B)

t_stat, p_value = stats.ttest_ind(group_A, group_B)

# 결과 출력
if p_value < alpha:
    print(f"t-값: {t_stat:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 기각합니다. 다이어트 프로그램은 효과가 있습니다.")
else:
    print(f"t-값: {t_stat:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 채택합니다. 다이어트 프로그램은 효과가 없습니다")

# %%
# 문제 5) 대응표본 t검정
# 대응표본 t-검정은 두 집단간 평균 차이를 비교할 때 사용되는 점은 독립 t검정 동일하나, 
# 같은 집단에서 두 번 수집할 때 사용되는 검정입니다.( ex 고혈압 투여 전후 환자 단일 그룹의 혈압의 차) 
# `scipy.stats` docs에서 적절한 함수를 찾아보고 적용해보세요. 

# 운동 프로그램 전후의 체중 변화를 분석하기 위해 10명의 참가자의 체중을 측정했습니다. 유의수준 5%에서 운동 프로그램이 체중 감소에 효과가 있는지 **대응 표본 t-검정**을 실시하세요.
# 또한, 대응표본t검정에서 등분산 검정이 필요한지 고민해봅시다.

before = np.array([70, 80, 65, 90, 75, 85, 78, 82, 68, 73])
after = np.array([68, 78, 64, 88, 74, 83, 77, 80, 67, 72])

alpha = 0.05

t_stat, p_value = stats.ttest_rel(before, after)

# 결과 출력
if p_value < alpha:
    print(f"t-값: {t_stat:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 기각합니다. 운동 프로그램은 효과가 있습니다.")
else:
    print(f"t-값: {t_stat:.4f}, p-value: {p_value:.4f}. 유의수준 {alpha}에서 귀무가설을 채택합니다. 운동 프로그램은 효과가 없습니다.")


# %%
# 문제 6) 표본 추출
# Quest 05-01의  Online Retail II 데이터에서 표본을 추출하여 모집단의 평균을 추정해보세요.
# 영국(United Kingdom)에서 주문된 데이터 에서 30개, 100개, 300개의 샘플을 무작위 추출하여 평균 구매 금액(Total Price)를 계산해보세요. 
# 표본의 크기가 커질 수록 모집단의 평균과 가까워지는지 확인해보세요.

retail_df = pd.ExcelFile("D:/Users/tonyn/Desktop/da_sci_4th/math_test/online_retail_II.xlsx")
df_1 = pd.read_excel(retail_df, sheet_name='Year 2009-2010')
df_2 = pd.read_excel(retail_df, sheet_name='Year 2010-2011')
df = pd.concat([df_1, df_2])
#%%
df['TotalPrice'] = df['Quantity'] * df['Price']

uk_df = df[df['Country'] == 'United Kingdom'].copy()

uk_df = uk_df.dropna(subset=['TotalPrice'])

sample_30 = uk_df['TotalPrice'].sample(n=30, random_state=1).mean()
sample_100 = uk_df['TotalPrice'].sample(n=100, random_state=1).mean()
sample_300 = uk_df['TotalPrice'].sample(n=300, random_state=1).mean()

print(f"Sample size: 30, Mean TotalPrice: {sample_30:.2f}")
print(f"Sample size: 100, Mean TotalPrice: {sample_100:.2f}")
print(f"Sample size: 300, Mean TotalPrice: {sample_300:.2f}")


# %%
# 문제 7) 신뢰구간
# 영국 데이터에서 TotalPrice를 사용하여 95% 신뢰 구간을 계산하세요. 
# 또한 표본의 크기가 30,100, 300으로 변하면서 신뢰구간이 변하는 형태를 확인해 보세요.

def confidence_interval(data, confidence=0.95):
    mean = data.mean()
    std_err = stats.sem(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=std_err)
    return mean, interval

sample_sizes = [30, 100, 300]

for size in sample_sizes:
    sample = uk_df['TotalPrice'].sample(size, random_state=42)
    mean, interval = confidence_interval(sample)
    min_interval, max_interval = float(interval[0]), float(interval[1])
    print(f"Sample size: {size}, Mean: {mean:.2f}, 95% CI: {min_interval:.2f} ~ {max_interval:.2f}")

# %%
# 문제 8) 가설검정 t-test
# 영국과 독일의 고객의 평균 구매금액(Total Price)가 동일한지 검정해보세요. 
# 귀무가설과 대립가설을 세우고 통계검정을 통해 결과를 해석하세요
# 영국과 독일의 분포는 등분산성은 따른다고 가정

uk = df[df['Country'] == 'United Kingdom']['TotalPrice']
germany = df[df['Country'] == 'Germany']['TotalPrice']

t_stat, p_value = stats.ttest_ind(uk, germany, equal_var=True)

# 결과 출력
print(f"t_stat: {t_stat:.4f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("영국과 독일 고객의 평균 구매 금액에 유의한 차이가 있습니다.")
else:
    print("영국과 독일 고객의 평균 구매 금액에 유의한 차이가 없습니다.")
# %%
### 2. A/B 테스트 ###
# 문제 9 ~ 10) A/B test스타트업A에서 새로운 여행 패키지 상품 판매를 진행하고자 합니다. 
# 패키지 판매 기획자는 새로운 패키지의 상품 판매 효율을 높이고 싶어하며, 
# 이를 위해 기존에 상품이 판매되던 웹 페이지(페이지 A)가 아닌 새로운 웹 페이지(페이지 B)를 통해 판매하고자 합니다. 
# 패키지 판매 기획자는 신규 웹페이지(페이지B)가 기존(페이지A) 대비 효과가 좋은 지 확인하기 위해 A/B 테스트를 진행하였습니다.

# 페이지 A: 기존에 운영하던 패키지 판매 웹 페이지
# 페이지 B: 새롭게 생성한 패키지 판매 웹 페이지

# 두 페이지는 스타트업A 패키지 판매 사이트에 접속하는 유저에게 랜덤으로 노출되었고, 테스트 결과는 다음과 같습니다.
# 페이지 A : {노출 수 : 1000, 구매 수 : 80}
# 페이지 B : {노출 수 : 200, 구매 수 : 22}

# 문제9)
# 결과를 바탕으로 패키지 기획자는 페이지 B의 효과에 대해 어떤 결정을 해야 할지 서술해 주세요.
# Hint) https://abtestguide.com/calc/  를 사용하여 해석해보세요

# test 결과 : https://abtestguide.com/calc/?ua=1000&ub=200&ca=80&cb=22&tail=2 
# 관찰된 구매율 차이( 37.50% )는 유의미한 승자를 결정하기에는 충분하지 않습니다. 
# 노출 수 대비 구매 수의 비율이 페이지 B가 더 높기는 하지만, SRM 문제가 발생하여 우연성을 배제할 수 없습니다.
# 이 결과로는 구매율의 차이가 유의미 하지 않기 때문에 이를 추가적으로 확인하기 위하여 더욱 많은 양의 데이터가 필요합니.
# 두 페이지를 더욱 정확히 비교하기 위해서는 노출 수를 최대한 동일하게 맞추는 방법이 있습니다


# 문제 10)
# A/B 테스트의 결과가 통계적으로 유의하나 효과의 차이 자체는 매우 작은 경우, 어떤 의사결정을 할 수 있을지 사례를 통해 설명해 주세요.
# Hint) A/B 관련 자료를 찾아보세요(추천 문헌 [요즘IT](https://yozm.wishket.com/))

# 만일 문제 9)의 예시에서 테스트 결과를 다음과 같이 가정하자
# 페이지 A : {노출 수 : 10050, 구매 수 : 3300}
# 페이지 B : {노출 수 : 9999, 구매 수 : 3300}
# 결과 : https://abtestguide.com/calc/?ua=10050&ub=9999&ca=3300&cb=3500&tail=2

# 위와 같은 결과가 나올 시 페이지 B의 구매율( 35.00% )은 페이지 A의 구매율( 32.84% ) 보다 상대적으로 6.60% 더 높습니다. 
# 이 결과는 우연이 아니라 변경 사항의 결과라고 95 % 확신 할 수 있습니다.
# 다만, 페이지 변경을 통하여 얻을 것이라 예상되는 비용 대비 소모된 비용을 비교하여 필요성을 판단하고 새로운 페이지를 수정하여
# 더욱 좋은 결과를 예상할 수 있는 페이지로 만드는 것이 좋아보인다.
# 실제로 페이지를 변경하였을 때 사용되는 리소스, 상품 판매율, 사용자 만족도, 기업의 영향 등 
# 다양한 부분에서의 장단점 및 효율성을 확인하고 비교하여 더욱 기대효과 및 가치가 큰 쪽으로 결정을 하는 것이 타당해 보입니다.


#%%
#회고
# 어제보다는 더욱 빠르게 프로젝트를 완성할 수 있었던 것 같다.
# 내용적인 문제도 있을 테지만, 문제를 빠르게 이해하고 어떠한 방법을 사용하여 해결할 지에 관한 방법론이 머리속에 잘 정리된 것 같다.
# 아직 많이 부족하지만, 하나하나 학습해 나가며 더욱 성장하고 발전하는 내가 되겠습니다.
