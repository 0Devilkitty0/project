#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # acf plot 및 pacf plot을 그리기 위한 라이브러리
from statsmodels.tsa.arima_model import ARIMA # ARIMA 모델
import pmdarima as pm # Auto ARIMA 모델
# %%
ap = pd.read_csv("D:/Users/tonyn/Desktop/da_sci_4th/시계열/AirPassengers.csv")
#%%
ap.drop('Month', axis = 1, inplace = True)
# %%
ap
# %%
plt.plot(ap)
plt.show()
# %%
# np.log를 통해서 log transformation
ap_transformed = # [[YOUR CODE]]