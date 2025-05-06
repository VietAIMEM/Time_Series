import pandas as pd
import numpy as np
from pykalman import KalmanFilter

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_monday = df[df['Publication_Day'] == 'Monday']
df_test_monday = df_test[df_test['Publication_Day'] == 'Monday']

time_series = df_monday['Listening_Time_minutes'].dropna().values

# Khởi tạo bộ lọc Kalman
kf = KalmanFilter(
    initial_state_mean=time_series[0],
    initial_state_covariance=1.0,      
    observation_covariance=1.0,      
    transition_covariance=0.1,         
    transition_matrices=1.0,         
    observation_matrices=1.0    
)

state_means, state_covariances = kf.filter(time_series)

if not df_test_monday.empty:
    n_test = len(df_test_monday)
    last_state_mean = state_means[-1]
    predictions = np.ones(n_test) * last_state_mean  # Dự đoán giá trị cố định
else:
    predictions = []

result = df_test_monday[['id']].copy()
result['Predicted_Listening_Time_minutes'] = predictions if len(predictions) > 0 else np.nan

print(result)
