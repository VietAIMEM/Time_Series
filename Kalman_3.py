import pandas as pd
import numpy as np
from pykalman import KalmanFilter

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_3_train = df[df['Publication_Day'] == 'Wednesday']
df_3_test = df_test[df_test['Publication_Day'] == 'Wednesday']

time_series = df_3_train['Listening_Time_minutes'].dropna().values

# Khởi tạo bộ lọc Kalman với xu hướng tuyến tính
kf = KalmanFilter(
    initial_state_mean=[time_series[0], 0],  # [giá trị ban đầu, velocity ban đầu]
    initial_state_covariance=np.eye(2) * 1.0,  # Phương sai ban đầu
    observation_covariance=1.0,  # Phương sai quan sát
    transition_covariance=np.eye(2) * 0.1,  # Phương sai chuyển trạng thái
    transition_matrices=[[1, 1], [0, 1]],  # Ma trận chuyển trạng thái: [x_t+1 = x_t + v_t, v_t+1 = v_t]
    observation_matrices=[[1, 0]]  # Chỉ quan sát giá trị, không quan sát velocity
)

state_means, state_covariances = kf.filter(time_series)

if not df_3_test.empty:
    n_test = len(df_3_test)
    last_state_mean = state_means[-1] 
    predictions = np.ones(n_test) * last_state_mean[0] 
else:
    predictions = []

result = df_3_test[['id']].copy()
result['Predicted_Listening_Time_minutes'] = predictions if len(predictions) > 0 else np.nan

print("Kết quả từ mô hình Kalman với xu hướng tuyến tính:")
print(result)
