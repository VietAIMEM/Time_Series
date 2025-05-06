import pandas as pd
import numpy as np
from pykalman import KalmanFilter

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_3_train = df[df['Publication_Day'] == 'Monday']
df_3_test = df_test[df_test['Publication_Day'] == 'Monday']

df_3_train = df_3_train.dropna(subset=['Listening_Time_minutes', 'Episode_Length_minutes'])
time_series = df_3_train['Listening_Time_minutes'].values
episode_length = df_3_train['Episode_Length_minutes'].values

observation_matrices = np.array([[[1, length]] for length in episode_length])

# Khởi tạo bộ lọc Kalman
kf = KalmanFilter(
    initial_state_mean=[time_series[0], 0],  # [giá trị ban đầu, trọng số của Episode_Length]
    initial_state_covariance=np.eye(2) * 1.0,  # Phương sai ban đầu
    observation_covariance=1.0,  # Phương sai quan sát
    transition_covariance=np.eye(2) * 0.1,  # Phương sai chuyển trạng thái
    transition_matrices=np.eye(2),  # Trạng thái không đổi (trừ nhiễu)
    observation_matrices=observation_matrices  # Ma trận quan sát động
)

state_means, state_covariances = kf.filter(time_series)

if not df_3_test.empty:
    episode_length_test = df_3_test['Episode_Length_minutes'].fillna(episode_length.mean()).values
    last_state_mean = state_means[-1] 
    predictions = []
    for length in episode_length_test:
        pred = last_state_mean[0] + last_state_mean[1] * length 
        predictions.append(pred)
else:
    predictions = []

result = df_3_test[['id']].copy()
result['Predicted_Listening_Time_minutes'] = predictions if len(predictions) > 0 else np.nan

print("Kết quả từ mô hình Kalman với đặc trưng bổ sung:")
print(result)
