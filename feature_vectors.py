import yfinance as yf
import numpy as np
from tqdm import tqdm 

def calculate_log_momentum_T(data, t):
    data = np.array(data)
    log_momentum = np.log(data[50:]) - np.log(data[50-t:-t])
    return log_momentum

def calculate_risk_t(data, t):
    all_data = np.array(data).squeeze()
    print(all_data.shape)
    r_data = np.diff(np.log(all_data), prepend=all_data[0])  #ri = log(P i) - log(P i-1)
    
    vector_length = len(r_data[50:])
    risk_t = np.zeros(vector_length)
    day_index = 50+1 
    for i in tqdm(range(vector_length)):
        t_days_r_data = r_data[day_index+i-t : day_index+i] 
        mean = np.sum(t_days_r_data)/t
        risk_t[i] = np.sqrt(np.sum(np.square(t_days_r_data - mean))/t)
    return risk_t

def generate_data_matrix(data):
    matrix_column_list = []
    for t in [5,10,20,30,40,50]:
        momentum = calculate_log_momentum_T(data,t)
        risk = calculate_risk_t(data, t)

        matrix_column_list.append(momentum)
        matrix_column_list.append(risk)
    matrix = np.column_stack(matrix_column_list)
    return matrix

# ticker = yf.Ticker("^GSPC")
# history = ticker.history(period="10y")["Close"].to_numpy()
# # open  high  low  close  volume  dividends  stock-splits
# data = generate_data_matrix(history)
# np.save('data_matrix.npy', data)
# print(data.shape)





