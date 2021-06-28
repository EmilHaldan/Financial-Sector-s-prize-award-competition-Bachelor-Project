
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 
import time
from sklearn.model_selection import train_test_split
# Own function
from data_set_builder import data_set_builder


def transform_df_to_LSTM_train_test(train_df, test_df, future_candles, window_size, return_dates=False):
    """
    train_df: A list of dataframes, or a dataframe with Datetime, OHLC, features, and target
    test_df : A dataframe with Datetime, OHLC, features, and target
    future_candles example: str(24_candle_target) or int(24)
    window_size : a variable dictating how large the window for a sample should be
    """
    if type(future_candles) == int:
        future_candles = "{}_candle_target".format(future_candles)
    
    if train_df.shape[1] != test_df.shape[1]:
        print("WARNING:")
        print("Train and Test dataframe are not same column shapes.")
    
    # Marking non continous states, by defining their respective entry as False for ['Continous']
    
    train_df['Continous'] = True
    test_df['Continous']  = True
    
    train_df['Date_time'] = pd.to_datetime(train_df['Date_time'], infer_datetime_format=True) 
    test_df['Date_time'] = pd.to_datetime(test_df['Date_time'], infer_datetime_format=True) 
    
    for idx,date_info in tqdm(enumerate(train_df['Date_time']),total=len(train_df), desc="Train, creating Continous"):
        if idx == 0:
            continue # skip the first entry of the dataframe, as it cant be compared with the previous
        cur_datetime  = date_info
        prev_datetime = train_df.loc[idx-1,'Date_time']
        time_left = (cur_datetime - prev_datetime).total_seconds() 
    
        if time_left != 600:  # 600 seconds
            train_df.loc[idx,'Continous'] = False
            
    for idx,date_info in tqdm(enumerate(test_df['Date_time']),total=len(test_df), desc="Test, creating Continous"):
        if idx == 0:
            continue # skip the first entry of the dataframe, as it cant be compared with the previous
        cur_datetime  = date_info
        prev_datetime = test_df.loc[idx-1,'Date_time']
        time_left = (cur_datetime - prev_datetime).total_seconds() 
    
        if time_left > 600:  # 600 seconds
            test_df.loc[idx,'Continous'] = False
    
    continous_train = train_df['Continous']
    continous_test = test_df['Continous']
    
    train_df.drop(columns=['Continous'], inplace =True)
    test_df.drop(columns=['Continous'], inplace =True)
    
    X_train = train_df[[col for col in train_df if col not in [future_candles,"Date_time"]]].values 
    y_train = train_df[future_candles].values 
    
    X_test = test_df[[col for col in train_df if col not in [future_candles,"Date_time"]]].values 
    y_test = test_df[future_candles].values 

    if return_dates:
        test_dates_list = test_df[[col for col in train_df if col in ["Date_time"]]].values
    
    
    # Add samples of time series in here, along with their respective y_train/test
    LSTM_X_train_list = []
    LSTM_X_test_list =  []
    LSTM_y_train_list = []
    LSTM_y_test_list =  []

    if return_dates:
        LSTM_test_dates_list = []
    
    
    # Prepping train set data
    for idx,row in tqdm(enumerate(X_train),total=len(X_train), desc="Train, creating 3D samples"):       
        if idx < window_size:
            continue
        complete = True
        
        for i in range(idx-window_size+1, idx +1 ): # If IDX == 3 , and windows size is 2, then it should contain idx [2,3]
            if not continous_train[i]: 
                # If it is incomplete at any time in the window frame, don't append it.
                complete = False
        if complete:
            # Append sample, along with target.
            sample = X_train[idx-window_size+1 : idx+1, :]
            LSTM_X_train_list.append(sample)
            LSTM_y_train_list.append(y_train[idx])
        
    LSTM_X_train = np.array(LSTM_X_train_list)    
    LSTM_y_train = np.array(LSTM_y_train_list)  
    
    # Prepping test set data 
    for idx,row in tqdm(enumerate(X_test),total=len(X_test), desc="Test, creating 3D samples"):
        if idx < window_size:
            continue
            
        complete = True
        for i in range(idx-window_size+1, idx+1):
            if not continous_test[i]: 
                complete = False
        if complete:
            sample = X_test[idx-window_size+1 : idx+1, :]
            LSTM_X_test_list.append(sample)
            LSTM_y_test_list.append(y_test[idx])
            if return_dates:
                LSTM_test_dates_list.append(np.datetime_as_string(test_dates_list[idx][0], unit='m'))
            
    LSTM_X_test = np.array(LSTM_X_test_list) 
    LSTM_y_test = np.array(LSTM_y_test_list)    

    # Shuffling X_train, y_train together, and creating validation set.
    LSTM_X_train, LSTM_X_val, LSTM_y_train, LSTM_y_val = train_test_split(LSTM_X_train, LSTM_y_train, test_size=0.20, random_state=42069)

    print("")
    print("LSTM_y_train.shape: ", LSTM_y_train.shape)
    print("LSTM_X_train.shape: ", LSTM_X_train.shape)
    print("")
    print("LSTM_y_val.shape: ", LSTM_y_val.shape)
    print("LSTM_X_val.shape: ", LSTM_X_val.shape)
    print("")
    print("LSTM_y_test.shape: ", LSTM_y_test.shape)
    print("LSTM_X_test.shape: ", LSTM_X_test.shape) 
    
    if return_dates:
        return LSTM_X_train, LSTM_X_val, LSTM_X_test, LSTM_y_train, LSTM_y_val, LSTM_y_test, LSTM_test_dates_list
    else:
        return LSTM_X_train, LSTM_X_val, LSTM_X_test, LSTM_y_train, LSTM_y_val, LSTM_y_test



if __name__ == "__main__":


    future_candles = 6
    window_size = 24

    train_df = data_set_builder('EURUSD', [i for i in range(2019,2020)], future_candles)
    test_df = data_set_builder('EURUSD', [i for i in range(2020,2021)],  future_candles)

    X_train, X_val, X_test, y_train, y_val, y_test, test_dates_list = transform_df_to_LSTM_train_test(train_df, test_df, future_candles, window_size, return_dates= True)

    print("\nX_train: ")
    print("")
    print(X_train)
    print("")
    print("")
    print("y_train: ")
    print("")
    print(y_train)
    print("")
    print("")

    print("X_val: ")
    print("")
    print(X_val)
    print("")
    print("")
    print("y_val: ")
    print("")
    print(y_val)
    print("")
    print("")
    
    print("X_test: ")
    print("")
    print(X_test)
    print("")
    print("")
    print("y_test: ")
    print("")
    print(y_test)
    print("")
    print("")

    print("test_dates_list[:40]:")
    print(test_dates_list)

