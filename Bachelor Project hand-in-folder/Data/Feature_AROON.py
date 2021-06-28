import pandas as pd 
import time 
import pickle 
import os
from tqdm import tqdm


"""
General description: 
This is a template for a feature. The python script should be run, and placed inside a folder within the Feature_Bank.

"""

def create_feature_for_entire_df(fx_pair, year, time_frame):
    """
    This function takes a clean OHLC dataset, and returns a csv file of date_time and feature value.
    fx_pair: 'EURUSD' , 'GBPUSD', 'EURCHF' , etc.
    year   : integer describing year
    time_frame: decides the window used in the pandas functions. A variable to decide which indicator timeframe might be best.

    """
    feature_name = str(__file__)[8:-3] + "_" + str(time_frame*10) + "min"    
    print("feature_name: ", feature_name)

    # Opening normalized dataframe for the currency pair
    source_file = "..\\Cleaned_OHLC_FOREX_Data_10_min\\{}\\{}_{}_10min.csv".format(fx_pair,fx_pair,year) 
    target = "{}\\{}\\".format(feature_name,fx_pair) 
    if not os.path.exists(target):
        os.makedirs(target)

    filename = "{}_{}_{}.csv".format(feature_name,fx_pair,year)
    df = pd.read_csv(source_file, )

    # Trying to speed up the process... (RESULTS: dropping 60% of the data frame, including the datetime values peed things up by 10x)
    df.drop(columns = ['Date_time','Open', 'Close'] , inplace = True)
    
    # Creating the feature AROON

    candles_since_high = 0
    candles_since_low  = 0
    cur_high_idx = 0
    cur_low_idx = 0

    #Note to self. Accessing dataframes takes ages.... only do it if needed.
    # Removing 4 df.loc[idx,col] calls to defined values speed up the process incredibly much. From 200 iterations pr. sec, to 5200 iterations pr. sex
    list_since_high = []
    list_since_low  = []

    high_list = []
    low_list = []

    for idx in tqdm(range(len(df))):
        start_idx = idx - time_frame
        if start_idx <= 0: start_idx = 0
        end_idx = idx
        
        high_list.append(df.loc[idx , 'High'])
        low_list.append(df.loc[idx , 'Low'])
        
        if len(high_list) > time_frame +1: 
            high_list = high_list[1:]
        if len(low_list) > time_frame +1:
            low_list = low_list[1:]

        High_index = len(high_list)-1 - high_list[::-1].index(max(high_list))  + (idx-len(high_list) +1)
        Low_index =  len(low_list)-1  - low_list[::-1].index(min(low_list))    + (idx-len(low_list)  +1)
        
        if cur_high_idx >= High_index:
            candles_since_high += 1
        else:
            #print("idx: {}  High_index: {}".format(idx, Low_index))
            candles_since_high = idx - High_index
            cur_high_idx = High_index

        if cur_low_idx >= Low_index:
            candles_since_low += 1
        else:
            #print("idx: {}  Low_index: {}".format(idx, Low_index))
            candles_since_low = idx - Low_index 
            cur_low_idx = Low_index

        list_since_high.append(candles_since_high)
        list_since_low.append(candles_since_low)
            
    df['since_High'] =   list_since_high  
    df['since_Low']  =   list_since_low  
        
    df[feature_name + "_up"] =   ((time_frame - df['since_High'] ) / time_frame )
    df[feature_name + "_down"] = ((time_frame - df['since_Low'] ) / time_frame )

    feature_df = df[[feature_name + "_up",feature_name + "_down"]]
    
    # Saving the column dataframe feature
    feature_df.to_csv(target + filename)


if __name__ == "__main__":
    years = list(range(2012,2021))
    currency_pairs = ['EURUSD','EURGBP','GBPUSD']
    time_frames = [6,9,12,24,36,48,60]

    start_time = time.time()

    dumb_time_counter = len(years) * len(currency_pairs) * len(time_frames)

    for time_frame in time_frames:
        for year in years: 
            for fx_pair in currency_pairs:
                iter_time = time.time()
                feature_name = str(__file__)[8:-3] + "_" + str(time_frame*10) + "min"   
                if not os.path.isfile("{}\\{}\\".format(feature_name,fx_pair) +  "{}_{}_{}.csv".format(feature_name,fx_pair,year)):
                    print("Year: {}  FX-pair: {}  time_frame: {}".format(year,fx_pair,time_frame))
                    create_feature_for_entire_df(fx_pair, year, time_frame)
                    print("Time for iteration: " , time.time()- iter_time)
                    print("Total time: " , time.time()- start_time)
                    print("Total remaining time: ", round(dumb_time_counter*(time.time()- iter_time)), " seconds.")
                    print("")
                else:
                    print("Skipping iteration: ")
                    print("Year: {}  FX-pair: {}  time_frame: {}".format(year,fx_pair,time_frame))
                    print("")
                dumb_time_counter-= 1

            
        