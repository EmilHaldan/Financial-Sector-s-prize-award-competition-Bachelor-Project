import pandas as pd 
import time 
import pickle 
import os

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
    df = pd.read_csv(source_file)
    
    # Creating the feature (Example)
    time_frame = 12
    alpha = 2/(time_frame+1)

    df[feature_name] = df['Close'].rolling(window=time_frame , min_periods=0).mean() 

    close_list = list(df['Close'])

    for idx, ema_val in enumerate(df[feature_name]):
        if idx < time_frame:
            continue 
        else:
            prev_ema = df.loc[idx-1,feature_name]
            df.loc[idx,feature_name] = prev_ema + alpha * (close_list[idx] - prev_ema)
    
    # Reducing the Close price to use the difference
    df[feature_name] = df[feature_name] - df['Close']

    # Saving the column dataframe feature
    df[feature_name].to_csv(target + filename)


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
                

            
        