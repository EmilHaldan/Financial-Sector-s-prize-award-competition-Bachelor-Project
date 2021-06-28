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
    
    # Creating the feature 

    df['U'] = df['Close'] - df['Close'].shift(periods = time_frame)
    df['U'] = [x if x>= 0 else 0 for x in df['U']]

    df['D'] = df['Close'] - df['Close'].shift(periods = time_frame)
    df['D'] = [abs(x) if x<= 0 else 0 for x in df['D']]

    df['MA_U'] = (df['U']*1 + df['U'].rolling(window=time_frame , min_periods=0).mean()*(time_frame-1) ) / time_frame
    df['MA_D'] = (df['D']*1 + df['D'].rolling(window=time_frame , min_periods=0).mean()*(time_frame-1) ) / time_frame

    df['MA_U'] = [x if x>0 else 0.000001 for x in df['MA_U']]
    df['MA_D'] = [x if x>0 else 0.000001 for x in df['MA_D']]

    df[feature_name] = [1 - ( 1 / (1+(df.loc[i,'MA_U']/df.loc[i,'MA_D']))) for i in range(len(df))]
    
    # Saving the column dataframe feature
    df[feature_name].to_csv(target + filename)


if __name__ == "__main__":
    years = list(range(2012,2021))
    currency_pairs = ['EURUSD','EURGBP','GBPUSD']
    time_frames = [6,9,12,24,36,48,60]

    for year in years: 
        for fx_pair in currency_pairs:
            for time_frame in time_frames:
                print("Year: {}  FX-pair: {}  time_frame: {}".format(year,fx_pair,time_frame))
                create_feature_for_entire_df(fx_pair, year, time_frame)

            
        