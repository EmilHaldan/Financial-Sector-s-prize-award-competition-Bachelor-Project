import pandas as pd 
import time 
import pickle 
import os

"""
General description: 
The ROC index is a measure of volatility. If it's low, the waters are calm, but if it's high, 
there's some shit going on.

The way it is measured by investors suggests that the value on depends on the n-prior value.
ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100

That means the formula is sensitive to "Spikes", in terms of being dependent on them for the better and worse.
Instead, is should be the following: 

ROC_close = (Mean(Close).rolling(n) - Close / (Mean(Close).rolling(n))) * 100
ROC_high =  (Max(High).rolling(n) - High    / (Max(High).rolling(n))) * 100
ROC_low =   (Low(Low).rolling(n) - Low      / (Low(Low).rolling(n))) * 100
ROC = (ROC_close + ROC_high + ROC_low ) /3

The suggested methods will weigh in a large spike, 

"""
# The featurename is a unique identifier to be used as the column name, folder name, and file names


def create_feature_for_entire_df(fx_pair, year, time_frame, candle_data_type):
    """
    This function takes a clean OHLC dataset, and returns a csv file of date_time and feature value.
    fx_pair: 'EURUSD' , 'GBPUSD', 'EURCHF' , etc.
    year   : integer describing year
    time_frame: decides the window used in the pandas functions. A variable to decide which indicator timeframe might be best.

    """
    feature_name = str(__file__)[8:-3] + "_" + candle_data_type + "_" + str(time_frame*10) + "min"    
    print("feature_name: ", feature_name)

    # Opening normalized dataframe for the currency pair
    source_file = "..\\Cleaned_OHLC_FOREX_Data_10_min\\{}\\{}_{}_10min.csv".format(fx_pair,fx_pair,year) 
    target = "{}\\{}\\".format(feature_name,fx_pair) 
    if not os.path.exists(target):
        os.makedirs(target)

    filename = "{}_{}_{}.csv".format(feature_name,fx_pair,year)
    df = pd.read_csv(source_file)


    # Write formula here 
    df[feature_name] = (df[candle_data_type] - df[candle_data_type].shift(periods = time_frame)) / df[candle_data_type].shift(periods = time_frame) *100

    feature_df = df[feature_name]

   # Saving the column dataframe feature
    feature_df.to_csv(target + filename)


if __name__ == "__main__":
    years = list(range(2012,2021))
    currency_pairs = ['EURUSD','EURGBP','GBPUSD']
    time_frames = [6,9,12,24,36,48,60]
    candle_data_type = 'Close'

    for year in years: 
        for fx_pair in currency_pairs:
            for time_frame in time_frames:
                print("Year: {}  FX-pair: {}  time_frame: {}".format(year,fx_pair,time_frame))
                create_feature_for_entire_df(fx_pair, year, time_frame, candle_data_type)
