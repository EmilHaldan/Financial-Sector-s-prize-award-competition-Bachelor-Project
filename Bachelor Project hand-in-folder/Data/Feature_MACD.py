import pandas as pd 
import time 
import pickle 
import numpy as np
import os
from tabulate import tabulate

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
    

    # Creating the feature (MACD, MACD_signal, MACD_crossover)
    og_time_frame = time_frame

    slow_period_proportion = 26/26
    fast_period_proportion= 12/26
    signal_period_proportion= 9/26

    df['EMA_slow'] =  df['Close'].rolling(window=int(time_frame * slow_period_proportion) , min_periods=0).mean()
    df['EMA_fast'] =  df['Close'].rolling(window=int(time_frame * fast_period_proportion) , min_periods=0).mean()

    emas = ['EMA_slow','EMA_fast']
    close_list = list(df['Close'])

    # Calculate the MACD line first
    for ema_type in emas:
        time_frame = og_time_frame
        if ema_type == "EMA_slow":
            time_frame = int(time_frame * slow_period_proportion)
        elif ema_type == "EMA_fast":
            time_frame = int(time_frame * fast_period_proportion)
        else:
            pass
        alpha = 2/(time_frame+1)
        for idx, ema_val in enumerate(df[ema_type]):
            if idx < time_frame:
                continue 
            else:
                prev_ema = df.loc[idx-1,ema_type]
                # Formula for EMA
                df.loc[idx,ema_type] = prev_ema + alpha * (close_list[idx] - prev_ema)

    # Creating MACD
    df[feature_name] = df['EMA_fast'] - df['EMA_slow']      


    # Calculate signal line 
    time_frame = og_time_frame
    df[feature_name+'_signal'] = df[feature_name].rolling(window=int(time_frame * signal_period_proportion) , min_periods=0).mean()
    macd_list = list(df[feature_name])

    time_frame = int(og_time_frame*signal_period_proportion)
    alpha = 2/(time_frame+1)
    for idx, ema_val in enumerate(df[ema_type]):
        if idx < time_frame:
            continue 
        else:
            prev_ema = df.loc[idx-1,feature_name+'_signal']
            df.loc[idx,feature_name+'_signal'] = prev_ema + alpha * (macd_list[idx] - prev_ema)

    time_frame = og_time_frame
        
    

    # According to traders. The interesting part happens when the MACD crosses the MACD_signal line.
    # Additionally, this must happen when both lines are verry possitive or verry negative (with respect to the deviation)
    # As we're interested in when MACD and MACD_signal intersect
    # we will keep track of the "cross overs", and create scalar values of that, corresponding to the MACD.
    # As i wish to incooperate this "logical thinking" into data, a trinary feature of this will be created.


    signal_list = df[feature_name+'_signal']
    macd_list = df[feature_name] # Not needed, but makes code easier to read.
    macd_crossover = []

    # MACD:True or Signal:False (attribute used to indicate which line is on top, as intersections do not 
    # occour as the cross overs happend between candles)
    # The cross overs are additonally only interesting at certain points, hence the need for taking the standard deviation.
    # Let the values in the MacD cross over be a float, IFF it crosses over. The float is the product of MACD * std
    macd_more_than_zero = True 
    macd_more_than_zero_past = False 

    macd_minus_signal_list = [macd_list[i] - signal_list[i] for i in range(len(df))] 
    macd_minus_signal_std = np.std(macd_minus_signal_list) # Measure to determine significance for the cross over.

    for idx in range(len(df)):
        signal = signal_list[idx]
        macd = macd_list[idx]
        
        cross_over_val = macd_minus_signal_list[idx]
        if macd > 0:
            macd_more_than_zero = True
        else:
            macd_more_than_zero = False
            
        if macd_more_than_zero != macd_more_than_zero_past: # It crossed the line!
            if abs(cross_over_val) > macd_minus_signal_std:
                cross_over_val *= 5 # a magnitude larger value to show significance.
        macd_crossover.append(cross_over_val)
        
        # Saving macd_on_top for previous value.
        macd_more_than_zero_past = macd_more_than_zero
        
    df[feature_name+'_crossover'] = macd_crossover
        
    # Saving the column dataframe feature
    #print(tabulate(df[[feature_name , feature_name+'_signal' , feature_name+'_crossover']], headers='keys', tablefmt='psql'))
    df[[feature_name , feature_name+'_signal' , feature_name+'_crossover']].to_csv(target + filename )


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
    

            
        