import pandas as pd
import os
from tqdm import tqdm


def data_set_builder(fx_pair, years, future_candles):
    """
    This file should be placed in the directory "Bachelor Project/Machine Learning"
    fx_pair: a given currency pair: "EURUSD", "GBPCHF", etc...
    years: if int, a single year, else, a list of years.
    features: a list of features, as described in its file structure. E.G "ROC_Low_90min"
    future_candles: an int describing future candles, used for mapping target.
    """

    features = [folder for folder in os.listdir("../Data/Feature_Bank") if "." not in folder]

    if type(years) == int: years = [years]
    df = None
    for year in tqdm(years):
        temp_df = pd.read_csv("../Data/Cleaned_OHLC_FOREX_Data_10_min/{}/{}_{}_10min.csv".format(fx_pair, fx_pair,  year)).drop(columns =['Unnamed: 0'])     
        
        target_df = pd.read_csv("../Data/Target_Bank/Regression/{}/{}_{}_{}_candle_future.csv".format(fx_pair,fx_pair,year,future_candles)).drop(columns =['Unnamed: 0'])
        temp_df = pd.concat([temp_df, target_df], axis=1)
        
        for feature in features:
            feature_df = pd.read_csv("../Data/Feature_Bank/{}/{}/{}_{}_{}_standardized.csv".format(feature, fx_pair, feature, fx_pair, year)).drop(columns =['Unnamed: 0'])
            #feature_df = pd.read_csv("../Data/Feature_Bank/{}/{}/{}_{}_{}.csv".format(feature, fx_pair, feature, fx_pair, year)).drop(columns =['Unnamed: 0'])
            temp_df = pd.concat([temp_df, feature_df], axis=1)
        
            
        if type(df) == None:
            df = temp_df.copy()
        else: 
            df = pd.concat([df, temp_df], axis=0)
    
    drop_the_index = None
    df_nans = df[df.isna().any(axis=1)]
    if len(df_nans) >= 1:
        print("")
        print("Amount of rows with Nans:", len(df_nans))
        drop_the_index = df_nans.index.values.tolist()
        print("Dropping index in {} year-span {}-{} : {}".format(fx_pair,years[0], years[-1], drop_the_index))
        df.drop(drop_the_index, inplace = True)
        
    
    df.drop(columns = ['Open','High','Low','Close'], inplace = True)
    df.reset_index(inplace = True, drop = True )
    return df



if __name__ == "__main__":

    clean_df = data_set_builder('EURUSD', [i for i in range(2019,2020)], 6)
    
    print("\n\n\n  Printing first row of dataframe:\n\n")
    print(clean_df.loc[0,:])
    print("")
    print(clean_df.describe())