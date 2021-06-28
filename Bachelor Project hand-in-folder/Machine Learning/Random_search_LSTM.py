import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
import time
import keras
import seaborn as sns
import random as rn
from tqdm import tqdm
import shutil
import random as rn


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Own functions
from data_set_builder import data_set_builder
from create_lstm_train_test import transform_df_to_LSTM_train_test



def build_LSTM_model(X_train, X_val, y_train, y_val):
    global loss_func
    global opt
    global drop_out_rate
    global start_time
    global list_of_run_times
    global model_iteration_id
    global add_dense_layer
    global learning_rate
    global epochs
    global init_epochs 
    global hidden_units
    global input_units
    global num_hidden_layer

    # The model builds, then trains. To append the runtime, we must assume that the second iteration of random search, is the time of the first models completion.
    # This is an estimate, but a fairly close one.

    model_iteration_id += 1

    # Input layer
    model = Sequential()        

    print("num_hidden_layer: ", num_hidden_layer)
    
    if num_hidden_layer == 0:
        print("No hidden layers")
        return_sequences_bool = False
    else:
        print(f"{num_hidden_layer} layers to model")
        return_sequences_bool = True
    
    model.add(LSTM(units= input_units, 
                   input_shape=(window_size, feature_space, ), return_sequences=return_sequences_bool,
                   name = "input_layer_LSTM_id{}".format(model_iteration_id)))
    
    # Hidden layers
    
    for i in range(num_hidden_layer):
        if num_hidden_layer-1 == i:
            return_sequences_bool = False
        else:
            return_sequences_bool = True
        model.add(keras.layers.BatchNormalization(name = "{}_h_layer_BatchNorm_id{}".format(i+1,model_iteration_id))) # removed with the hypothesis that LSTM already normalize with sigmoid functions.
        model.add(LSTM(units= hidden_units, 
                       input_shape=(window_size, feature_space, ), return_sequences=return_sequences_bool, 
                       dropout = drop_out_rate, name = "{}_h_layer_LSTM_id{}"   .format(i+1,model_iteration_id)))

    if add_dense_layer == 2:
        model.add(keras.layers.BatchNormalization(name = "1_dense_layer_BatchNorm_id{}".format(model_iteration_id)))
        model.add(Dense(100, name = "{}_dense_layer_id{}".format(1,model_iteration_id) , activation='elu'))
        model.add(Dense(50, name = "{}_dense_layer_id{}".format(2,model_iteration_id), activation='elu')) 

    if add_dense_layer == 1:
        model.add(keras.layers.BatchNormalization(name = "1_dense_layer_BatchNorm_id{}".format(model_iteration_id)))
        model.add(Dense(100, name = "{}_dense_layer_id{}".format(1,model_iteration_id) , activation='elu'))
    
    # Last layer
    model.add(Dense(1, name = "Output_layer_id{}".format(model_iteration_id))) 
    
    opt.learning_rate = learning_rate
    
    model.compile(optimizer= opt, loss= loss_func, metrics=['MAE','MSE', 'MAPE'])


    find_model_number(model_type)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience , restore_best_weights = True)
    cb_list = [es]
    
    print(model.summary())

    model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=init_epochs, batch_size=batch_size, verbose=1
                , shuffle = True)

    model_history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1
                , shuffle = True  ,callbacks = cb_list)

    epochs_trained = len(model_history.history['loss']) - patience + init_epochs
    print("Number of Epochs trained: ", epochs_trained)


    return model, epochs_trained



def find_model_number(model_type):

    existing_model_numbers = []

    for model_folder in os.listdir(f"Saved_models/{model_type}_models"):
        if "Model_" in model_folder:
            existing_model_numbers.append(int(model_folder[6:]))

    existing_model_numbers.sort()

    for potential_model_num in range(len(existing_model_numbers)+1):
        if potential_model_num in existing_model_numbers:
            print(f"Model_{potential_model_num} already exists")
        else:
            print(f"Model_{potential_model_num} does not exist")
            print(f"Using model_{potential_model_num} as save file.")
            return potential_model_num
    
    return None # should never happen


def usefull_model_info(model, model_type , X_test, y_test, batch_size):
    global opt
    global future_candles
    global window_size
    global epochs
    global drop_out_rate
    global start_time
    global list_of_run_times
    global patience
    
    # Save 10 best models from random search

    experiment_num = find_model_number(model_type)

    save_dir = f"Saved_models\\{model_type}_models\\Model_{experiment_num}\\"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    run_time = list_of_run_times[-1] - list_of_run_times[-2]

    with open(save_dir + 'model_report.txt','w') as file:
        loss,mae,mse,mape = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        file.write("Scores on Test: Loss:{:.10f}  MAE:{:.10f}  MSE:{:.10f}  MAPE:{:.10f} \n".format(loss, mae, mse, mape))
        print("Scores on Test: Loss:{:.10f}  MAE:{:.10f}  MSE:{:.10f}  MAPE:{:.10f} \n".format(loss, mae, mse, mape))
        file.write("Optimzer: {}  Learning_rate: {:.5f}  Dense Layer: {} \n".format(opt.get_config()['name'], 
                                                                opt.get_config()['learning_rate'], add_dense_layer))
        file.write("Future_candles: {}  Window_size: {}  Drop_out: {} \n".format(future_candles, window_size, drop_out_rate))
        file.write("Epochs: {}  Batch_size: {}  Run_time: {} min {} seconds \n".format(epochs, batch_size, round(run_time//60), round(run_time%60)))
        
        file.write("\n")
        model.summary()
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        
    y_pred = [y[0] for y in model.predict(X_test)]
    results_df = pd.DataFrame({'y_test' : y_test, 'y_pred':  y_pred})
    results_folder = save_dir + "/y_test_y_pred.csv"
    results_df.to_csv(results_folder)
        
    # Possible Error:
    # FIX: Remove files from D:\Bachelor_project_models\LSTM_models
    
    model.save(save_dir + "model_{}.h5".format(experiment_num))   
    model = None # This fixes the error of (OSError: Unable to create link (name already exists)) by clearing the memory of model perhaps. Doesn't matter. it works now.
    
    print("model_{}.h5 Saved".format(experiment_num))    



if __name__ == "__main__":
    num_iterations = 20
    start_iteration_time = time.time()
    iteration_times = []
    list_of_run_times = [time.time()]

    for i in tqdm(range(num_iterations)):
        
        model_type = "LSTM"
        experiment_num = find_model_number(model_type)

        loss_func = rn.choice(["MSE"])
        

        model_iteration_id = 0
        init_epochs = 30
        epochs = 1000 # Patience is a parameter
        patience = 2
        batch_size = rn.choice([32,48])
        future_candles = rn.choice([6,12])
        window_size = rn.choice([12,18,24]) 
        drop_out_rate = rn.choice([0.2])
        opt = rn.choice([keras.optimizers.Adamax()]) 
        add_dense_layer = rn.choice([0, 1,2])
        learning_rate = rn.choice([0.003, 0.001]) #, 0.0005, 0.0001])
        
        input_units = rn.choice([128])
        hidden_units = rn.choice([128])
        num_hidden_layer = rn.choice([0, 1 , 2])

            

        model_iteration_id = 0
        init_epochs = 30
        epochs = 1000 # Patience is a parameter
        patience = 4
        batch_size = rn.choice([32,48])
        future_candles = rn.choice([6,12])
        window_size = rn.choice([24]) 
        drop_out_rate = rn.choice([0.1,0.2])
        opt = rn.choice([keras.optimizers.Adamax()]) 
        add_dense_layer = rn.choice([0,1])
        learning_rate = rn.choice([0.003, 0.001]) #, 0.0005, 0.0001])

        input_units = rn.choice([512])
        hidden_units = rn.choice([512])
        num_hidden_layer = 0



        


        #Testing parameters
        # window_size = 24
        # epochs = 100
        # batch_size = 64

        try:
            # To avoid OOM
            del X_train
            del X_val 
            del X_test
            del y_train
            del y_val
            del y_test
        except: 
            pass

        train_df = data_set_builder('EURUSD', [i for i in range(2012,2020)], future_candles)
        test_df  = data_set_builder('EURUSD', [i for i in range(2020,2021)], future_candles)


        # train_df = train_df.loc[:1000,:] 
        # test_df = test_df.loc[:1000,:] 

        X_train, X_val, X_test, y_train, y_val, y_test = transform_df_to_LSTM_train_test(train_df, test_df, future_candles, 
                                                                                        window_size) 
        feature_space = X_train.shape[2]
        

        print("")
        print("experiment_num : ", experiment_num)
        print("optimizer      : ", opt.get_config()['name'])
        print("loss_func      : ", loss_func)
        print("epochs         : ", epochs)
        print("patience       : ", patience )
        print("batch_size     : ", batch_size)
        print("window_size    : ", window_size)
        print("drop_out_rate  : ", drop_out_rate)
        print("future_candles : ", future_candles)
        print("learning_rate  : ", learning_rate)
        print("add_dense_layer: ", add_dense_layer) 
        print("")
        
        start_time = time.time()

        model, epochs = build_LSTM_model(X_train, X_val, y_train, y_val )

        # Adding the last iteration of run times
        list_of_run_times.append(time.time())
        
        usefull_model_info(model, model_type , X_test, y_test, batch_size)
        
            
        if i == 0:
            iteration_times.append(time.time() - start_iteration_time)
        else:
            iteration_times.append(time.time() - start_iteration_time - sum(iteration_times[:i]))

        eta_run_times = (sum(iteration_times)/(i+1))*num_iterations - (sum(iteration_times))
        print("\n\nIteration time: {:.0f}h {:.0f}m {:.0f}s  \nEstimated Finished time: {:.0f}h {:.0f}m {:.0f}s\n\n".format(
                                                            iteration_times[i]//3600 ,  iteration_times[i]//60%60 , iteration_times[i]%60 ,
                                                            eta_run_times//3600, (eta_run_times//60)%60, eta_run_times%60))
        time.sleep(1) # for viewing purposes


            
            
            
