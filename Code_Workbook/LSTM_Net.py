# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:16:44 2020

@author: mkhodaverdi
"""
################################# %% initial step

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from pylab import rcParams
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from bayes_opt import BayesianOptimization

    
# read data
covid_7 = pd.read_csv('C:/Users/khodaverdi/Desktop/R_code/WV_COVID_data_tot.csv') 
covid_14 = pd.read_csv('C:/Users/khodaverdi/Desktop/R_code/WV_COVID_data_tot1.csv')

popu = pd.read_csv( 'C:/Users/khodaverdi/Desktop/data/IncidRtLam.csv')
holiday = pd.read_csv('C:/Users/khodaverdi/Desktop/data/Holidays.csv')

# set variables
n_future = 7    # Number of days ahead to predict
n_past = 7      # Number of past days used to predict the future

sum_report =[]
county_names = sorted(list( covid_7.county.unique() )) 
pop = popu[['county_name', 'population']]



##################################%% second step:preping data 

# merge and scale data
def data_prep_new(covid_7, covid_14, popu, holiday):
    
    # make bumped dataset
    covid_7['date'] = pd.to_datetime(covid_7.date)
    covid_14['date'] = pd.to_datetime(covid_14.date)
    covid_7 = covid_7[ covid_7["date"] >= pd.to_datetime('20200317', format='%Y%m%d')]
    covid_7 = covid_7[ covid_7["date"] < max(covid_7['date'])- pd.to_timedelta(2,unit='d') ]
    covid_14 = covid_14[ covid_14["date"] >= max(covid_14['date'])- pd.to_timedelta(2,unit='d') ]
    covid = pd.concat([covid_7, covid_14])
    
    # edit format
    holiday['date'] = pd.to_datetime(holiday.date)
    popu = popu.rename(columns={"county_name": "county"})
    covid = popu[['county', 'population']].merge(covid, on='county', how='left')
    covid = holiday.merge(covid, on='date', how='right')
    covid = covid[['date', 'county', 'incid', 
                   'R_exp7', 'R_sig7', 'R_param_a7', 'R_param_b7', 'Prob_R7', 
                   'population', 'weekend', 'holidays', 'hdistance', 'tholiday', 
                   'R_exp14', 'R_sig14', 'R_param_a14', 'R_param_b14', 'Prob_R14'
                   ]]
    covid = covid.rename(columns={ "incid": "incid_x"})
    # covid = covid.sort_values(by =['county', 'date'])
    
    # print summary 
    print('data summary')
    print('Covid set shape == {}'.format(covid.shape))
    print('Featured selected: {}'.format(list(covid)[3:] ))      
    return covid



def data_scale_withincid(covid):
    cols_vars = covid.iloc[:, 2:]
    sc = StandardScaler()   
    cols_vars_scaled = sc.fit_transform(cols_vars)
    cols_vars_scaled = pd.concat([covid["date"], covid["county"], covid["incid_x"], 
                                  pd.DataFrame(cols_vars_scaled) ], axis=1)    
    return cols_vars_scaled
 
 
        
# creating train set
def input_output_set(covid, cols_vars_scaled, n_past, n_future):
    
    # define train set
    X_set = []
    y_set = []
    X_set_single = []
    
    # make train set
    county_names = sorted(list( covid.county.unique() ))   
    date_list = sorted(list( set(pd.to_datetime(covid.date)) ))
    
    for c in range(0, len(county_names) ) :
        df = cols_vars_scaled.loc[ covid["county"] == county_names[c]].sort_values(by ="date")
        
        for i in range(n_past, len(date_list)-n_future+1 ):  
            X_set.append(df.iloc[i - n_past:i, 3:].to_numpy())
            X_set_single.append (df.iloc[i-1, 3:].to_numpy())    
            y_set.append(df.iloc[i + n_future-1:i + n_future, 2].to_numpy())
                           
    X_set, y_set = np.array(X_set), np.array(y_set)
    
    # print summary
    print('train set summary')
    print('X_set shape == {}.'.format(X_set.shape))  
    print('y_set shape == {}.'.format(y_set.shape))    
    return (X_set, y_set)



def input_pred_set(covid, cols_vars_scaled, n_past, n_future):
    
    # define pred set 
    X_pred = []
    
    # make pred set
    county_names = sorted(list( covid.county.unique() ))
    date_list = sorted(list( set(pd.to_datetime(covid.date)) ))
    
    for c in range(0, len(county_names) ) :
        df = cols_vars_scaled.loc[ covid["county"] == county_names[c]].sort_values(by ="date")
        
        for i in range(  len(date_list)-n_future+1, len(date_list)+1 ):    
            X_pred.append(df.iloc[i-n_past:i, 3:].to_numpy())
    X_pred = np.array(X_pred)
    
    # print summary
    print('X_set shape == {}.'.format(X_pred.shape))
    return (X_pred)



def train_test_set_f_dropaweek(X_set, y_set):
    
    s= int(len(X_set)/55)
    for c in range(0, 55 ) :
       if c == 0 :
            npx= X_set[int(s*c):int(s*(c+1)-7)]
            npy= y_set[int(s*c):int(s*(c+1)-7)]
       else:
            npx= np.concatenate((npx, X_set[int(s*c):int(s*(c+1)-7)]))
            npy= np.concatenate((npy, y_set[int(s*c):int(s*(c+1)-7)]))
    X_train, X_test, y_train, y_test = train_test_split(npx, npy, test_size=0.30)  
    
    # print summary
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, npx.shape, npy.shape)
    return (X_train, X_test, y_train, y_test)



#%% data call
    
covid = data_prep_new(covid_7,covid_14, popu, holiday)
cols_vars_scaled = data_scale_withincid(covid)  

X_set, y_set = input_output_set(covid, cols_vars_scaled, n_past, n_future)
X_pred = input_pred_set(covid, cols_vars_scaled, n_past, n_future)

X_train, X_test, y_train, y_test = train_test_set_f_dropaweek(X_set, y_set )



##################################%% third step: making model

# Initializing
def generate_model(dropout, neuronCount, neuronShrink, activFun):
    model = Sequential()
    
    model.add(LSTM(units=int(neuronCount), activation=activFun, return_sequences=True, input_shape=(n_past, X_train.shape[2])))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units=int(neuronCount * neuronShrink), activation=activFun, return_sequences=True))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units=int(neuronCount* neuronShrink* neuronShrink), activation=activFun, return_sequences=False))  
    model.add(Dropout(dropout))
    
    model.add(Dense(units=1, activation='linear'))  #linear/sigmoid
    model.summary()
    return model 

   

# Compiling 
def fitted_model(dropout,neuronCount,neuronShrink, lr, activFun, epoch, batch):
    
    model = generate_model(dropout, neuronCount, neuronShrink, activFun)
    model.compile(optimizer = Adam(learning_rate=lr), loss='mean_absolute_error')  
    
    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=300, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=300, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tb = TensorBoard('logs') 
    history = model.fit(X_train, y_train, shuffle=True, epochs=epoch, callbacks=[es, rlr, mcp, tb],
                        validation_data=(X_test, y_test), verbose=1, batch_size=batch) 
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('C:/Users/makhodaverdi/Desktop/R_code/image_loss.png')
    plt.show()
    
    return model, history



# evaluate  
def evaluate_model(neuronCount, dropout, neuronShrink, lr, activFun,  epoch, batch): 
    model, history = fitted_model(dropout,neuronCount,neuronShrink, lr, activFun,  epoch, batch)
    pred = model.predict(X_test)
    mae_test = round(np.abs(np.subtract(y_test , pred)).mean(),3) 
    return -mae_test



#%% call model

pbound = { 'dropout': (0.1, 0.8),  
           'neuronCount': (4, 64),  
           'neuronShrink': (0.1, 0.9),
           'lr': (0.00001, 0.05),
           'epoch': (4, 512), 
	   'batch' : (4, 512),
	   'activFun' :("relu", "tanh")
          }

optimizer = BayesianOptimization(f=evaluate_model, pbounds=pbound, verbose=2, random_state=1)
start_time = time.time()
optimizer.maximize(init_points=128, n_iter=256,)
time_took = time.time() - start_time

dropout = optimizer.max['params']['dropout']
neuronCount = optimizer.max['params']['neuronCount']
neuronShrink = optimizer.max['params']['neuronShrink']
lr = optimizer.max['params']['lr']
epoch = optimizer.max['params']['epoch']
batch = optimizer.max['params']['batch']
activFun = optimizer.max['params']['activFun']

print(optimizer.max['params'])

model, history = fitted_model(dropout,neuronCount,neuronShrink, lr, activFun,  epoch, batch)

 

##################################%% prediction       

date_list = sorted(list( set(pd.to_datetime(covid.date)) ))
date_list_future = pd.date_range(date_list[-1], periods=n_future, freq='1d').tolist()
datelist_future2 = []
for this_timestamp in date_list_future:
    datelist_future2.append(this_timestamp.date())

    
for c in range(0, len(county_names) ) : 
    df = covid.loc[ covid["county"] == county_names[c]].sort_values(by ="date")
    df = df.set_index('date')
    cpop = float(pop[ pop["county_name"] == county_names[c] ].population)
          
    predictions_future = model.predict(X_pred[7*c:7*(c+1),:,:])   
    predictions_train = model.predict(X_set[int(len(X_set)/55)*c:int(len(X_set)/55)*(c+1)-7,:,:])  
    
    PREDICTIONS_FUTURE = pd.DataFrame(predictions_future, columns=['incid']).set_index(pd.Series(datelist_future2))
    PREDICTION_TRAIN = pd.DataFrame(predictions_train, columns=['incid']).set_index(pd.Series(date_list[n_past+n_future-1:-7]))
     
    PREDICTIONS_FUTURE2 = PREDICTIONS_FUTURE
    for i in range (0,n_future-1):
        PREDICTIONS_FUTURE2.incid[i] = max(PREDICTIONS_FUTURE2.incid[i], 0)
    
    sum_report.append ([round(sum(PREDICTIONS_FUTURE2.incid),2) ,
                        sum(df.incid_x[-n_past-1:-1]),  
                        round(round(sum(PREDICTIONS_FUTURE2.incid),2)/max(sum(df.incid_x[-n_past-1:-1]), 0.0001),2), 
                        round(round(round(sum(PREDICTIONS_FUTURE2.incid),2)/max(sum(df.incid_x[-n_past-1:-1]), 0.0001),2)/cpop*100000, 2), 
                        round(round(sum(PREDICTIONS_FUTURE2.incid),2)-max(sum(df.incid_x[-n_past-1:-1]), 0),2)
                        ])



final_report = pd.DataFrame(sum_report,columns=['sum_predicted', 'sum_week_before', 'percent_increase_predicted',
                                                'percent_population_increase_predicted', 'case_increase_predicted']
                            ).set_index(pd.Series(county_names)
                            ).sort_values(by =['percent_increase_predicted', 'sum_predicted'], ascending=False)
final_report.to_csv('C:/Users/khodaverdi/Desktop/reports/predict_bumped7_{}.csv'.format(datelist_future2[0]))


