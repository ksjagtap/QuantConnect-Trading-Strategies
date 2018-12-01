import tensorflow as tf

import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics
from keras import backend as K

from datetime import datetime, timedelta

import pandas as pd

### <summary>
### Basic template algorithm simply initializes the date range and cash. This is a skeleton
### framework you can use for designing an algorithm.
### </summary>

seed = 123
random.seed(seed)
np.random.seed(seed)

class ForexTrade(QCAlgorithm):
    '''Deep Learning Model'''

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2018,1, 20)  #Set Start Date
        self.SetEndDate(2018,11,5)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        # Find more symbols here: http://quantconnect.com/data
        self.currency = "EURUSD"
        self.AddForex(self.currency, Resolution.Daily)
        
        ## define a long list, short list and portfolio
        self.long_list = []
        self.short_list = []

        # Initialise indicators
        self.rsi = RelativeStrengthIndex(9)
        self.bb = BollingerBands(30,2,2)
        self.macd = MovingAverageConvergenceDivergence(12, 26, 9)
        self.stochastic = Stochastic(14, 3, 3)
        self.ema = ExponentialMovingAverage(9)

        prev_rsi = []
        prev_bb = []
        prev_macd = []
        lower_bb = []
        upper_bb = []
        sd_bb = []
        prev_stochastic = []
        prev_ema = []

        # Historical Currencty Data
        self.currency_data = self.History([self.currency,], 150, Resolution.Daily) # Drop the first 20 for indicators to warm up
        
        ytd_open = self.currency_data["open"][-1]
        ytd_close = self.currency_data["close"][-1]
        
        self.currency_data = self.currency_data[:-1]

        for tup in self.currency_data.loc[self.currency].itertuples():        
            # making Ibasedatabar for stochastic
            bar = QuoteBar(
                tup.Index, 
                "EURUSD",
                Bar(tup.bidclose, tup.bidhigh, tup.bidlow, tup.bidopen),
                0,
                Bar(tup.askclose, tup.askhigh, tup.asklow, tup.askopen),
                0,
                timedelta(days=1)
                )
        
    
            self.stochastic.Update(bar)
            prev_stochastic.append(float(self.stochastic.ToString()))
    
            self.rsi.Update(tup.Index, tup.close)
            prev_rsi.append(float(self.rsi.ToString()))
    
            self.bb.Update(tup.Index, tup.close)
            prev_bb.append(float(self.bb.ToString()))
            lower_bb.append(float(self.bb.LowerBand.ToString()))
            upper_bb.append(float(self.bb.UpperBand.ToString()))
            sd_bb.append(float(self.bb.StandardDeviation.ToString()))
    
            self.macd.Update(tup.Index, tup.close)
            prev_macd.append(float(self.macd.ToString()))
            
            self.ema.Update(tup.Index, tup.close)
            prev_ema.append(float(self.ema.ToString()))
        
        rsi_df = pd.DataFrame(prev_rsi, columns = ["rsi"])
        macd_df = pd.DataFrame(prev_macd, columns = ["macd"])
        upper_bb_df = pd.DataFrame(upper_bb, columns = ["upper_bb"])
        lower_bb_df = pd.DataFrame(lower_bb, columns = ["lower_bb"])
        sd_bb_df = pd.DataFrame(sd_bb, columns = ["sd_bb"])
        stochastic_df = pd.DataFrame(prev_stochastic, columns = ["stochastic"])
        ema_df = pd.DataFrame(prev_ema, columns = ["ema"])


        self.indicators_df = pd.concat([rsi_df, macd_df, upper_bb_df, lower_bb_df, sd_bb_df, stochastic_df, ema_df], axis=1)
        self.indicators_df = self.indicators_df.iloc[20:]
        self.indicators_df.reset_index(inplace=True, drop=True)


        self.currency_data.reset_index(level = [0, 1], drop = True, inplace = True)
        
        self.currency_data.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh", "bidlow", "bidclose"], inplace=True)
        self.currency_data = self.currency_data.iloc[20:]
        self.currency_data.reset_index(inplace=True, drop=True)


        close_prev_prices = self.previous_prices("close", self.currency_data["close"], 6)
        open_prev_prices = self.previous_prices("open", self.currency_data["open"], 6)
        high_prev_prices = self.previous_prices("high", self.currency_data["high"], 6)
        low_prev_prices = self.previous_prices("low", self.currency_data["low"], 6)
        
        all_prev_prices = pd.concat([close_prev_prices, open_prev_prices, high_prev_prices, low_prev_prices], axis=1)
        
        final_table = self.currency_data.join(all_prev_prices, how="outer")
        final_table = final_table.join(self.indicators_df, how="outer")
        
        # Drop NaN from feature table
        self.features = final_table.dropna()
        
        self.features.reset_index(inplace=True, drop=True)
        
        # Make labels
        self.labels = self.features["close"]
        self.labels = pd.DataFrame(self.labels)
        self.labels.index -= 1
        self.labels = self.labels[1:]
        new_row = pd.DataFrame({"close": [ytd_close]})
        self.labels = self.labels.append(new_row)
        self.labels.reset_index(inplace=True, drop=True)
        
        
        self.Debug(len(self.labels))
        self.Debug(len(self.features))
        
        
        ## Define scaler for this class
        self.scaler_X = MinMaxScaler()
        self.scaler_X.fit(self.features)
        self.scaled_features = self.scaler_X.transform(self.features)
        
        self.scaler_Y = MinMaxScaler()
        self.scaler_Y.fit(self.labels)
        self.scaled_labels = self.scaler_Y.transform(self.labels)
        
        ## fine tune the model to determine hyperparameters
        ## only done once (upon inititialize)
        
        tscv = TimeSeriesSplit(n_splits=2)
        cells = [100, 200]
        epochs = [100, 200]
        
        ## create dataframee to store optimal hyperparams
        params_df = pd.DataFrame(columns = ["cells", "epoch", "mse"])
        
        # # ## loop thru all combinations of cells and epochs
        # for i in cells:
        #     for j in epochs:
                
        #         print("CELL", i, "EPOCH", j)
                
        #         # list to store the mean square errors
        #         cvscores = []
                
        #         for train_index, test_index in tscv.split(self.scaled_features):
        #             #print(train_index, test_index)
        #             X_train, X_test = self.scaled_features[train_index], self.scaled_features[test_index]
        #             Y_train, Y_test = self.scaled_labels[train_index], self.scaled_labels[test_index]
                    
        #             X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        #             X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                    
        #             model = Sequential()
        #             model.add(LSTM(i, input_shape = (1, X_train.shape[2]), return_sequences = True))
        #             model.add(Dropout(0.10))
        #             model.add(LSTM(i,return_sequences = True))
        #             model.add(LSTM(i))
        #             model.add(Dropout(0.10))
        #             model.add(Dense(1))
        #             model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
        #             model.fit(X_train,Y_train,epochs=j,verbose=0)
                    
        #             scores = model.evaluate(X_test, Y_test)
        #             cvscores.append(scores[1])
                    
        #         ## get average value of mean sq error    
        #         MSE = np.mean(cvscores)
                
        #         ## make this df for cells, epoch and mse and append to params_df
        #         this_df = pd.DataFrame({"cells": [i], "epoch":[j], "mse": [MSE]})
        #         # self.Debug(this_df)
        #         # params_df = params_df.append(this_df)
                
        #         params_df = params_df.append(this_df)
        #         self.Debug(params_df)
                    
        
        
        # # Check the optimised values (O_values) obtained from cross validation
        # # This code gives the row which has minimum mse and store the values to O_values
        # O_values = params_df[params_df['mse'] == params_df['mse'].min()]
                
        # # Extract the optimised values of cells and epochcs from abbove row (having min mse)
        self.opt_cells = 200
        self.opt_epochs = 100
        # self.opt_cells = O_values["cells"][0]
        # self.opt_epochs = O_values["epoch"][0]
        
        
        X_train = np.reshape(self.scaled_features, (self.scaled_features.shape[0], 1, self.scaled_features.shape[1]))
        y_train = self.scaled_labels
        
        
        self.session = K.get_session()
        self.graph = tf.get_default_graph()

        # Intialise the model with optimised parameters
        self.model = Sequential()
        self.model.add(LSTM(self.opt_cells, input_shape = (1, X_train.shape[2]), return_sequences = True))
        self.model.add(Dropout(0.20))
        self.model.add(LSTM(self.opt_cells,return_sequences = True))
        self.model.add(Dropout(0.20))
        self.model.add(LSTM(self.opt_cells, return_sequences = True))
        self.model.add(LSTM(self.opt_cells))
        self.model.add(Dropout(0.20))
        self.model.add(Dense(1))
        
        # self.model.add(Activation("softmax"))
        self.model.compile(loss= 'mean_squared_error',optimizer = 'adam', metrics = ['mean_squared_error'])
        
        
        # self.model.fit(X_train, y_train, epochs=opt_epochs, verbose=0)


    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.

        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''
        current_price = data[self.currency].Price
        
        ## call in previous 1 day data
        ytd_data = self.History([self.currency,], 1, Resolution.Daily)
        
        if ytd_data.empty:
            ytd_data = self.History([self.currency,], 2, Resolution.Daily)

        ### Update the Features and Labels for Fitting ###
        
        # ## pop first row from features and labels
        # self.features = self.features[1:]
        # self.labels = self.labels[1:]
        
        # ## Append new row to features
        # self.features = self.features.append(self.t_minus_2)

        # ## APPEND TODAYS PRICE TO LABELS
        # new_val = pd.DataFrame({"open":[current_price]})
        # self.labels = pd.concat([self.labels, new_val], ignore_index=True)
        
        ## isolate last 6 rows of df for lookback
        features_t_minus_1 = self.features[-6:]
        
        ## generate prev 6 datapoints (as features)
        close_prev_prices = self.previous_prices("close", features_t_minus_1["close"], 6)
        open_prev_prices = self.previous_prices("open", features_t_minus_1["open"], 6)
        high_prev_prices = self.previous_prices("high", features_t_minus_1["high"], 6)
        low_prev_prices = self.previous_prices("low", features_t_minus_1["low"], 6)
        
        ## join all
        all_prev_prices = pd.concat([close_prev_prices, open_prev_prices, high_prev_prices, low_prev_prices], axis=1)
        all_prev_prices.reset_index(drop=True, inplace=True)
        
        
        ## get the indicators
        prev_stochastic, prev_rsi, prev_bb, lower_bb, upper_bb, prev_macd, sd_bb, prev_ema = [],[],[],[],[],[],[],[]
        
        for tup in ytd_data.loc[self.currency].itertuples():        
            # making Ibasedatabar for stochastic
            bar = QuoteBar(tup.Index, 
                           self.currency,
                           Bar(tup.bidclose, tup.bidhigh, tup.bidlow, tup.bidopen),
                           0,
                           Bar(tup.askclose, tup.askhigh, tup.asklow, tup.askopen),
                           0,
                           timedelta(days=1)
                          )
            
            self.stochastic.Update(bar)
            prev_stochastic.append(float(self.stochastic.ToString()))
            
            self.rsi.Update(tup.Index, tup.close)
            prev_rsi.append(float(self.rsi.ToString()))
            
            self.bb.Update(tup.Index, tup.close)
            prev_bb.append(float(self.bb.ToString()))
            lower_bb.append(float(self.bb.LowerBand.ToString()))
            upper_bb.append(float(self.bb.UpperBand.ToString()))
            sd_bb.append(float(self.bb.StandardDeviation.ToString()))
            
            self.macd.Update(tup.Index, tup.close)
            prev_macd.append(float(self.macd.ToString()))
            
            self.ema.Update(tup.Index, tup.close)
            prev_ema.append(float(self.ema.ToString()))
            
        
        rsi_df = pd.DataFrame(prev_rsi, columns = ["rsi"])
        macd_df = pd.DataFrame(prev_macd, columns = ["macd"])
        upper_bb_df = pd.DataFrame(upper_bb, columns = ["upper_bb"])
        lower_bb_df = pd.DataFrame(lower_bb, columns = ["lower_bb"])
        sd_bb_df = pd.DataFrame(sd_bb, columns = ["sd_bb"])
        stochastic_df = pd.DataFrame(prev_stochastic, columns = ["stochastic"])
        ema_df = pd.DataFrame(prev_ema, columns = ["ema"])
        
        indicators_df = pd.concat([rsi_df, macd_df, upper_bb_df, lower_bb_df, sd_bb_df, stochastic_df, ema_df], axis=1)
        indicators_df.reset_index(inplace=True, drop=True)
        
        ytd_data.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh", "bidlow", "bidclose"], inplace=True)
        ytd_data.reset_index(drop=True, inplace=True)
        
        ytd_data = ytd_data.join(all_prev_prices, how="outer")
        ytd_data = ytd_data.join(indicators_df, how="outer")
        
        # # Set Next day's T-2 as ytd_data
        # self.t_minus_2 = ytd_data
        
        # Scaling
        # scaled_features = self.scaler_X.transform(self.features)
        
        # scaler_Y = MinMaxScaler()
        # scaler_Y.fit(self.labels)
        # scaled_labels = scaler_Y.transform(self.labels)
        
        # Reshape Features
        # X_train = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))
        # y_train = scaled_labels
        
        # Fit the model at every onData
        with self.session.as_default():
            with self.graph.as_default():
                # self.model.fit(X_train, y_train, epochs=self.opt_epochs, verbose=0)
        
        
                # Get prediction for ytd_data instance to predict T+1
                # scaler_X.fit(ytd_data)
                self.Debug(str(ytd_data))
                scaled_ytd_data = self.scaler_X.transform(ytd_data)
                X_predict = np.reshape(scaled_ytd_data, (scaled_ytd_data.shape[0], 1, scaled_ytd_data.shape[1]))
                
                close_price = self.model.predict_on_batch(X_predict)
                                
        
                close_price_prediction = self.scaler_Y.inverse_transform(close_price)
                close_price_prediction = close_price_prediction[0][0]
                self.Debug(close_price_prediction)
        
        
        ## BUY/SELL STRATEGY BASED ON PREDICTED PRICE
        #Make decision for trading based on the output from LSTM and the current price.
        #If output ( forecast) is greater than current price , we will buy the currency; else, do nothing.
        # Only one trade at a time and therefore made a list " self.long_list". 
        #As long as the currency is in that list, no further buying can be done.
        # Risk and Reward are defined: Ext the trade at 1% loss or 1 % profit.
        # Generally the LSTM model can predict above/below the current price and hence a random value is used
        #to scale it down/up. Here the number is 1.1 but can be backtested and optimised.
        
        # If RSI is below 30, shouldn't 
        if close_price_prediction > current_price and self.currency not in self.long_list and self.currency not in self.short_list:
            
            self.Debug("output is greater")
            # Buy the currency with X% of holding in this case 90%
            self.SetHoldings(self.currency, 0.9)
            self.long_list.append(self.currency)
            self.Debug("long")
            
        if self.currency in self.long_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            #self.Debug("cost basis is " +str(cost_basis))
            if  ((current_price <= float(0.99) * float(cost_basis)) or (current_price >= float(1.03) * float(cost_basis))):
                self.Debug("SL-TP reached")
                #self.Debug("price is" + str(price))
                #If true then sell
                self.SetHoldings(self.currency, 0)
                self.long_list.remove(self.currency)
                self.Debug("squared")
        #self.Debug("END: Ondata")
        
        # Short
        if close_price_prediction < current_price and self.currency not in self.short_list and self.currency not in self.long_list:
                
            self.SetHoldings(self.currency, -0.9)
            self.short_list.append(self.currency)
            self.Debug("short")
            
                
        if self.currency in self.short_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            #self.Debug("cost basis is " +str(cost_basis))
            if  ((current_price <= float(0.97) * float(cost_basis)) or (current_price >=float(1.01) * float(cost_basis))):
                self.Debug("SL-TP reached")
                #self.Debug("price is" + str(price))
                #If true then sell
                self.SetHoldings(self.currency, 0)
                self.short_list.remove(self.currency)
                self.Debug("squared")
        
                
    

    def previous_prices(self, raw_type, data, num_lookback):
        
            '''
            num_lookback is the number of previous prices
            Data is open, high, low or close
            Data is a series
            Returns a dataframe of previous prices
            '''
            
            prices = []
            length = len(data)
        
            for i in range(num_lookback, length+1):
                this_data = np.array(data[i-num_lookback : i])
                prices.append(this_data)
        
            prices_df = pd.DataFrame(prices)
                
            columns = {}
            
            for index in prices_df.columns:
                columns[index] = "{0}_shifted_by_{1}".format(raw_type, num_lookback - index)
            
            prices_df.rename(columns = columns, inplace=True)
            prices_df.index += num_lookback
            
            return prices_df