import numpy as np
from numpy.random import seed

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import pandas as pd


class Forex_Trade_Logreg(QCAlgorithm):
	
#####  17 to 34:  Initialization of Algo ####
    def Initialize(self):
  
        #self.Debug("START: Initialize")
        self.SetStartDate(2018,7,1)    #Set Start Date
        self.SetEndDate(2018,10,15)     #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin) #Set Brokerage Model
        self.currency = "EURUSD"
        self.variable1 = "GBPJPY" 
        self.variable2 = "USDJPY"
        self.AddForex(self.currency,Resolution.Daily)
        self.AddForex(self.variable1,Resolution.Daily)
        self.AddForex(self.variable2,Resolution.Daily)
        self.long_list =[]
        self.short_list =[]
        self.model = LogisticRegression() 
        self.x=0
        #self.Debug("End: Initialize")

#####  37 to 48 : Defining OnData function and Geting the Historical Data  ####
    def OnData(self, data): #This function runs on every resolution of data mentioned. 
                            #(eg if resolution = daily, it will run daily, if resolution = hourly, it will run hourly.)
        
        #self.Debug("START: Ondata")
        currency_data  = self.History([self.currency], 500, Resolution.Daily) # Asking for historical data
        currency_data1 = self.History([self.variable1], 500, Resolution.Daily)
        currency_data2 = self.History([self.variable2], 500, Resolution.Daily)
        
        L= len(currency_data) # Checking the length of data
        L1= len(currency_data1)
        L2= len(currency_data2)
        #self.Debug("The length is " + str (L))
    
#####  52 to 81 : Check condition for required data and prepare X and Y for modeling  ####    
        # Making sure that the data is not empty and then only proceed with the algo
        if ( not currency_data.empty and  not currency_data1.empty and  not currency_data2.empty and L == L1 ==L2 ): 

            data = pd.DataFrame(currency_data.close)  #Get the close prices. Also storing as dataframe
            data1 = pd.DataFrame(currency_data1.close) # dataframes are good for calculating lags and percent change
            data2 = pd.DataFrame(currency_data2.close)
            
            #Data Preparation for input to Logistic Regression
            stored = {} # To prepare and store data
            for i in range(11): # For getting 10 lags ...Can be increased if morr lags are required
                stored['EURUSD_lag_{}'.format(i)] = data.shift(i).values[:,0].tolist() #creating lags
                stored['GBPJPY_lag_{}'.format(i)] = data1.shift(i).values[:,0].tolist()
                stored['USDJPY_lag_{}'.format(i)] = data2.shift(i).values[:,0].tolist()
    
            stored = pd.DataFrame(stored)
            stored = stored.dropna() # drop na values
            stored = stored.reset_index(drop=True)
            
            corelation = stored.corr()
            self.Debug("corr is" +str(corelation))
            stored["Y"] = stored["EURUSD_lag_0"].pct_change()# get the percent change from previous time
            
            for i in range(len(stored)): # loop to make Y as categorical
                if stored.loc[i,"Y"] > 0:
                    stored.loc[i,"Y"] = "UP"
                else:
                    stored.loc[i,"Y"] = "DOWN"
                    
            #self.Debug("All X_data is " +str(stored))    
            
            X_data = stored.iloc[:,np.r_[3:33]]  # extract only lag1, Lag2, lag3.. As Lag 0 is the data itself and will  not be available during prediction
            #self.Debug( "X data is" +str(X_data))
            
            Y_data = stored["Y"]
            #self.Debug( "Y data is" +str(Y_data))
            
#####  85 to 97 : Build the Logistic Regression model, check the training accuracy and coefficients  ####     
            if self.x==0:  #To make sure the model is build only once and avoid computation at every new data
                
                self.model.fit(X_data,Y_data)
                score = self.model.score(X_data, Y_data)
                self.Debug("Train Accuracy of final model: " + str(score))
                
                # To get the coefficients from model
                A = pd.DataFrame(X_data.columns)
                B = pd.DataFrame(np.transpose(self.model.coef_))
                C =pd.concat([A,B], axis = 1)
                self.Debug("The coefficients are: "+ str(C))
                    
            self.x=1     # End the model
   
#####  102 to 110 : Prepare data for prediction   ####             
            
            #Prepare test data similar way as earlier
            test = {}
            for i in range(10):
                test['EURUSD_lag_{}'.format(i+1)] = data.shift(i).values[:,0].tolist()
                test['GBPJPY_lag_{}'.format(i+1)] = data1.shift(i).values[:,0].tolist()
                test['USDJPY_lag_{}'.format(i+1)] = data2.shift(i).values[:,0].tolist()
    
            test = pd.DataFrame(test)
            test = pd.DataFrame(test.iloc[-1, :]) # take the last values 
            test = pd.DataFrame(np.transpose(test)) # transose to get in desired model shape

    
#####  115 to 116 : Make Prediction   #### 
    
            output = self.model.predict(test)
            self.Debug("Output from LR model is" + str(output))
            
            
            #Checking the current price 
            price = currency_data.close[-1]
            #self.Debug("Current price is" + str(price))
            
            #Make decision for trading based on the output from LR and the current price.
            #If output ( forecast) is UP we will buy the currency; else, Short.
            # Only one trade at a time and therefore made a list " self.long_list". 
            #As long as the currency is in that list, no further buying can be done.
            # Risk and Reward are defined: Ext the trade at 1% loss or 1 % profit.
 
##### 130 to 168 : Entry /Exit Conditions for trading  #### 
            if output == "UP"  and self.currency not in self.long_list and self.currency not in self.short_list :
                
                #self.Debug("output is greater")
                # Buy the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, 0.9)
                self.long_list.append(self.currency)
                self.Debug("long")
                
            if self.currency in self.long_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(0.995) * float(cost_basis)) or (price >= float(1.01) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then sell
                    self.SetHoldings(self.currency, 0)
                    self.long_list.remove(self.currency)
                    self.Debug("squared long")
                    
                    
            if output =="DOWN"  and self.currency not in self.long_list and self.currency not in self.long_list:
                
                #self.Debug("output is lesser")
                # Buy the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, -0.9)
                self.short_list.append(self.currency)
                self.Debug("short")
                
            if self.currency in self.short_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(0.99) * float(cost_basis)) or (price >= float(1.005) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then buy back
                    self.SetHoldings(self.currency, 0)
                    self.short_list.remove(self.currency)
                    self.Debug("squared short")
            #self.Debug("END: Ondata")