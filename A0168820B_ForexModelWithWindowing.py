import tensorflow as tf
import random
import numpy as np
import pandas as pd


seed  = 15
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


class BasicTensorFlowAlgorithmSingleAssetClassifier(QCAlgorithm):

    def Initialize(self):

        # init the tensorflow model object
        self.model = Model()

        # setup backtest
        self.SetStartDate(2016,1,1)  #Set Start Date
        self.SetEndDate(2018,9,30)    #Set End Date
        self.SetCash(100000)         #Set Strategy Cash
        
        ## Find more symbols here: http://quantconnect.com/data
        self.symbol       = self.AddCrypto("BTCUSD", Resolution.Minute).Symbol
        self.model.symbol = self.symbol

        #self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        #self.AddSecurity(self.symbol, Resolution.Minute)
        #self.AddForex("USDBTC", Resolution.Minute)
        

        # Our big history call, only done once to save time
        self.model.hist_data = self.History([self.symbol,], self.model.warmup_count, Resolution.Minute).astype(np.float32)
        
        # Flag to know when to start gathering history in OnData or Rebalance
        self.do_once         = True

        # prevent order spam by tracking current weight target and comparing against new targets
        self.target          = 0.0
        
        # We are forecasting and trading on open-to-ooen price changes on a daily time scale. So work every morning.
        self.Schedule.On(self.DateRules.EveryDay(self.symbol), \
            self.TimeRules.AfterMarketOpen(self.symbol), \
            Action(self.Rebalance))
            


    def Rebalance(self):
        
        # store current price for model to use at end of historical data
        self.model.current_price = float(self.Securities[self.symbol].Price)
        
        # Accrew history over time vs making huge, slow history calls each step.
        if not self.do_once:
            new_hist      = self.History([self.symbol,], 1, Resolution.Minute).astype(np.float32)
            self.model.hist_data = self.model.hist_data.append(new_hist).iloc[1:] #append and pop stack
        else:
            self.do_once  = False
        
        # Prepare our data now that it has been updated
        self.model.preprocessing()
        
        # Perform a number of training steps with the new data
        self.model.train(self)
        
        # Using the latest input feature set, lets get the predicted assets expected to make the desired profit by the next open
        signal = self.model.predict(self)

        # In case of repeated forecast, lets skip rebalance and reduce fees/orders         
        if signal != self.target:
            
            # track our current target to allow for above filter
            self.target = signal*0.8
            # rebalance
            self.SetHoldings(self.symbol, self.target, liquidateExistingHoldings = False)


class Model():

    def __init__(self):

        # Number of inputs for training (will loose 1)
        self.eval_lookback  = 252*4 + 1# input batch size will be eval_lookback+n_features-1  #252*4+1
        
        # We will feed in the past n open-to-open price changes
        self.n_features     = 15 #15
        
        # How much historical data do we need?
        self.warmup_count   = self.eval_lookback + self.n_features

        # define our tensorflow model/network
        self.network_setup()
        
        # We track the current price(rebalance at open) to use at end of history
        self.current_price  = None

    def network_setup(self):
        
        self.sess = tf.InteractiveSession()

        # Our feed dicts pipe data into these tensors on runs/evals. Input layer and correct-labels.
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])

        # The brain of our network, the weights and biases. Nice and simple for a linear softmax network.
        self.W = tf.Variable(tf.zeros([self.n_features,2]))
        self.b = tf.Variable(tf.zeros([2]))

        # The actual model is a painfully simple linear regressor
        self.y = tf.matmul(self.x,self.W) + self.b
    
        # output layer
        self.y_pred = tf.nn.softmax(self.y)

        # loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        # For fun we use AdamOptimizer instead of basic vanilla GradientDescentOptimizer.
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.cross_entropy)

        # metric ops
        self.correct_prediction = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # This is done later vs Tensorflow Tutorial because of AdamOptimizer usage, which needs its own vars to be init'ed
        self.sess.run(tf.global_variables_initializer())

    def preprocessing(self):
        
        # Input features:
        # We are using a sliding window of past change in open prices per asset to act as our input "image". 
        #By no means a good idea to discover alpha...
        
        all_data = np.append(self.hist_data.open.values.flatten().astype(np.float32), self.current_price)
        features   = []
        labels     = []
        for i in range(self.n_features+1, len(all_data)-1):
            # input is change in price
            features.append( np.diff(all_data[i-self.n_features-1:i].copy()) )
            # label is change in price from last day in input to the next day
            dp = 100.*(all_data[i+1]-all_data[i])/all_data[i]
            if dp > 0.0:
                dp = 1
            else:
                dp = 0
            labels.append(dp)
        features = np.array(features)
        labels   = np.array(labels)

        # convert to one hot for tensorflow
        oh     = np.zeros((len(labels),2))
        oh[np.arange(len(labels)),labels] = 1.0
        labels = oh
        
        # Test train split. unfortunate to loose recent data, but need data not seen ever by train set.
        split_len    = int(len(labels)*0.05)
        self.X_train = features[:-split_len]
        self.X_test  = features[-split_len:]
        self.y_train = labels[:-split_len]
        self.y_test  = labels[-split_len:]

    def train(self, algo_context):
        
        # Perform  training step(s) and check train accuracy.
        for _ in range(100):
            #batch = np.random.permutation(np.arange(len(self.X_train)))[:100] #can switch to mini batch easily if need be
            self.train_step.run(session=self.sess, feed_dict={self.x: self.X_train, self.y_: self.y_train})
            
        # Collect some metrics for charting
        self.train_accuracy = self.accuracy.eval(session=self.sess, feed_dict={self.x: self.X_train, self.y_: self.y_train})
        self.test_accuracy  = self.accuracy.eval(session=self.sess, feed_dict={self.x: self.X_test, self.y_: self.y_test})
        self.train_ce       = self.cross_entropy.eval(session=self.sess, feed_dict={self.x: self.X_train, self.y_: self.y_train})
        self.test_ce        = self.cross_entropy.eval(session=self.sess, feed_dict={self.x: self.X_test, self.y_: self.y_test})
        #algo_context.Log("Train Accuracy: %0.5f %0.5f"%(self.train_accuracy,self.test_accuracy)) # commented out to reduce log
        
    def predict(self, algo_context):
        
        # Perform inference
        pred_feat  =  np.append(self.hist_data.open.values.flatten().astype(np.float32), self.current_price)[-self.n_features-1:]
        pred_feat  = np.diff(pred_feat)
        pred_proba = self.y_pred.eval(session=self.sess, feed_dict={self.x: [pred_feat]})
        
        #algo_context.Log("Forecast: Long p: %0.3f\tCashh p: %0.3f"%(pred_proba[0][0], pred_proba[0][1])) # commented out to reduce log
        self.current_forecast = pred_proba[0]
        
        # Return if probability suggests Cash or Long class
        return np.argmax(pred_proba[0])