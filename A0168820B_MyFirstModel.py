from math import floor
from datetime import timedelta
from datetime import datetime

""" My first model: Trading the Euro-USD currency pair MACD. In this model, there are two moving averages,
    one fast and one slow. The theory is that whenever they cross, there is going to be a reversion in 
    the direction of the security movement. However, this strategy does not work so well """


class CurrencyTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        
        self.SetStartDate(2016,1,1)
        self.SetEndDate(2018,9,30)
        self.SetCash(10000)

        self.AddForex("EURUSD", Resolution.Daily).Symbol

        self.macd = self.MACD("EURUSD", 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
        self.previous = datetime.min
        self.PlotIndicator("MACD", True, self.macd, self.macd.Signal)
        self.PlotIndicator("EURUSD", self.macd.Fast, self.macd.Slow)


    def OnData(self, data):
        if not self.macd.IsReady:
            return

        if self.previous.date() == self.Time.date():
            return

        tolerance = 0.0025

        holdings = self.Portfolio["EURUSD"].Quantity

        signalDeltaPercent = (self.macd.Current.Value - self.macd.Signal.Current.Value)/self.macd.Fast.Current.Value

        # if our macd is greater than our signal, then let's go long
        if holdings <= 0 and signalDeltaPercent > tolerance:  # 0.01%
            self.SetHoldings("EURUSD", 1.0)

        # of our macd is less than our signal, then let's go short
        elif holdings >= 0 and signalDeltaPercent < -tolerance:
            self.Liquidate("EURUSD")