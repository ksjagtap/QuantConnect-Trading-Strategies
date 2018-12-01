from math import floor
from datetime import timedelta
from datetime import datetime


class CurrencyTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        
        self.SetStartDate(2016,1,1)
        self.SetEndDate(2018,9,30)
        self.SetCash(10000)

        self.AddForex("EURUSD", Resolution.Daily).Symbol

        self.macd = self.MACD("SPY", 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
        self.previous = datetime.min
        self.PlotIndicator("MACD", True, self.macd, self.macd.Signal)
        self.PlotIndicator("EURUSD", self.macd.Fast, self.macd.Slow)


    def OnData(self, data):
        if not self.macd.IsReady:
            return

        if self.previous.date() == self.Time.date():
            return

        tolerance = 0.01

        holdings = self.Portfolio["EURUSD"].Quantity

        signalDeltaPercent = (self.macd.Current.Value - self.macd.Signal.Current.Value)/self.macd.Fast.Current.Value

        # if our macd is greater than our signal, then let's go long
        if holdings <= 0 and signalDeltaPercent > tolerance:  # 0.01%
            # longterm says buy as well
            self.SetHoldings("EURUSD", 1.0)

        # of our macd is less than our signal, then let's go short
        elif holdings >= 0 and signalDeltaPercent < -tolerance:
            self.Liquidate("EURUSD")