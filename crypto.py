namespace QuantConnect 
{   
    public class MultiCoinFramework : QCAlgorithm
    {
    	string tickersString ="BTCUSD,ETHUSD,LTCUSD";
    	
    	decimal changes1Ratio=-1.0m; //The influence of change upon fitness.
    	decimal changes2Ratio=0.0m; //The influence of change in change upon fitness.
    	int emaOfChanges1Length=24; //The length of the change indicator.
		int emaOfChanges2Length=24; //The length of the change in change indicator.
		decimal leverage=1m;
		
		int historyLength=2;
		int changes1Length=2;
		int changes2Length=2;
		Resolution resolution=Resolution.Hour;
		List<StockData> stockDatas = new List<StockData>();
		string stockHeld="";
		
        public override void Initialize() 
        {
            SetStartDate(2017, 3, 1); 
            SetEndDate(2018, 3, 1);
            SetCash(1000);
			string[] tickers = tickersString.Split(new string[1] { "," }, StringSplitOptions.RemoveEmptyEntries);
			foreach (string ticker in tickers)
			{
				Symbol symbol = QuantConnect.Symbol.Create(ticker, SecurityType.Crypto, Market.GDAX);
				AddCrypto(symbol, resolution);
				StockData stockData=new StockData();
				stockData.Ticker=ticker;
				stockData.emaOfChanges1Indicator = new ExponentialMovingAverage(emaOfChanges1Length);
            	stockData.emaOfChanges2Indicator = new ExponentialMovingAverage(emaOfChanges2Length);
				stockDatas.Add(stockData);
			}
        }

        public override void OnData(Slice data) 
        {
        	foreach (StockData stockData in stockDatas)
        	{
	        	stockData.history.Add(data[stockData.Ticker].Close);
	        	if (stockData.history.Count>historyLength)
	        	{
	        		stockData.history.RemoveAt(0);
	        	}
				if (stockData.history.Count>=2)
				{
					if (stockData.history[stockData.history.Count-2]!=0)
					{
						decimal change=(stockData.history.Last()-stockData.history[stockData.history.Count-2])/stockData.history[stockData.history.Count-2];
						stockData.changes1History.Add(change);
						if (stockData.changes1History.Count>changes1Length)
						{
							stockData.changes1History.RemoveAt(0);
						}
					}
				}
				if (stockData.changes1History.Count>=2)
				{
					decimal change=stockData.changes1History.Last()-stockData.changes1History[stockData.changes1History.Count-2];
					stockData.changes2History.Add(change);
					if (stockData.changes2History.Count>changes2Length)
					{
						stockData.changes2History.RemoveAt(0);
					}
				}
				if (stockData.changes1History.Count>0)
				{
					stockData.emaOfChanges1Indicator.Update(Time,stockData.changes1History.Last());
				}
				if (stockData.changes2History.Count>0)
				{
					stockData.emaOfChanges2Indicator.Update(Time,stockData.changes2History.Last());
				}
				stockData.Fitness=changes1Ratio*stockData.emaOfChanges1Indicator+changes2Ratio*stockData.emaOfChanges2Indicator;
        	}

    	    var q1 = from x in stockDatas
    			orderby x.Fitness descending
    			select x;
        	
        	List<StockData> q2=q1.ToList();
        	if (q2.Count>0)		
        	{
        		StockData selectedStockData=q2.First();
        		if (selectedStockData.Ticker != stockHeld)
        		{
        			Liquidate();
        			SetHoldings(selectedStockData.Ticker, leverage);
					stockHeld=selectedStockData.Ticker;
        		}

        	}
        }
        
        class StockData
        {
        	public string Ticker;
			public List<decimal> history=new List<decimal>();
			public List<decimal> changes1History=new List<decimal>();
			public List<decimal> changes2History=new List<decimal>();
			public ExponentialMovingAverage emaOfChanges1Indicator;
			public ExponentialMovingAverage emaOfChanges2Indicator;
			public decimal Fitness;
        }
        
    }
}
