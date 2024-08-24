from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pytz


def getTimeLeft(year, month, day):
  """
  Returns time left to a specified date
  params:
    year (int): Year in which the bet ends
    month (int): Month in which the bet ends
    day (int): Day in which the bet ends

  The function is set by default to ET timezone and 00:00 of the given day
  """
  et_timezone = pytz.timezone('America/New_York')
  today = datetime.now(et_timezone)
  last = et_timezone.localize(datetime(year, month, day, 0, 0, 0))
  time_left = last - today
  time_left = round(time_left.total_seconds()/3600)
  return time_left


def getTickerData(ticker):
  """
  Returns DataFrame, returns mean (Mu), returns std (Sigma)
  params:
    ticker (str): Cryptocurrency ticker eg: 'ETH', 'BTC'
  """
  df = yf.download(tickers=f'{ticker}-USD', interval= '1H', start='2023-08-23')
  df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
  returns = df['Close'].pct_change(1).dropna()
  mu = np.mean(returns)
  sigma = np.std(returns)

  return df, mu, sigma

def simulatePricePaths(ticker, time_left, df, mu, sigma, price_limit):
  """
  Returns total paths, paths which satisfy a condition
  params:
  ticker (str): Ticker you are analyzing
  time_left (int): Time left until the bet ends
  df (DataFrame): Desired ticker dataframe
  mu (float): Mean of the desired ticker returns
  sigma (float): Std deviation of the desired ticker returns
  price_limit (float): Price filter e.g. BTC > 60000 -> This is price_limit
  """
  #Amount of desired iterations
  N = 1000
  #Time left to trade
  T = time_left
  paths = []
  #Last Price
  S0= df['Close'].iloc[-1]
  #Montecarlo loop
  for i in range(N):
    path = [S0]
    for i in range(T):
      dt=1
      Z= np.random.normal()
      P_t = path[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
      path.append(P_t)
    paths.append(path)

  paths = np.array(paths)

  amount_paths =sum(paths[:,-1] > price_limit) / len(paths)

  return amount_paths, paths


def plotPaths(paths):
    """
    Returns a plot of N paths based on the Montecarlo probability
    params:
        paths (np.array): Simulated price paths from the function simulatePricePaths
    """
    plt.plot(paths.T, alpha=0.6)

    plt.title("Price Paths")
    plt.xlabel("Time Step (Hours)")
    plt.ylabel("Price")

    plt.show()