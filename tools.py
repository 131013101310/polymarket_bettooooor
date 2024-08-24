from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pytz
from arch import arch_model


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


def simulatePricePaths_GARCH(ticker, time_left, df, price_limit):
  """
  Returns total paths, paths which satisfy a condition using GARCH(1,1) model
  params:
  ticker (str): Ticker you are analyzing
  time_left (int): Time left until the bet ends
  df (DataFrame): Desired ticker dataframe
  price_limit (float): Price filter e.g. BTC > 60000 -> This is price_limit
  """
  # Prepare returns data
  returns = df['Close'].pct_change().dropna()

  # Fit GARCH(1,1) model
  model = arch_model(returns, vol='GARCH', p=1, q=1)
  results = model.fit(disp='off')

  # Extract parameters
  omega = results.params['omega']
  alpha = results.params['alpha[1]']
  beta = results.params['beta[1]']

  # Amount of desired iterations
  N = 1000
  # Time left to trade
  T = time_left
  paths = []
  # Last Price
  S0 = df['Close'].iloc[-1]

  # Last volatility (you might want to use a more sophisticated method to estimate this)
  last_vol = returns.std()

  # Montecarlo loop
  for _ in range(N):
    path = [S0]
    vol = last_vol
    returns_list = returns.tolist()
    for _ in range(T):
      # Update volatility
      vol = np.sqrt(omega + alpha * returns_list[-1] ** 2 + beta * vol ** 2)

      # Generate return
      r = np.random.normal(0, vol)

      # Update price
      P_t = path[-1] * (1 + r)
      path.append(P_t)

      # Update returns for next iteration
      returns_list.append(r)

    paths.append(path)

  paths = np.array(paths)

  amount_paths = sum(paths[:, -1] > price_limit) / len(paths)

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