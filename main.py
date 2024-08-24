from tools import *

#Set parameters
year = 2024 #Year the bet closes
month = 8 #Month the bet closes
day = 30 #Day the bet closes
ticker = 'BTC' #Desired ticker
price_limit = 60000 #Bet threshold

#Get all values needed
time_left = getTimeLeft(year=year, month=month, day=day)
df, mu, sigma = getTickerData(ticker=ticker)
amount_paths, paths = simulatePricePaths(ticker=ticker, time_left=time_left, df=df, mu=mu, sigma=sigma, price_limit = price_limit)
plotPaths(paths=paths)
#Print % data
print(f'The probability that {ticker} is worth more than {price_limit} at 00:00 of the date {day}/{month}/{year} is an estimated of: {amount_paths*100}%')

#Calculation of EV and optimal fraction to bet
bias = input('Buy or sell: ').upper()

if bias == 'BUY':
    ganancia = float(input('Possible profit (In percentage): '))
    roi_final = ganancia/100 + 1
    ExpVal = (amount_paths * roi_final) + ((1-amount_paths) * 0)
    win_odds = amount_paths
    f = max(0, min(1, (win_odds) - ((1-win_odds)/(roi_final-1))))
    print(f'This bet has an estimated expected value of: {ExpVal:.4f}')
    print(f'The optimal fraction of your stake you should bet is: {f*100:.2f}% ')
elif bias == 'SELL':
    ganancia = float(input('Possible profit (In percentage): '))
    roi_final = ganancia/100 + 1
    ExpVal = (amount_paths * 0) + ((1-amount_paths) * roi_final)
    win_odds = 1-amount_paths
    f = max(0, min(1, (win_odds) - ((1-win_odds)/(roi_final-1))))
    print(f'This bet has an estimated expected value of: {ExpVal:.4f}')
    print(f'The optimal fraction of your stake you should bet is: {f*100:.2f}% ')
else:
    print('You did not type neither buy nor sell, try again.')