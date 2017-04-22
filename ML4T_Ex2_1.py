import os
import pandas as pd
import ML4T_Ex1_1 as ml1
from datetime import datetime
import copy
from importlib import reload

# http://quantsoftware.gatech.edu/CS7646_Fall_2016
# http://quantsoftware.gatech.edu/MC2-Project-1


"""
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # TODO: Your code here
    return portvals
"""
def compute_portvals(orders_file = "orders-01.csv", start_val = 1000000):

    # Get file with list of orders
    orders_file = os.getcwd() + '\\Machine Learning for Trading\\'+orders_file
    orders_data = pd.read_csv(orders_file)
    # Sort Order data
    orders_data.sort_values('Date',axis=0, inplace=True)
    # Get start and end date
    sd = orders_data.iloc[0]['Date']
    sd = datetime.strptime(sd, '%Y-%m-%d').date()
    ed = orders_data.iloc[-1]['Date']
    ed = datetime.strptime(ed, '%Y-%m-%d').date()
    # Get symbols that will be used
    syms = list(set(orders_data['Symbol']))
    # Get real data for those symbols over that time period
    stock_data, spy = ml1.get_data(syms, sd, ed)
    # Setup portvals as a dataframe with dates as index and a value
    portvals = pd.DataFrame(data=0, index=stock_data.index,columns=['Value'])

    # Run simulation
    portfolio = {}
    portfolio['CASH'] = start_val
    for index, order in orders_data.iterrows():
        # Get date of order
        order_date = order['Date']
        order_date = datetime.strptime(order_date, '%Y-%m-%d').date()
        # Get stock data for date of order
        stock_prices = stock_data.loc[order_date]
        # Get the symbol to buy or sell
        symbol = order['Symbol']
        # Get # of shares
        shares = order['Shares']
        # Buy or Sell
        # Create copy of portfolio to attempt a buy or sell to check leverage
        temp_portfolio = copy.deepcopy(portfolio)
        if order['Order'] == 'BUY':
            temp_portfolio['CASH'] -= stock_prices[symbol] * shares
            current_amt = temp_portfolio.get(symbol, 0)
            temp_portfolio[symbol] = current_amt + shares
        elif order['Order'] == 'SELL':
            temp_portfolio['CASH'] += stock_prices[symbol] * shares
            current_amt = temp_portfolio.get(symbol, 0)
            temp_portfolio[symbol] = current_amt - shares

        # Check leverage of proposed trade
        # leverage = (sum(abs(all stock positions))) / (sum(all stock positions) + cash)
        leverage = sum_stocks(temp_portfolio, stock_prices, take_abs=True) / (sum_stocks(temp_portfolio, stock_prices, take_abs=False) + temp_portfolio['CASH'])
        # Only perform the trade if leverage is under 1.5
        if leverage < 1.5:
            portfolio = temp_portfolio

    portvals = portfolio



    return portvals


def sum_stocks(portfolio, stock_prices, take_abs=False):
    sum = 0
    for symbol, shares in portfolio.items(): # iteritems in Python 2
        if symbol == 'CASH':
            continue
        value = shares * stock_prices[symbol]
        if (take_abs):
            sum += abs(value)
        else:
            sum += value
    return sum


def sum_portfolio(portfolio, stock_prices):
    return portfolio['CASH'] + sum_stocks(portfolio, stock_prices)



result = compute_portvals()
print(result)