import os
import pandas as pd
import ML4T_Ex1_1 as ml1
from datetime import datetime
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
    # Get start and end date
    sd = orders_data.iloc[0]['Date']
    sd = datetime.strptime(sd, '%Y-%m-%d').date()
    ed = orders_data.iloc[-1]['Date']
    ed = datetime.strptime(ed, '%Y-%m-%d').date()
    # Get symbols that will be used
    syms = list(set(orders_data['Symbol']))
    # Get real data for those symbols over that time period
    stock_data, spy = ml1.get_data(syms, sd, ed)

    # Run simulation
    cash = start_val
    portfolio = {}
    for index, order in orders_data.iterrows():
        order_date = order['Date']
        order_date = datetime.strptime(order_date, '%Y-%m-%d').date()
        stock_prices = stock_data.loc[order_date]
        symbol = order['Symbol']
        shares = order['Shares']
        if order['Order'] == 'BUY':
            cash -= stock_prices[symbol] * shares
            current_amt = portfolio.get(symbol, 0)
            portfolio[symbol] = current_amt + shares
        elif order['Order'] == 'SELL':
            cash += stock_prices[symbol] * shares
            current_amt = portfolio.get(symbol, 0)
            portfolio[symbol] = current_amt - shares

    portvals = portfolio

    return portvals


compute_portvals()