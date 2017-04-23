import os
import pandas as pd
import ML4T_Ex1_1 as ml1
import copy
import math
from importlib import reload

# http://quantsoftware.gatech.edu/CS7646_Fall_2016
# http://quantsoftware.gatech.edu/MC2-Project-1

class MarketSimulator(object):
    def __init__(self, orders_file, start_val, use_test=False):
        # Get file with list of orders
        self.orders_file = os.getcwd() + orders_file
        self.start_val = start_val
        self.orders_data = pd.read_csv(self.orders_file, index_col='Date', parse_dates=True)
        # Sort Order data
        self.orders_data.sort_index(axis=0, inplace=True)
        # Get start and end date
        self.start_date = self.orders_data.iloc[0].name
        #self.start_date = datetime.strptime(self.start_date , '%Y-%m-%d').date()
        self.end_date = self.orders_data.iloc[-1].name
        #self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d').date()
        # Get symbols that will be used
        self.symbols = list(set(self.orders_data['Symbol']))
        # Get real data for those symbols over that time period
        if (use_test):
            data_file = os.getcwd() + '\\Machine Learning for Trading\\AAPL-Test.csv'
            self.stock_data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
            throw_away, self.spy = ml1.get_data(self.symbols, self.start_date, self.end_date)
        else:
            self.stock_data, self.spy = ml1.get_data(self.symbols, self.start_date, self.end_date)
        # Setup portvals as a dataframe with dates as index and a value
        self.portvals = pd.DataFrame(data=0, index=self.stock_data.index,columns=['Value'] )
        # Initialize portfolio
        self.portfolio = {}
        self.portfolio['CASH'] = start_val

    def simulate_market(self, gen_plot=False):
        # Step through each day the market is open from start_date to end_date
        for date, values in self.portvals.iterrows():
            # If there are are buy/sell orders on this day, perform them
            today_orders = self.orders_data[self.orders_data.index == date]
            if not today_orders.empty:
                self.transact(today_orders)

            # Now calculate values for the day
            stock_prices = self.stock_data.loc[date]
            self.compute_total(stock_prices, date)

        # print results
        cr, adr, sddr, sr, ev = compute_portfolio_stats(self.portvals)
        cr2, adr2, sddr2, sr2, ev2 = compute_portfolio_stats(self.spy)
        print("")
        print("**********")
        print("Start Date: " + str(self.start_date))
        print("End Date: " + str(self.end_date))
        print("Symbols: " + str(self.symbols))
        print("Sharpe Ratio: " + str(sr) + " vs SPY: " + str(sr2))
        print("Cumulative Return: " + str(cr) + " vs SPY: " + str(cr2))
        print("Volatility (stdev of daily returns): " + str(sddr) + " vs SPY: " + str(sddr2))
        print("Average Daily Return: " + str(adr) + " vs SPY: " + str(adr2))
        print("Final Portfolio Value: " + str(ev))
        print("**********")

        # Create Plots
        if gen_plot:
            # Normalzied Portfolio returns vs SPY
            df_compare = self.portvals.join(self.spy)
            df_port_norm = df_compare / df_compare.ix[0]
            ml1.plot_data(df_port_norm, title='Portfolio vs SPY', ylabel='Return')

    def compute_total(self, stock_prices, date):
        total_val = sum_stocks(self.portfolio, stock_prices, take_abs=True)
        self.portvals.loc[date]['Value'] = sum_portfolio(self.portfolio, stock_prices)



    def transact(self, orders):
        # Buy or Sell
        if len(orders) > 1:
            new_order = orders.iloc[0:0]
            for index, order in orders.iterrows():
                new_order.loc[index] = order
                #empty_order.append(order)
                self.sub_transact(new_order)
        else:
            self.sub_transact(orders)


    def sub_transact(self, order):
        if not (type(order) == pd.DataFrame and len(order) == 1) :
            raise Exception("sub_transact can only take a single order or type DataFrame.")

        # Create copy of portfolio to attempt a buy or sell to check leverage
        temp_portfolio = copy.deepcopy(self.portfolio)
        # Get date of order
        order_date = order.iloc[0].name
        #order_date = datetime.strptime(order_date, '%Y-%m-%d').date()
        # Get stock data for date of order
        stock_prices = self.stock_data.loc[order_date]
        # Get the symbol to buy or sell
        symbol = order.iloc[0]['Symbol']
        # Get # of shares
        shares = order.iloc[0]['Shares']
        order_type = order.iloc[0]['Order']

        if order_type == 'BUY':
            temp_portfolio['CASH'] -= stock_prices[symbol] * shares
            current_amt = temp_portfolio.get(symbol, 0)
            temp_portfolio[symbol] = current_amt + shares
        elif order_type == 'SELL':
            temp_portfolio['CASH'] += stock_prices[symbol] * shares
            current_amt = temp_portfolio.get(symbol, 0)
            temp_portfolio[symbol] = current_amt - shares

        # Check leverage of proposed trade
        # leverage = (sum(abs(all stock positions))) / (sum(all stock positions) + cash)
        leverage = sum_stocks(temp_portfolio, stock_prices, take_abs=True) / (sum_stocks(temp_portfolio, stock_prices, take_abs=False) + temp_portfolio['CASH'])
        # Only perform the trade if leverage is under 1.5
        if leverage < 1.5:
            self.portfolio = temp_portfolio

"""
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # TODO: Your code here
    return portvals
"""
def compute_portvals(orders_file = "\\Machine Learning for Trading\\orders-01.csv", start_val = 1000000):
    port_sim = MarketSimulator(orders_file, start_val)
    port_sim.simulate_market(gen_plot=True)

    portvals = port_sim.portvals
    #print(port_sim.stock_data)

    return portvals


def sum_stocks(temp_portfolio, stock_prices, take_abs=False):
    sum = 0
    for symbol, shares in temp_portfolio.items():  # iteritems in Python 2
        if symbol == 'CASH':
            continue
        value = shares * stock_prices[symbol]
        if (take_abs):
            sum += abs(value)
        else:
            sum += value
    return sum



def sum_portfolio(temp_portfolio, stock_prices):
    return temp_portfolio['CASH'] + sum_stocks(temp_portfolio, stock_prices)



def compute_portfolio_stats(df_value, rfr = 0.0, sf = 252.0):
    # Normalize data
    df_norm = df_value / df_value.ix[0]
    # Sum each row (i.e. all position values for each day). That is your daily portfolio value.
    # Use axis 1 because that means we are going to sum across the columns *for every row*
    df_port_val = df_value.sum(axis=1)
    # Normalize again to make it easy to get stats
    df_portolio_norm = df_port_val / df_port_val.ix[0]
    # Turn into daily returns
    df_port_dr =  ml1.compute_daily_returns(df_portolio_norm)[1:]


    # Cumulative Return
    cr = (df_port_val.ix[-1] / df_port_val.ix[0]) - 1.0
    # Average daily return
    adr = df_port_dr.mean()
    # Standard deviation of daily return
    sddr = df_port_dr.std()
    # Sharpe Ratio
    df_ret_minus_rfr = (df_port_dr - rfr)
    sr = df_ret_minus_rfr.mean() / df_ret_minus_rfr.std()
    # Adjust sharpe ratio from frequence given to annual
    sr = sr * math.sqrt(sf)
    if (adr/sddr) * math.sqrt(sf) != sr:
        raise Exception("Sharpe doesn't match!")
    # end value
    ev = df_port_val[-1]

    return cr, adr, sddr, sr, ev




result = compute_portvals(orders_file='\\Machine Learning for Trading\\orders-01.csv')
print("")
print("Portfolio Results:")
print(result)
# {'CASH': 1062782.0728, 'IBM': 0, 'XOM': 0, 'AAPL': 20, 'GOOG': 0}
