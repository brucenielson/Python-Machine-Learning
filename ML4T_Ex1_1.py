import datetime as dt
import pandas_datareader.data as pdr
import pandas as pd
import matplotlib.pyplot as plt
import math
from importlib import reload

# This is for MC1-Project-1
# http://quantsoftware.gatech.edu/MC1-Project-1
# http://quantsoftware.gatech.edu/CS7646_Fall_2016

def get_data(syms, sd, ed):
    # TODO: In the real class I'll load CSV files and I am not sure adding SPY then dropping it will work with the CSV files or not
    dates = pd.date_range(sd, ed)
    df = pd.DataFrame(index=dates)

    # I always insert SPY at the front even if it's already in the list. This way I can drop it freely afterwards but still be sure
    # all trading days are accounted for
    symbols = syms.copy()
    symbols.insert(0, 'SPY')

    do_once = True
    for symbol in symbols:
        data = pdr.DataReader(symbol, 'yahoo', sd, ed)
        df = df.join(data['Adj Close'])# , how='inner')
        df = df.rename(columns={'Adj Close': symbol})
        if symbol == 'SPY' and do_once:  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
            do_once = False
            df = df.rename(columns={'SPY':'DROP'})

    # Fill in missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Drop the SPY I added
    spy = df['DROP'].copy(deep=True)
    spy = pd.DataFrame(spy,spy.index,dtype=dt.datetime)
    spy = spy.rename(columns={'DROP': 'SPY'})
    df = df.drop('DROP', axis=1)

    return df, spy



def assess_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), syms=['GOOG','AAPL','GLD','XOM'], allocs=[0.1,0.2,0.3,0.4],
    sv=1000000, rfr=0.0, sf=252.0, gen_plot=False):
    """
    Where the returned outputs are:
    cr: Cumulative return
    adr: Average period return (if sf == 252 this is daily return)
    sddr: Standard deviation of daily return
    sr: Sharpe ratio
    ev: End value of portfolio
    
    The input parameters are:
    sd: A datetime object that represents the start date
    ed: A datetime object that represents the end date
    syms: A list of 2 or more symbols that make up the portfolio (note that your code should support any symbol in the data directory)
    allocs: A list of 2 or more allocations to the stocks, must sum to 1.0
    sv: Start value of the portfolio
    rfr: The risk free return per sample period that does not change for the entire date range (a single number, not an array).
    sf: Sampling frequency per year
    gen_plot: If False, do not create any output. If True it is OK to output a plot such as plot.png
    """
    # Adjust risk free rate to a daily risk free rate
    rfr = calc_daily_rfr(rfr)
    # Get the data for the symbols over the time period asked for
    df_prices, spy = get_data(syms, sd, ed)
    # Get Statistics
    cr, adr, sddr, sr, ev = compute_portfolio_stats(df_prices, allocs=allocs, rfr=rfr, sf=sf)
    print("**********")
    print("Start Date: " + str(sd))
    print("End Date: " + str(ed))
    print("Symbols: " + str(syms))
    print("Allocations: " + str(allocs))
    print("Sharpe Ratio: " + str(sr))
    print("Volatility (stdev of daily returns): " + str(sddr))
    print("Average Daily Return: " + str(adr))
    print("Cumulative Return: " + str(cr))
    print("**********")

    # Normalize data
    df_norm = df_prices / df_prices.ix[0]
    # Turn into daily returns
    df_daily = compute_daily_returns(df_prices)
    # Multiply each column by the allocation to the corresponding equity
    # Multiply these normalized allocations by starting value of overall portfolio, to get position values.
    df_value = (df_norm * allocs) * sv
    # Sum each row (i.e. all position values for each day). That is your daily portfolio value.
    # Use axis 1 because that means we are going to sum across the columns *for every row*
    df_daily_portfolio = df_value.sum(axis=1)
    df_daily_portfolio = pd.DataFrame(df_daily_portfolio,df_value.index, ['Portfolio'])
    # Get statistic based on portfolio value, such as...

    # Create Plots
    # TODO: For the class, do the specific plots they ask for. I just did everything I was interested in.
    if gen_plot:
        # Show plots
        # Straight Prices
        """
        plot_data(df_prices, title='Stock Prices', ylabel='Price')
        # Normalized to 1.0
        plot_data(df_norm, title='Stock Returns', ylabel='Return')
        # Daily Returns
        plot_data(df_daily, title='Daily Returns', ylabel='Daily Return')
        # Daily Portfolio Value
        plot_data(df_daily_portfolio, title='Portfolio Value', ylabel='Dollars')
        """
        # Normalzied Portfolio returns vs SPY
        df_compare = df_daily_portfolio.join(spy)
        df_port_norm = df_compare / df_compare.ix[0]
        plot_data(df_port_norm, title='Portfolio vs SPY', ylabel='Return')

    #return cr
    #return cr, adr, sddr, ev


def calc_daily_rfr(rfr):
    rfr = 1+rfr
    rfr_daily = (math.pow(rfr, 252) - 1.0)
    return rfr_daily


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    # pd.options.display.float_format = '{:.10f}'.format
    dr = df.copy(deep=True)
    # Start at second date and work all the way down
    # Note: daily_ret[t] = (price[t]/price[t-1]) - 1
    # i.e. Day X price (start on second day) / Day X-1 (i.e. day before) Price, then adjust to % by subtracting one.
    #  Top doesn't include first day, Bottom doesn't include last day. That way they match # of days and we don't overrunn array.
    #  Because first day can't be included, set to zero, since first day's returns are always zero by definition
    dr[1:] = (df[1:] / df[:-1].values) - 1.0
    if type(dr) == pd.Series:
        dr.ix[0] = 0
    else:
        dr.ix[0, :] = 0

    return dr




def compute_portfolio_stats(df_prices, allocs=[0.1,0.2,0.3,0.4], rfr = 0.0, sf = 252.0, sv=1000000):
    """
    cr, adr, sddr, sr = \
        compute_portfolio_stats(prices = df_prices, \
        allocs=[0.1,0.2,0.3,0.4],\
        rfr = 0.0, sf = 252.0)
        Note: I added ev as end value because, why the heck didn't they include that since they want it returned
    """

    # Check for mismatching data
    if len(df_prices.columns) != len(allocs):
        raise Exception("Dataframe and allocations do not match")
    if sum(allocs) != 1.0:
        pass
        #raise Exception("Allocations must sum to 1.0")

    # Normalize data
    df_norm = df_prices / df_prices.ix[0]
    # Turn into daily returns
    df_daily = compute_daily_returns(df_prices)
    # Multiply each column by the allocation to the corresponding equity
    # Multiply these normalized allocations by starting value of overall portfolio, to get position values.
    df_value = (df_norm * allocs) * sv
    # Sum each row (i.e. all position values for each day). That is your daily portfolio value.
    # Use axis 1 because that means we are going to sum across the columns *for every row*
    df_port_val = df_value.sum(axis=1)
    #df_port_val = pd.DataFrame(df_port_val, df_value.index, ['Dollars'])
    # Normalize again to make it easy to get stats
    df_portolio_norm = df_port_val / df_port_val.ix[0]
    # Turn into daily returns
    df_port_dr =  compute_daily_returns(df_portolio_norm)[1:]


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




def plot_data(df, title="Stock Prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    plt.savefig(title)


"""
def get_data(symbols, dates):
    # Read stock data (adjusted close) for given symbols from CSV files.
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df
"""



def main():
    # Turn on interactive mode for matplotlib
    plt.ion()


    """
    cr, adr, sddr, sr, ev = assess_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1),
        syms=['GOOG','AAPL','GLD','XOM'],
        allocs=[0.1,0.2,0.3,0.4],
        sv=1000000, rfr=0.0, sf=252.0,
        gen_plot=False)
    """

    #cr, adr, sddr, sr, ev =
    cr = assess_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), syms=['GOOG','AAPL','GLD','XOM'],
        allocs=[0.1,0.2,0.3,0.4], sv=1000000, rfr=0.0, sf=252.0, gen_plot=True)


    """
    Example 1
    Start Date: 2010-01-01
    End Date: 2010-12-31
    Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
    Allocations: [0.2, 0.3, 0.4, 0.1]
    Sharpe Ratio: 1.51819243641
    Volatility (stdev of daily returns): 0.0100104028
    Average Daily Return: 0.000957366234238
    Cumulative Return: 0.255646784534
    """
    print("")
    print("Example 1:")
    assess_portfolio(sd=dt.datetime(2010,1,1), ed=dt.datetime(2010,12,31), syms=['GOOG','AAPL','GLD','XOM'],
        allocs=[0.2,0.3,0.4,0.1], sv=1000000, rfr=0.0, sf=252.0, gen_plot=True)

    """
    Start Date: 2010-01-01
    End Date: 2010-12-31
    Symbols: ['AXP', 'HPQ', 'IBM', 'HNZ']
    Allocations: [0.0, 0.0, 0.0, 1.0]
    Sharpe Ratio: 1.30798398744
    Volatility (stdev of daily returns): 0.00926153128768
    Average Daily Return: 0.000763106152672
    Cumulative Return: 0.198105963655
    """
    # TODO: For some reason, the answers he gives and my answers do not match. Can't really check this without his CSV file. But I did a double check and it looks like I'm right
    print("")
    print("Example 2:")
    assess_portfolio(sd=dt.datetime(2010,1,1), ed=dt.datetime(2010,12,31), syms=['AXP', 'HPQ', 'IBM', 'HNZ'],
        allocs=[0.0, 0.0, 0.0, 1.0], sv=1000000, rfr=0.0, sf=252.0, gen_plot=True)
    print("--Double Check--")
    assess_portfolio(sd=dt.datetime(2010,1,1), ed=dt.datetime(2010,12,31), syms=['HNZ'],
        allocs=[1.0], sv=1000000, rfr=0.0, sf=252.0, gen_plot=False)


    """
    Start Date: 2010-06-01
    End Date: 2010-12-31
    Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
    Allocations: [0.2, 0.3, 0.4, 0.1]
    Sharpe Ratio: 2.21259766672
    Volatility (stdev of daily returns): 0.00929734619707
    Average Daily Return: 0.00129586924366
    Cumulative Return: 0.205113938792
    """
    print("")
    print("Example 3:")
    assess_portfolio(sd=dt.datetime(2010,6,1), ed=dt.datetime(2010,12,31), syms=['GOOG','AAPL','GLD','XOM'],
        allocs=[0.2, 0.3, 0.4, 0.1], sv=1000000, rfr=0.0, sf=252.0, gen_plot=True)


