import datetime as dt
import pandas_datareader.data as pdr
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt


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
    dates = pd.date_range(sd, ed)
    df = pd.DataFrame(index=dates)

    if 'SPY' not in syms:  # add SPY for reference, if absent
        syms.insert(0, 'SPY')

    for symbol in syms:
        data = pdr.DataReader(symbol, 'yahoo', sd, ed)
        df = df.join(data['Adj Close'])
        df = df.rename(columns={'Adj Close': symbol})

    # Fill in missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    # Show plot
    plot_data(df)

    df_temp = df / df.ix[0]
    plot_data(df_temp)

    # Gather Stats
    cr = df_temp.ix[-1] - df_temp.ix[0]
    print(cr)
    #return cr, adr, sddr, ev


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


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


"""
cr, adr, sddr, sr, ev = assess_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1),
    syms=['GOOG','AAPL','GLD','XOM'],
    allocs=[0.1,0.2,0.3,0.4],
    sv=1000000, rfr=0.0, sf=252.0,
    gen_plot=False)
"""

# Turn on interactive mode for matplotlib
plt.ion()

#cr, adr, sddr, sr, ev =
assess_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), syms=['GOOG','AAPL','GLD','XOM'],
    allocs=[0.1,0.2,0.3,0.4], sv=1000000, rfr=0.0, sf=252.0, gen_plot=False)
