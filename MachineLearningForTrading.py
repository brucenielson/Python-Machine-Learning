import datetime as dt


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



    return cr, adr, sddr, ev


"""
cr, adr, sddr, sr, ev = assess_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1),
    syms=['GOOG','AAPL','GLD','XOM'],
    allocs=[0.1,0.2,0.3,0.4],
    sv=1000000, rfr=0.0, sf=252.0,
    gen_plot=False)
"""

