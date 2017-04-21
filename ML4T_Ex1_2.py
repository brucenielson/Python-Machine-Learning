import ML4T_Ex1_1 as ex1
import datetime as dt
import scipy.optimize as spo
import numpy as np


def compute_portfolio_sharpe(allocs, df_prices, rfr = 0.0, sf = 252.0, sv=1000000):
    cr, adr, sddr, sr, ev = ex1.compute_portfolio_stats(df_prices['df_prices'], allocs=allocs, rfr = rfr, sf = sf, sv=sv)
    return -sr



"""
Optimize Function
allocs, cr, adr, sddr, sr = \
    optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False)
"""
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    df_prices, spy = ex1.get_data(syms, sd, ed)
    num_symbols = len(syms)
    first_guess = []
    bounds = []
    for i in range(0,num_symbols):
        first_guess.append(1.0/num_symbols)
        bounds.append((0.0, 1.0))
    allocs = np.array(first_guess)
    parameters = dict(df_prices=df_prices, rfr = 0.0, sf = 252.0, sv=1000000)
    constraints = ({'type': 'eq', 'fun': constrain})
    min_result = spo.minimize(compute_portfolio_sharpe, allocs, args=(parameters), method='SLSQP', options={'disp': False}, constraints=constraints, bounds=bounds)
    print("Min Result: " + str(min_result))
    return min_result

def constrain(x):
    result = np.sum(x) - 1.0
    return result




#min_result = optimize_portfolio()

"""
Start Date: 2010-01-01
End Date: 2010-12-31
Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
Optimal allocations: [  5.38105153e-16   3.96661695e-01   6.03338305e-01  -5.42000166e-17]
Sharpe Ratio: 2.00401501102
Volatility (stdev of daily returns): 0.0101163831312
Average Daily Return: 0.00127710312803
Cumulative Return: 0.360090826885
"""
sd=dt.datetime(2010, 1, 1)
ed=dt.datetime(2010, 12, 31)
syms=['GOOG', 'AAPL', 'GLD', 'XOM']
min_result = optimize_portfolio(sd=sd, ed=ed, syms=syms)

print("")
print("Example 1:")
ex1.assess_portfolio(sd=sd, ed=ed, syms=syms,
    allocs=min_result['x'], sv=1000000, rfr=0.0, sf=252.0, gen_plot=True)



"""
Start Date: 2004-01-01
End Date: 2006-01-01
Symbols: ['AXP', 'HPQ', 'IBM', 'HNZ']
Optimal allocations: [  7.75113042e-01   2.24886958e-01  -1.18394877e-16  -7.75204553e-17]
Sharpe Ratio: 0.842697383626
Volatility (stdev of daily returns): 0.0093236393828
Average Daily Return: 0.000494944887734
Cumulative Return: 0.255021425162
"""
print("")
print("Example 2:")


"""
Start Date: 2004-12-01
End Date: 2006-05-31
Symbols: ['YHOO', 'XOM', 'GLD', 'HNZ']
Optimal allocations: [ -3.84053467e-17   7.52817663e-02   5.85249656e-01   3.39468578e-01]
Sharpe Ratio: 1.5178365773
Volatility (stdev of daily returns): 0.00797126844855
Average Daily Return: 0.000762170576913
Cumulative Return: 0.315973959221
"""
print("")
print("Example 3:")


"""
Start Date: 2005-12-01
End Date: 2006-05-31
Symbols: ['YHOO', 'HPQ', 'GLD', 'HNZ']
Optimal allocations: [ -1.67414005e-15   1.01227499e-01   2.46926722e-01   6.51845779e-01]
Sharpe Ratio: 3.2334265871
Volatility (stdev of daily returns): 0.00842416845541
Average Daily Return: 0.00171589132005
Cumulative Return: 0.229471589743
"""
print("")
print("Example 4:")
