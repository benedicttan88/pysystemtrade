# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:42:03 2025

@author: benedicttan
"""

import pandas as pd
from matplotlib.pyplot import show

## Data
from sysdata.sim.csv_futures_sim_data import csvFuturesSimData

## Rules and TradingRules
from systems.forecasting import Rules
from systems.trading_rules import TradingRule
from systems.provided.rules.ewmac import ewmac_forecast_with_defaults as ewmac

## quant
from sysquant.estimators.vol import robust_vol_calc

## System
from systems.basesystem import System

## Account
from systems.accounts.account_forecast import pandl_for_instrument_forecast
import syscore.pandas.strategy_functions

## Config
from sysdata.config.configdata import Config


#%%

def calc_ewmac_forecast(price, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback
    """
    ## price: This is the stitched price series
    ## We can't use the price of the contract we're trading, or the volatility will be jumpy
    ## And we'll miss out on the rolldown. See https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html

    price = price.resample("1B").last()
    if Lslow is None:
        Lslow = 4 * Lfast

    ## We don't need to calculate the decay parameter, just use the span directly

    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma

    vol = robust_vol_calc(price.diff())

    return raw_ewmac / vol


if __name__ == "__main__":
    
    ### Settings
    instrument_code =   'AUD'
    
    ### Data
    data            =   csvFuturesSimData()

    price           =   data.daily_prices(instrument_code)
    ewmac_custom    =   calc_ewmac_forecast(price, 32, 128)
    ewmac_custom.plot()
    show()

    ### Account
    account = pandl_for_instrument_forecast(forecast = ewmac_custom, price=price)
    account.percent.stats()             
    account.sharpe()                                            ## get the Sharpe Ratio (annualised), and any other statistic which is in the stats list
    account.curve().plot()                                      ## plot the cumulative account curve (equivalent to account.cumsum().plot() inicidentally)
    account.percent                                             ## gives a % curve
    # syscore.pandas.strategy_functions.drawdown().plot()       ## see the drawdowns as a percentage
    account.weekly                                              ## weekly returns (also daily [default], monthly, annual)
    account.gross.ann_mean()                                    ## annual mean for gross returns, also costs (there are none in this simple example)


#%%

    ### Rules and Trading Rules
    # my_rules=Rules(ewmac)
    # my_rules.trading_rules()

    my_rules    =   Rules(dict(ewmac=ewmac))
    my_rules.trading_rules()
    
#%%
    # ewmac_rule  =   TradingRule(ewmac)
    # my_rules    =   Rules(dict(ewmac=ewmac_rule))

    ewmac_8     =   TradingRule((ewmac, [], dict(Lfast=8, Lslow=32))) ## as a tuple (function, data, other_args) notice the empty element in the middle
    ewmac_32    =   TradingRule(dict(function=ewmac, other_args=dict(Lfast=32, Lslow=128)))  ## as a dict
    my_rules    =   Rules(dict(ewmac8=ewmac_8,
                               ewmac32=ewmac_32))
    my_rules.trading_rules()['ewmac32']

    ### System
    my_system   =   System([my_rules],
                           data)
    my_system.rules.get_raw_forecast(instrument_code, "ewmac8").tail(5)
    my_system.rules.get_raw_forecast(instrument_code, "ewmac32").tail(5)


#%%

    ### Config Object
    my_config   =  Config()
    my_config.trading_rules     =   dict(ewmac8=ewmac_8, ewmac32=ewmac_32)

    empty_rules =   Rules()

    my_system   =   System([empty_rules],
                           data,
                           my_config)

    print(my_system.rules.get_raw_forecast("EDOLLAR", "ewmac8"))
    







































