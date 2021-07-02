# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:22:06 2021

@author: Swarnendub
"""

from pykalman import KalmanFilter
import numpy as np
import pandas as pd
from dateutil.parser import parse


#Used in clustering test
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, OPTICS
import itertools

#Used in ADF test
import warnings
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def KFSmoother(prices):
    """Estimate rolling mean"""

    kf = KalmanFilter(transition_matrices=np.eye(1),
                      observation_matrices=np.eye(1),
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.05)

    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.flatten(),
                     index=prices.index)


def KFHedgeRatio(x, y):
    """Estimate Hedge Ratio"""
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                      initial_state_mean=[0, 0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)

    state_means, _ = kf.filter(y.values)
    return -state_means


def estimate_half_life(spread):
    X = spread.shift().iloc[1:].to_frame().assign(const=1)
    y = spread.diff().iloc[1:]
    beta = (np.linalg.inv(X.T @ X) @ X.T @ y).iloc[0]
    halflife = int(round(-np.log(2) / beta, 0))
    return max(halflife, 1)


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 50)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def half_life_pair(data, pair):
    prices = data[pair].apply(KFSmoother)

    prices['hedge_ratio'] = KFHedgeRatio(y=prices[pair['y']], x=prices[pair['x']])[:, 0]
    prices['spread'] = prices[pair['y']].add(prices[pair['x']].mul(prices.hedge_ratio))
    half_life = estimate_half_life(prices.spread)

    return half_life

#Check if any stock in a pairs has earnings on the day
def check_result(pairname, df, current_date):
    current_date = parse(current_date).strftime('%Y-%m-%d')
    stockY = pairname.split('_')[0]
    stockX = pairname.split('_')[1]
    resultY = df[stockY][current_date]
    resultX = df[stockX][current_date]
    
    if resultY or resultX:
        return 0
    else:
        return 1
    
    
#Half life for pairs available for that period. This half life will be used to prioritize opening of stocks
def half_life_candidates(data, all_pairs, stocks_to_avoid):
    half_life_df = pd.DataFrame(columns=all_pairs['y'] + '_' + all_pairs['x'])
    data = data.fillna(method = 'bfill')
    
    for j in range(len(all_pairs)):
        pair = all_pairs.loc[j][['y', 'x']]

        if (pair['y'] in stocks_to_avoid) or (pair['x'] in stocks_to_avoid) or data[pair].isnull().sum().any():
            half_life_df.loc[0, pair['y'] + '_' + pair['x']] = 9999999999
        else:
            half_life_df.loc[0, pair['y'] + '_' + pair['x']] = half_life_pair(data, pair)
            #half_life_df.loc[0, pair['y'] + '_' + pair['x']] = 99999
            
    return half_life_df

#Filter the penny stocks and bond/treasury ETF stocks
def stock_filter(df):
    penny_stocks = []
    delisted = []
    bond_stocks = ['SHV','BIL','USFR','BSV','FTSM','MINT','JPST','HYS','SHYG','SJNK','VYM','JNK','GOVT']

    for stock in df.columns:
        if (len(np.where(df[stock] < 5)[0]) / len(df)) > 0.5:
            penny_stocks.append(stock)

    penny_stocks.remove('rebalance')
    penny_stocks.extend(bond_stocks)
    return penny_stocks


#This function keeps the delist date in check with respect to the current date
def delist_stock(df, stockY, stockX, delist_data):
    delist_data['DSCD'] = delist_data.DSCD.astype(str)

    index_Y = np.where(delist_data.DSCD == stockY)[0]
    index_X = np.where(delist_data.DSCD == stockX)[0]

    if len(index_Y):
        delist_date = delist_data['Delist Date'][index_Y[-1]]
        if parse(delist_date) <= parse(df.index[-1]):
            return 1

    if len(index_X):
        delist_date = delist_data['Delist Date'][index_X[-1]]
        if parse(delist_date) <= parse(df.index[-1]):
            return 1
        else:
            return 0

    if len(index_Y) == 0 or len(index_X) == 0:
        return 0

    
def check_behaviour(hurst, stockY, stockX, current_date):
    #0.45 is taken because sometimes the 0.45 is acting as a trending market as compared to 0.5 (check analysis.ipynb file)
    
    if (hurst[stockY][current_date] > 0.45) or (hurst[stockX][current_date] > 0.45):
        return 0 #Trending
    else:
        return 1 #Mean Reverting or Random walk
    
    
#ADF cointegration test
def test_cointegration(stocks, possible_pairs, lookback=2):
    #test_start = '2016-01-01'
    #test_end = '2016-12-31'
    results = []

    
    # possible_pairs = pairs
    #stock_data = stocks.loc[test_start:test_end]
    stock_data = stocks.copy()
    test_end  = stocks.index[-1]
    
    for pair in possible_pairs:
        s1 = pair.split('_')[0]
        s2 = pair.split('_')[1]
        df = stock_data[[s1, s2]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            var = VAR(df)
            try:
                lags = var.select_order()
                order = lags.selected_orders['aic']
            except:
                #For some stocks price was same throughout the period...for these kind of data, VAR gives error, so order 1 is taken.
                lags = 'NA'
                order = 1
            result = [test_end, s1, s2]

            result += [coint(df[s1], df[s2], trend='c')[1], coint(df[s2], df[s1], trend='c')[1]]
        try:
            cj = coint_johansen(df, det_order=0, k_ar_diff=order)
            #lr1 is the trace statistics, l2 is maximum eigenvalue statistic, cj.evec is the eigenvectors of VECM coeff matrix
            result += (list(cj.lr1) + list(cj.lr2) + list(cj.evec[:, cj.ind[0]]))
        except:
            #Same price throughout the data, for this Johansen Test gives error
            result += (list([0,0]) + list([0,0]) + list([0,0]))

        results.append(result)
    return results

#Selecting pairs based on Johansen and Eigen tests that we ran.
def select_candidate_pairs(data):
        candidates = data[data.joh_sig | data.eg_sig]
        
        if candidates.empty:
            return candidates
        else:
            candidates['y'] = candidates.apply(lambda x: x.s1 if x.s1_dep else x.s2, axis=1)
            candidates['x'] = candidates.apply(lambda x: x.s2 if x.s1_dep else x.s1, axis=1)
            return candidates.drop(['s1_dep', 's1', 's2'], axis=1)
    
#Calculate zscore in the closing of previous quarter loop
def calculate_zscore(df1, stockY, stockX):
    prices = df1.apply(KFSmoother)
            
    prices['hedge_ratio'] = KFHedgeRatio(y = prices[stockY], x = prices[stockX])[:,0]
    prices['spread'] = prices[stockY].add(prices[stockX].mul(prices.hedge_ratio))
    half_life = estimate_half_life(prices.spread)
    max_window = len(prices.index)
    spread = prices.spread.rolling(window=min(2 * half_life, max_window))
    
    prices['z_score'] = prices.spread.sub(spread.mean()).div(spread.std())
    
    return prices['z_score'].iloc[-1]
    
    
    
#Build pairs for each quarter
def pairs_building(df):
    returns = df.dropna(axis=1).pct_change()
    
    #Replace infinite values with 0
    returns = returns.replace([np.inf, -np.inf], 0)
    
    #Drop those columns with all nan values
    returns = returns[1:].dropna(axis = 1, how = 'all')
    
    #Drop those column where there is even single 1 nan values
    nan_values = returns.columns[np.where(returns.isnull().sum() > 0)].values
    returns = returns.drop(nan_values, axis = 1)
    
    #clustering start
    
    # compute PCA on returns matrix with 90% of variance retained
    pca = PCA(n_components=0.9)
    pca.fit(returns)
    #print(pca.components_.T.shape)
    
    # standardize data
    X = preprocessing.StandardScaler().fit_transform(pca.components_.T)
    
    # check results for DBScan
    for val in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1,1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3]:
        clf = DBSCAN(eps=val, min_samples=2)
    
        clf.fit(X)
        labels = clf.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # print(f"\nClusters discovered: {n_clusters_} for val {val}")
    
        clustered = clf.labels_
    
    # fit OPTICS
    clustering = OPTICS(min_samples=2).fit(X)
    
    n_clusters_ = len(set(clustering.labels_)) - (1 if -1 in labels else 0)
    clustered = clustering.labels_
    
    clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
    clustered_series_all = pd.Series(index=returns.columns, data=clustered.flatten())
    clustered_series = clustered_series[clustered_series != -1]
    
    
    pairs = []

    for clust in clustered_series.unique():
        tickers = list(clustered_series[clustered_series==clust].index)
        all_comb = list(itertools.combinations(tickers,2))
        
        #Create all pairs within a cluster with more than 2 stocks
        for i in all_comb:
            pairs.append(i[0] + '_' + i[1])
            
    clustered_pairs = pd.DataFrame(columns = ['Pairs'])
    clustered_pairs['Pairs'] = pairs
    
    #clustering ends
    
    #Cointegration
    result1 = test_cointegration(df, pairs)
    
    
    columns = ['test_end', 's1', 's2', 'eg1', 'eg2',
           'trace0', 'trace1', 'eig0', 'eig1', 'w1', 'w2']

    critical_values = {0: {.9: 13.4294, .95: 15.4943, .99: 19.9349},
                       1: {.9: 2.7055, .95: 3.8415, .99: 6.6349}}
    
    trace0_cv = critical_values[0][.95] # critical value for 0 cointegration relationships
    trace1_cv = critical_values[1][.95] # critical value for 1 cointegration relationship
    
    all_tests = []
    all_tests.append(pd.DataFrame(result1, columns=columns))
    test_results = pd.concat(all_tests)
    
    # extract results that are significant from Johansen
    test_results['joh_sig'] = ((test_results.trace0 > trace0_cv) &
                               (test_results.trace1 > trace1_cv))
    
    test_results['eg'] = test_results[['eg1', 'eg2']].min(axis=1)
    test_results['s1_dep'] = test_results.eg1 < test_results.eg2
    # extract results that are significant from Englage-Granger...Less than 0.05 in either of eg1 and eg2
    test_results['eg_sig'] = (test_results.eg < .05)
    
    # select pairs that are significant from both methods
    test_results['coint'] = (test_results.eg_sig & test_results.joh_sig)
    test_results.coint.value_counts(normalize=True)
    
    

    candidates = select_candidate_pairs(test_results)
    
    
    if candidates.empty:
        print('Availabe stocks: ', returns.shape[0], ' Selected pairs: ', candidates.shape[0])
        return candidates
    else:
        
        best_candidates = candidates[candidates['coint'] == True]
        best_candidates = best_candidates.reset_index(drop = True)
        print('Availabe stocks: ', returns.shape[0], ' Selected pairs: ', best_candidates.shape[0])
        return best_candidates