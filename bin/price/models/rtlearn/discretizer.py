"""
Class for discretizing features to be used in modeling.
This class will discretize features in the following ways:

    1. Binning: 
        - Bins continuous data into intervals. 
    2. KMeans: 
        - Here we will fit a Kmeans model to each of the features in the dataset. 
        - The number of clusters will be optimally chosen using shillouette score.
        - The cluster labels will be used as the new features.
    


"""

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class Discretizer:
    def __init__(self, data=None):
        self.data = data
        
    def bins(self, data, n_bins = 5, strategy = 'uniform', encode= 'ordinal'):
        """
        Bins continuous data into intervals. 
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to be binned. 
        n_bins : int
            Number of bins to use. 
        strategy : str
            Strategies include: 
                1. 'uniform': For equal-width bins
                2. 'quantile': For equal size bins, (each has the same number of points)
                3. 'kmeans': Values in the bin have the same nearest center of 1d kmeans cluster.        
        encode : str
            Encoding method. 
            Options include:
                1. 'ordinal': Integer encoding, returns bin id as integer
                2. 'onehot': One hot encoding -> sparse matrix
                3. 'onehot-dense': One hot encoding with dense output. -> dense array
        Returns
        -------
        pd.DataFrame
            Binned data. 
        """
        # Initialize discretizer
        discretizer = KBinsDiscretizer(
            n_bins = n_bins,
            encode = encode, 
            strategy = strategy
        )
        # Fit and transform data
        data_binned = discretizer.fit_transform(data)    
        return pd.DataFrame(data_binned, columns = data.columns, index = data.index)
    
    def _kmean(self, data):
        """
        Fits a 1d Kmeans model
        Finds the optimal number of clusters using the silhouette score.
        Returns the cluster labels for the feature
        """
        try: 
            scores = []
            # iterate through number of clusters 
            for i in range(2, 8):
                # Kmeans model
                kmeans = KMeans(n_clusters=i, random_state=0)
                kmeans.fit(data)
                # Score
                score = silhouette_score(data, kmeans.labels_)
                scores.append((i, score))
            
            # Optimal number of clusters
            best = max(scores, key = lambda x: x[1])
            mod = KMeans(n_clusters=best[0], random_state=0).fit(data)
            return mod.labels_
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def kmeans(self, data, state_vector = False):
        """ 
        Fits the Kmeans model to each of the features in the dataset. 
        Returns the discretized dataset
        """
        scaler = StandardScaler()
        cols = list(data.columns)
        fs = pd.DataFrame(scaler.fit_transform(data), columns = cols, index = data.index)
        pbar = tqdm(cols, desc = "KMeans:")
        states = {x: self._kmean(fs[[x]]) for x in pbar}
        states = pd.DataFrame(states, index = fs.index)
        if state_vector == True: 
            states['state'] = states.apply(lambda x: ''.join([str(i) for i in x]), axis = 1)
        return states
        
        
        
if __name__ == "__main__":
    print(""" 7.4: Earth, water, fire, air, ether, mind, spiritual intelligence and false ego; thus these are the eightfold divisions of my external energy.\n""")

    import sys
    sys.path.append("/Users/jerald/Documents/Dir/Python/Stocks")
    from jetaa.sat.indicators import Indicators
    from bin.main import Manager 
    M = Manager()
    prices = M.Pricedb.ohlc('pfe')
    df = Indicators(prices).indicator_df()["2023-01-01":].drop(columns = ['Close'])
    
    d = Discretizer()
    print("Binned:", '----'*10)
    print(d.bins(df))
    print("Kmean:", '----'*10)
    print(d.kmeans(df))