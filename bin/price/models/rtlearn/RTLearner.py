''' Contains the code for the regression Random Tree Class '''

import numpy as np
from scipy.stats import mode

class RTLearner:
    def __init__(self, leaf_size = 1, verbose = False):
        ''' 
        Random Tree Learner: 
        A random tree learner is a decision tree learner that splits on a random feature. 
        Leaf Size: The maximum number of samples to be aggregated at a given leaf node. 
        Verbose: Print Out debugging information. 
        '''
        
        self.leaf_size = leaf_size 
        self.verbose = verbose
        self.tree = None

    def author(self):
        return 'jachaibar3'
    
    def add_evidence(self, data_x, data_y):
        """ Builds the Tree given the training data. """
        self.X = data_x
        self.y = data_y[:, np.newaxis]
        self.data = np.concatenate((self.X, self.y), axis = 1)
        self.tree = self.build_tree(self.data)
        
    def _show_current_tree(self):
        """ Pretty Print the current tree. """
        for i in range(self.tree.shape[0]):
            if self.tree[i, 0] == 'Leaf':
                print(f'Leaf: {self.tree[i, 1]}')
            else:
                print(f'Node: {self.tree[i, 0]} Split Val: {self.tree[i, 1]} Left: {self.tree[i, 2]} Right: {self.tree[i, 3]}')

    def feature_selection(self, data):
        # Return random column index
        return np.random.randint(data.shape[1] - 1)

    def build_tree(self, data):
        """ Implements the recursive tree building algorithm. 
                Given a Feature Set X: 
                    1. If the number of samples is less than the leaf size, return a leaf node.
                    2. If the number of unique Y values is less than or equal to 1, return a leaf node.
                    3. If the maximum value of the feature is the same as the split value, return a leaf node.
                    4. Otherwise, split the data into two branches and recursively build the tree.
        """
        if self.verbose: print(f"\nBuilding Tree: Dimensions {data.shape}")
        if data.shape[0] <= self.leaf_size: 
            stop1 = np.array([['Leaf', data[0, -1], np.nan, np.nan]], dtype = object)
            if self.verbose: print(f'\tS1: Leaf found: {stop1}')
            return stop1
        if np.unique(data[:, -1]).shape[0] <= 1:
            stop2 = np.array([['Leaf', data[0, -1], np.nan, np.nan]], dtype = object)
            if self.verbose: print(f'\t\tS2: Leaf found: {stop2}')
            return stop2

        else:
            x_ind = self.feature_selection(data)
            #split_val = mode(data[:, x_ind]).astype(float)
            split_val = np.median(data[:, x_ind]).astype(float)
            if self.verbose: print(f'\t\t\tX vals: {data[:, x_ind][:5]} Split val: {split_val}')
            if np.max(data[:, x_ind]) == split_val:
                stop3 = np.array([['Leaf', mode(data[:, -1]), np.nan, np.nan]], dtype = object)
                if self.verbose: print(f'\t\t\t\tS3: Leaf found: {stop3}')
                return stop3
            left_tree = self.build_tree(data[data[:, x_ind] <= split_val])
            right_tree = self.build_tree(data[data[:, x_ind] > split_val])
            root = np.array([[x_ind, split_val, 1, left_tree.shape[0] + 1]])
            if self.verbose: print(f'\t\t\t\t\tVar Entry: {root}\n')
            return np.append(root, np.append(left_tree, right_tree, axis = 0), axis = 0)
    
    def query(self, points):
        """
        Predict Y given the test set of X. 
        Given X (data points) evaluate the tree to return a leaf value for the prediction of Y. 
        """  
        pred = np.zeros(points.shape[0])
        if self.verbose: print(f'Querying points...')
        for i, j in enumerate(points): # iterate through test features
            if self.verbose: print('Iteration: ', i)
            x = 0 
            while self.tree[x, 0] != 'Leaf':
                if self.verbose: print(f'\tFeature: {self.tree[x, 0]}\n\t\tSplit val: {self.tree[x, 1]}, test val: {j[int(self.tree[x, 0])]}')
                if j[int(self.tree[x, 0])] <= float(self.tree[x, 1]):
                    if self.verbose: print(f'\t\t\tLeft branch: {x}')
                    x += int(self.tree[x, 2])
                    if self.verbose: print(f'\t\t\t\tRight branch: {x}')
                else:
                    x += int(self.tree[x, 3])
                    if self.verbose: print(f'\t\t\t\tRight branch: {x}')
            if self.verbose: print(f'\t\t\t\t\tLeaf found: {self.tree[x, 1]}')
            if type(self.tree[x, 1]) != np.float64:
                pred[i] = self.tree[x, 1][0]
            
            else:
                pred[i] = self.tree[x, 1]
        return pred
    
    
if __name__ == '__main__':
    import pandas as pd 
    import sys
    sys.path.append("/Users/jerald/Documents/Dir/Python/Stocks")
    from models.rtlearn.plugin import data
    
    d = data()
    xtrain, ytrain, xtest, ytest = d.split('spy', start_date = "2023-01-01", discretize_features=True)
    print(f""" xtrain: {xtrain.shape}, ytrain: {ytrain.shape} xtest: {xtest.shape}, ytest: {ytest.shape}""")
    
    learner = RTLearner(leaf_size=20, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    print(d.accuracy(pred, ytest))