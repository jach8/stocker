''' Contains the code for the regression Decision Tree Class. '''

import numpy as np 


class DTLearner:
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size 
        self.verbose = verbose
        self.tree = None

    def author(self):
        return 'jachaibar3'
    
    def add_evidence(self, data_x, data_y):
        self.X = data_x
        self.y = data_y[:, np.newaxis]
        self.data = np.concatenate((self.X, self.y), axis = 1)
        self.tree = self.build_tree(self.data)

    def feature_selection(self, data):
        # Return the column index of the highest correlated feature. 
        # Transpose to get variables as rows. (see np.corcoef doc)
        # [:-1, -1]: -1 gives the last col. in correl matrix, :-1 excludes corr of y and itself (1). 
        # argmax returns the column index of the max correlation
        return np.abs(np.corrcoef(data.T))[:-1, -1].argmax()

    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size: 
            stop1 = np.array([['Leaf', data[0, -1], np.nan, np.nan]], dtype = object)
            if self.verbose: print(f'S1: Leaf found: {stop1}')
            return stop1
        if np.unique(data[:, -1]).shape[0] <= 1:
            stop2 = np.array([['Leaf', data[0, -1], np.nan, np.nan]], dtype = object)
            if self.verbose: print(f'S2: Leaf found: {stop2}')
            return stop2

        else:
            x_ind = self.feature_selection(data)
            split_val = np.median(data[:, x_ind]).astype(float)
            if self.verbose: print(f'X vals: {data[:, x_ind][:5]} Split val: {split_val}')
            if np.max(data[:, x_ind]) == split_val:
                stop3 = np.array([['Leaf', np.mean(data[:, -1]), np.nan, np.nan]], dtype = object)
                if self.verbose: print(f'S3: Leaf found: {stop3}')
                return stop3
            left_tree = self.build_tree(data[data[:, x_ind] <= split_val])
            right_tree = self.build_tree(data[data[:, x_ind] > split_val])
            root = np.array([[x_ind, split_val, 1, left_tree.shape[0] + 1]])
            if self.verbose: print(f'Var Entry: {root}\n')
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
                if self.verbose: print(f'Feature: {self.tree[x, 0]} Split val: {self.tree[x, 1]}, test val: {j[int(self.tree[x, 0])]}')
                if j[int(self.tree[x, 0])] <= float(self.tree[x, 1]):
                    if self.verbose: print('Left branch')
                    x += int(self.tree[x, 2])
                    if self.verbose: print(f'Right branch: {x}')
                else:
                    x += int(self.tree[x, 3])
            if self.verbose: print(f'Leaf found: {self.tree[x, 1]}')
            pred[i] = self.tree[x, 1]
        return pred