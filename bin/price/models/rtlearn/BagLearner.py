''' Contains the code for the regression Bag learner (i.e. a BagLearner containing Random Trees. ) '''
import numpy as np   
                                     
class BagLearner(object):                                                                       
    def __init__(self,learner, bags, kwargs, verbose = False, boost = False):
        self.learner = learner
        self.verbose = verbose
        self.bags = bags
        self.boost = boost
        self.learner_list = [self.learner(**kwargs) for i in range(self.bags)]                                     
                                        
    def author(self):                                        
        return "jachaibar3"  # replace tb34 with your Georgia Tech username                             
    
    def add_evidence(self, data_x, data_y):
        self.X = data_x
        self.y = data_y[:, np.newaxis]
        self.data = np.concatenate((self.X, self.y), axis = 1)
        # Update learners 
        for i in range(self.bags):
            # train with random subsets 
            index = np.random.choice(np.arange(self.data.shape[0]), self.data.shape[0])
            self.learner_list[i].add_evidence(self.data[index, :-1], self.data[index, -1])

    def query(self, points):
        assert points is not None
        return np.mean(np.array([learner.query(points) for learner in self.learner_list]), axis = 0)

if __name__ == '__main__':
    import pandas as pd 
    import sys
    sys.path.append("/Users/jerald/Documents/Dir/Python/Stocks")
    from models.rtlearn.plugin import data
    from models.rtlearn.RTLearner import RTLearner
    from models.rtlearn.DTLearner import DTLearner
    
    d = data()
    xtrain, ytrain, xtest, ytest = d.split('spy')
    print(f""" xtrain: {xtrain.shape}, ytrain: {ytrain.shape} xtest: {xtest.shape}, ytest: {ytest.shape}""")
    
    learner = DTLearner(leaf_size=20, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(f"RT ACC:\t\t{acc:.2%}")
    
    learner = DTLearner(leaf_size=20, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(f"DT ACC:\t\t{acc:.2%}")
    
    learner = BagLearner(learner=DTLearner, bags=20, kwargs={'leaf_size': 20}, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(f"BAG DT ACC:\t\t{acc:.2%}")

    learner = BagLearner(learner=RTLearner, bags=20, kwargs={'leaf_size': 20}, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(f"BAG RT ACC:\t\t{acc:.2%}")

# ''' Contains the code for the regression Bag learner (i.e. a BagLearner containing Random Trees. ) '''
# import numpy as np   
# from scipy.stats import mode

# class BagLearner(object):                                                                       
#     def __init__(self,learner, bags, kwargs, verbose = False, boost = False):
#         self.learner = learner
#         self.verbose = verbose
#         self.bags = bags
#         self.boost = boost
#         self.learner_list = [self.learner(**kwargs) for i in range(self.bags)]
                                        
#     def author(self):                                        
#         return "jachaibar3"  # replace tb34 with your Georgia Tech username                             
    
#     def add_evidence(self, data_x, data_y):
#         self.X = data_x
#         self.y = data_y[:, np.newaxis]
#         self.data = np.concatenate((self.X, self.y), axis = 1)
#         # Update learners 
#         for i in range(self.bags):
#             # train with random subsets 
#             index = np.random.choice(np.arange(self.data.shape[0]), self.data.shape[0])
#             self.learner_list[i].add_evidence(self.data[index, :-1], self.data[index, -1])

#     def query(self, points):
#         # If Y is categorical; return the mode of the predictions 
#         return np.mean(np.array([learner.query(points) for learner in self.learner_list]), axis = 0)


# """
# Implements a Bootstrap Aggregating Learner
# """

# import numpy as np

# class BagLearner(object):
#     def __init__(self, learner=None, kwargs={}, bags=20, boost=False, verbose=False):
#         self._learner = learner
#         self._kwargs = kwargs
#         self._bags = bags
#         self._boost = boost
#         self._verbose = verbose

#         self._learners = self._create_learners()

#     def author(self):
#         return 'jachaibar3'

#     def add_evidence(self, data_x, data_y):
#         for learner in self._learners:
#             data_x_prime, data_y_prime = self._bootstrap(data_x, data_y)
#             learner.add_evidence(data_x_prime, data_y_prime)

#     def query(self, Xtest):
#         # Make sure the predictions are column-vectors by reshaping them to be of size (N, 1)
#         # and concatenating them horizontally, i.e. via the columns (axis=1)
#         predictions = np.concatenate([learner.query(Xtest).reshape(-1, 1) for learner in self._learners], axis=1)
#         # Aggregate the predictions of all learners, i.e. columns (axis=1) for all samples
#         return np.mean(predictions, axis=1)

#     def _create_learners(self):
#         return [self._learner(**self._kwargs) for _ in range(self._bags)]

#     def _bootstrap(self, data_x, data_y):
#         # Sample N indices from the range(0, N)
#         # See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
#         indices = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True)
#         assert indices.shape[0] == data_x.shape[0]

#         return data_x[indices, :], data_y[indices]