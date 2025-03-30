''' Contains the code for the regression Insane Learner of Bag Learners. '''
import numpy as np 
import sys
sys.path.append("/Users/jerald/Documents/Dir/Python/Stocks")
import models.rtlearn.BagLearner as bg
import models.rtlearn.LinRegLearner as lrl
import models.rtlearn.RTLearner as rt

class InsaneLearner(object):
    def __init__(self, verbose = False, bags = 20 ):
        ''' Initialize Insane Learner with 20 Bag Learners '''
        self.learner_list = [bg.BagLearner(lrl.LinRegLearner, kwargs = {}, bags = bags, verbose = verbose) for x in range(bags)]
        #self.learner_list = [bg.BagLearner(rt.RTLearner, kwargs = {}, bags = bags, verbose = verbose) for x in range(bags)]

    def author(self):
        return "jachaibar3"
    
    def add_evidence(self, data_x, data_y):
        for x in self.learner_list: x.add_evidence(data_x, data_y) 
    
    def query(self, points):
        return np.median(np.array([learner.query(points) for learner in self.learner_list]), axis = 0)
    
    

if __name__ == '__main__':
    import pandas as pd 
    import sys
    sys.path.append("/Users/jerald/Documents/Dir/Python/Stocks")
    from models.rtlearn.plugin import data
    from models.rtlearn.RTLearner import RTLearner
    from models.rtlearn.DTLearner import DTLearner
    from models.rtlearn.BagLearner import BagLearner
    
    d = data()
    xtrain, ytrain, xtest, ytest = d.split('spy', discretize_features = True, t = 100)
    print(f""" xtrain: {xtrain.shape}, ytrain: {ytrain.shape} xtest: {xtest.shape}, ytest: {ytest.shape}""")
    
    learner = DTLearner(leaf_size=20, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(d.pred.tail())
    print(f"RT ACC:\t\t{acc:.2%}")
    
    learner = DTLearner(leaf_size=20, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(d.pred.tail())
    print(f"DT ACC:\t\t{acc:.2%}")
    
    learner = BagLearner(learner=DTLearner, bags=20, kwargs={'leaf_size': 20}, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(d.pred.tail())
    print(f"BAG DT ACC:\t\t{acc:.2%}")

    learner = BagLearner(learner=RTLearner, bags=20, kwargs={'leaf_size': 20}, verbose=False)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(d.pred.tail())
    print(f"BAG RT ACC:\t\t{acc:.2%}")
    
    learner = InsaneLearner(bags=20)
    learner.add_evidence(xtrain, ytrain)
    pred = learner.query(xtest)
    acc = d.accuracy(pred, ytest)
    print(d.pred.tail())
    print(f"INSANE ACC:\t\t{acc:.2%}")
    
    