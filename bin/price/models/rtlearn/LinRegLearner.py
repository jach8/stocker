import numpy as np

class LinRegLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, verbose=False):
        """
        Constructor method
        """
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jachaibar3"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # slap on 1s column so linear regression finds a constant term
        new_data_x = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data_x[:, 0 : data_x.shape[1]] = data_x

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(
            new_data_x, data_y, rcond=None
        )

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[
            -1
        ]

if __name__ == '__main__':
    import pandas as pd 
    from models.rtlearn.RTLearner import RTLearner
    from models.rtlearn.DTLearner import DTLearner
    from models.rtlearn.BagLearner import BagLearner
    from models.rtlearn.InsaneLearner import InsaneLearner
    
    # d = data()
    # xtrain, ytrain, xtest, ytest = d.split('spy', t = 200, start_date='2020-01-01', discretize_features=True, returns = 5)
    # print(f""" xtrain: {xtrain.shape}, ytrain: {ytrain.shape} xtest: {xtest.shape}, ytest: {ytest.shape}""")
    
    # learner = DTLearner(leaf_size=20, verbose=False)
    # learner.add_evidence(xtrain, ytrain)
    # pred = learner.query(xtest)
    # acc = d.accuracy(pred, ytest)
    # print(f"ORT ACC:\t\t{acc:.2%}")
    # del learner, pred, acc
    
    # learner = DTLearner(leaf_size=20, verbose=False)
    # learner.add_evidence(xtrain, ytrain)
    # pred = learner.query(xtest)
    # acc = d.accuracy(pred, ytest)
    # print(f"ODT ACC:\t\t{acc:.2%}")
    # del learner, pred, acc
    
    # learner = BagLearner(learner=DTLearner, bags=20, kwargs={'leaf_size': 20}, verbose=False)
    # learner.add_evidence(xtrain, ytrain)
    # pred = learner.query(xtest)
    # acc = d.accuracy(pred, ytest)
    # print(f"BDT ACC:\t\t{acc:.2%}")
    # del learner, pred, acc

    # learner = BagLearner(learner=RTLearner, bags=20, kwargs={'leaf_size': 20}, verbose=False)
    # learner.add_evidence(xtrain, ytrain)
    # pred = learner.query(xtest)
    # acc = d.accuracy(pred, ytest)
    # print(f"BRT ACC:\t\t{acc:.2%}")
    # del learner, pred, acc
    
    # learner = InsaneLearner(bags=20)
    # learner.add_evidence(xtrain, ytrain)
    # pred = learner.query(xtest)
    # acc = d.accuracy(pred, ytest)
    # print(f"INS ACC:\t\t{acc:.2%}")
    # del learner, pred, acc
    
    # learner = LinRegLearner(verbose=False)
    # learner.add_evidence(xtrain, ytrain)
    # pred = learner.query(xtest)
    # acc = d.accuracy(pred, ytest)
    # print(f"LiR ACC:\t\t{acc:.2%}")
    # del learner, pred, acc
    