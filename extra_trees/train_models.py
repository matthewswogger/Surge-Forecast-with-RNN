import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

class train_models:
    def __init__(self,data_dict):
        self.data_dict=data_dict


    def fit(self):
        '''
        Trains our entire model. Our model consits of 16 seperate
        ExtraTreesRegressors, one for each point, with 128 trees in them.
        Once a model is trained it is placed into a dictionary as the value
        where the keys to access it is the point name.
        '''
        def to_train(X, y):
            '''
            Returns one ExtraTreesRegressor model.
            '''
            return ExtraTreesRegressor(n_estimators=128,n_jobs=-1,max_depth=12).fit(X, y)

        self.model_dict = {}
        for key, value in self.data_dict.iteritems():
            self.model_dict[key] = to_train(value['X_train'], value['y_train'])


    def predict_on_test_sets(self):
        '''
        Does a prediction with all 16 models on the test set and stores them
        in a dictionary similar to the way the models were stored.
        '''
        self.predict_test_dict = {}
        for point, model in self.model_dict.iteritems():
            self.predict_test_dict[point] = model.predict(self.data_dict[point]['X_test'])
        return self.predict_test_dict


    def predict_on_hold_out_sets(self):
        '''
        Does a prediction with all 16 models on the hold out set and stores
        them in a dictionary similar to the way the models were stored. This
        is the more interesting prediction because it is still in timeseries
        format.
        '''
        self.predict_hold_out_dict = {}
        for point, model in self.model_dict.iteritems():
            self.predict_hold_out_dict[point] = model.predict(self.data_dict[point]['X_hold_out'])
        return self.predict_hold_out_dict


    def mse_on_test_sets(self):
        '''
        Computes the mean squared error for all 16 test sets and then
        averages them to give us the overall mean squared error of the
        entire model.
        '''
        return np.mean([np.mean((prediction - self.data_dict[point]['y_test'])**2) \
                                                        for point, prediction in self.predict_test_dict.iteritems()])


    def mse_on_hold_out_sets(self):
        '''
        Computes the mean squared error for all 16 hold out sets and then
        averages them to give us the overall mean squared error of the
        entire model.
        '''
        # np.append(np.delete(np.delete(self.data_dict[point]['y_hold_out'],0,0),0,0),[1,1])
        return np.mean([np.mean((prediction - self.data_dict[point]['y_hold_out'])**2) \
                                                    for point, prediction in self.predict_hold_out_dict.iteritems()])
