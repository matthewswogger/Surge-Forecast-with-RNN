import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

class prepare_data:
    def __init__(self, df):
        self.df = df
        self.dataframe_dict = {}

    def build_dataframes_for_points(self):
        '''
        makes different dataframes that only have data from one surge point.
        Also makes sure they all have the same amount of rows, that's what
        the number 31184 is doing, not the most generalized function but it
        doesn't need to be.
        '''
        point_list = [0, 1, 2, 3, 14, 15, 16, 12, 13, 24, 25, 26, 27, 28, 17, 29]
        self.all_frames = [('point_{}'.format(point),self.df[self.df.point == point].reset_index(drop=True)[:31184])\
                                                                                            for point in point_list]

    def time_series_to_regression(self, forecast=5, past=60):
        '''
        Takes data for all points that currently is in time series format and converts
        it to a form that can be be used in regressions. It also saves a little bit
        of the data ,referred to as hold_out, from the very end of the time series to be
        able to graph and analyze as actual time series. This is done because the process
        of turning this into a regression and the randomness that is created when doing
        a proper train_test_split destroys the timeseries nature of the data.

        How far into the future we choose to forecast is set with the forecast parameter
        where 5 is 5 minutes into the future. How far we look into the past is controlled
        by past, where 60 is 60 minutes into the past.

        This is then put into a dictionary for ease of access with the key the name of
        the individual points and the value is another dict like: {'X': X,
                                                                   'y': y,
                                                                   'X_hold_out': X_hold_out,
                                                                   'y_hold_out': y_hold_out'
                                                                   'X_train':X_train,
                                                                   'X_test':X_test,
                                                                   'y_train':y_train,
                                                                   'y_test':y_test}.
        '''
        for point, d_f in self.all_frames:
            d_f['difference'] = d_f.surge.diff()
            array = np.array([d_f.surge[i:i+past] for i in xrange(len(d_f.surge)-past)])
            array = array[:,::-1]

            array_dif = np.array([d_f.difference[i:i+past] for i in xrange(len(d_f.difference)-past)])
            array_dif = array_dif[:,::-1]
            array_dif = np.nan_to_num(array_dif)
            array_dif = array_dif[:,[1,3,5,7,9,11,13,15]]
            array = np.concatenate((array,array_dif), axis=1)
            # make a small hold out sample to conserve the timeseries nature of this, I'll graph it later
            hold_out_array = array[-500:,:]
            y_hold_out = hold_out_array[:,0]
            X_hold_out = hold_out_array[:,forecast:]

            # this is for the classical model training and testing
            training_testing_array = array[:-500,:]
            y = training_testing_array[:,0]
            X = training_testing_array[:,forecast:]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            self.dataframe_dict[point]={'X':X, 'y':y, 'X_hold_out':X_hold_out, 'y_hold_out':y_hold_out,
                                'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
        # for point, d_f in self.all_frames:
        #     array = np.array([d_f.surge[i:i+past] for i in xrange(len(d_f.surge)-past)])
        #     array = array[:,::-1]
        #     # make a small hold out sample to conserve the timeseries nature of this, I'll graph it later
        #     hold_out_array = array[-500:,:]
        #     y_hold_out = hold_out_array[:,0]
        #     X_hold_out = hold_out_array[:,forecast:]
        #
        #     # this is for the classical model training and testing
        #     training_testing_array = array[:-500,:]
        #     y = training_testing_array[:,0]
        #     X = training_testing_array[:,forecast:]
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #     self.dataframe_dict[point]={'X':X, 'y':y, 'X_hold_out':X_hold_out, 'y_hold_out':y_hold_out,
        #                         'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}

    def get_point_data(self):
        '''
        This returns the dictionary so that you can use it.
        '''
        return self.dataframe_dict
