import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn

def prepare_and_split_data(df, TIMESTEPS):
    """
    Inputs: dataframe: from a .csv
            integer: that represents the timesteps that will be used to forecast on

    Output: the X and y for the model split into: train: first .8 of data
                                                    val: next .1 of data
                                                   test: last .1 of data
            and stuck into a dictionary like so: y['train'] = training target
                                                 X['train'] = training data

    y is in the form:          p1,  p2,  p3, ... p14, p15, p16

                      array([[1.2, 1.0, 2.1, ... 1.8, 1.4, 1.6]   t
                             [1.0, 1.0, 1.2, ... 1.4, 1.4, 1.7]   t+1
                             ...                            ...
                             [1.0, 1.2, 2.4, ... 1.8, 1.3, 2.6]   t+n-1
                             [1.2, 1.0, 2.1, ... 1.8, 1.4, 1.6]]) t+n

    X is in the form: array([[[1.2, 1.0, 2.1, ... 1.8, 1.4, 1.6]
                              [1.0, 1.2, 1.2, ... 1.4, 1.0, 1.0]
                              [1.0, 1.0, 1.4, ... 1.9, 1.0, 1.9]
                              [1.2, 1.0, 1.4, ... 1.0, 1.0, 1.3]
                              [1.4, 1.2, 1.2, ... 1.4, 1.3, 1.7]]
                              ...                            ...
                             [[1.0, 1.0, 1.2, ... 1.4, 1.4, 1.7]
                              [1.0, 1.0, 1.2, ... 1.4, 1.4, 1.7]
                              [1.0, 1.0, 1.2, ... 1.4, 1.4, 1.7]
                              [1.0, 1.2, 2.4, ... 1.8, 1.3, 2.6]
                              [1.2, 1.0, 2.1, ... 1.8, 1.4, 1.6]]])
    """
    def build_dataframes_for_points(df):
        """
        makes different dataframes that only have data from one surge point
        """
        point_list = [0, 1, 2, 3, 14, 15, 16, 12, 13, 24, 25, 26, 27, 28, 17, 29]
        return [df[df.point == point].reset_index(drop=True)[:31184] for point in point_list]

    df_0,df_1,df_2,df_3,df_14,df_15,df_16,df_12,df_13,df_24,df_25,df_26,\
                                                            df_27,df_28,df_17,df_29 = build_dataframes_for_points(df)
    """
    this gives us a dataframe where the rows are one minute time intervals and
    each column is the surge value for a point"""
    df_surge = pd.concat([df_0.surge, df_1.surge, df_2.surge, df_3.surge, df_14.surge, df_15.surge, df_16.surge,\
                      df_12.surge, df_13.surge, df_24.surge, df_25.surge, df_26.surge, df_27.surge, df_28.surge,\
                      df_17.surge, df_29.surge], axis=1, keys=['point_0', 'point_1', 'point_2', 'point_3', 'point_14',\
                        'point_15', 'point_16', 'point_12', 'point_13', 'point_24', 'point_25','point_26', 'point_27',\
                        'point_28', 'point_17', 'point_29'])

    df_forecast = df_surge[5:].reset_index(drop=True)# the number 5 is how far ahead we will predict
    df_surge = df_surge[:len(df_forecast)].reset_index(drop=True)
    my_y = np.array(df_forecast) # this is the target forecast of 5min ahead
    surge_np = np.array(df_surge)
    my_x = np.empty([len(my_y),TIMESTEPS,16]) # filled with zeros that will be replaced with surge data
    for i, points_row in enumerate(surge_np):
        time_chunk = surge_np[i:i+TIMESTEPS,:]
        if time_chunk.shape == (TIMESTEPS, 16):
            my_x[i,:,:] = time_chunk
        else:
            my_x[i,:,:] = np.ones((TIMESTEPS,16))

    # get values to split data into train,val,test split
    train_data_len_end = round(len(my_y)*.8)
    val_data_len_end = round(len(my_y)*.9)

    return dict(train=my_x[:train_data_len_end], val=my_x[train_data_len_end:val_data_len_end],\
                test=my_x[val_data_len_end:]), dict(train=my_y[:train_data_len_end],\
                val=my_y[train_data_len_end:val_data_len_end], test=my_y[val_data_len_end:])


def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """
    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return learn.ops.dnn(input_layers,
                                 layers['layers'],
                                 activation=layers.get('activation'),
                                 dropout=layers.get('dropout'))
        elif layers:
            return learn.ops.dnn(input_layers, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unpack(X, axis=1, num=num_units)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = learn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model
