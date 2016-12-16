from __future__ import division, print_function, absolute_import

import sqlite3
import tflearn as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def graph_em(test, forecast, graph_name):
    subs = {0:421, 1:422, 2:423, 3:424, 4:425, 5:426, 6:427, 7:428}
    plt.figure(figsize=(16,8))
    for i, point in enumerate(test.T):
        plt.subplot(subs[i])
        plt.title('History='+str(STEPS_OF_HISTORY)+', Future='+str(STEPS_IN_FUTURE))
        plt.plot(point, 'k-', label='Actual')
        plt.plot(forecast[:,i], 'c-', label='Forecast')
        plt.legend()

#     plt.savefig(graph_name, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


conn = sqlite3.connect('surge_data.db')
c = conn.cursor()

point_list = [0, 1, 2, 3, 14, 15, 16, 12, 13, 24, 25, 26, 27, 28, 17, 29]

sql_output = np.array([np.array(pd.read_sql('''WITH a AS (SELECT surge
                                               FROM surge
                                               WHERE point = {}
                                               ORDER BY date
                                               LIMIT 31184)
                                               SELECT *
                                               FROM a'''.format(point), conn)).T for point in point_list])[:,0,:].T

# this gets run when I'm done working for the session
conn.close()

STEPS_OF_HISTORY = 30 # how far back in time I look
STEPS_IN_FUTURE = 5 # how far in future I forecast
FEATURES = 16 # number of features in model
SPLIT = -1000 # where I make my train test split, this gives about 16.5 hours of test data

# prepare data for model
y = sql_output[STEPS_OF_HISTORY+STEPS_IN_FUTURE-1:,:]
X = sql_output[:len(y),:]
my_x = np.empty([len(y),STEPS_OF_HISTORY,FEATURES])

for i, _ in enumerate(X):
    time_chunk = X[i:i+STEPS_OF_HISTORY,:]
    if time_chunk.shape == (STEPS_OF_HISTORY, FEATURES):
        my_x[i,:,:] = time_chunk
    else:
        my_x[i,:,:] = np.ones((STEPS_OF_HISTORY,FEATURES))

trainX, testX = my_x[:SPLIT,:], my_x[SPLIT:,:]
trainY, testY = y[:SPLIT,:], y[SPLIT:,:]

# Build my neural net
net = tf.input_data(shape=[None, STEPS_OF_HISTORY, FEATURES])
net = tf.lstm(net, n_units=128, activation='softsign', return_seq=False)
net = tf.fully_connected(net, FEATURES, activation='linear')
net = tf.regression(net, optimizer='sgd', loss='mean_square', learning_rate=0.3)

# Define model
model = tf.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)

# Training
# EPOCHS = 10
# epochs_performed = 10
# for _ in xrange(20):
#     # Fit model
#     model.fit(trainX, trainY, n_epoch=EPOCHS, validation_set=0.1, batch_size=128)
#     # Save model
#     epochs_performed += 10
#     model.save("saved_model/{}_epoch_act_softsign_nunits_128.tfl".format(epochs_performed))

# Load a model
model.load("saved_model/150_epoch_act_softsign_nunits_128.tfl")

# predict
predictY = np.array(model.predict(testX))

print ('')
print ('Mean Squared Error: {}%'.format(round(np.mean((predictY - testY)**2)*100,2)))
print ('')

# print graphs of surge forecast
graph_em(testY[:,:8], predictY[:,:8], 'forecast_1.png')

graph_em(testY[:,8:], predictY[:,8:], 'forecast_2.png')
