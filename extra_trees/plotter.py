import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class split_for_plot:
    def __init__(self,predicted_values, true_values):
        '''
        Went ahead and used this to split the predicted_values into two
        different dictionaries so that they can be graphed in sets of eight
        instead of 16 within a jupyter notebook, this works best visually.
        '''
        self.predicted_values=predicted_values
        self.true_values=true_values
        self.first_8 = {}
        self.second_8 = {}
        count = 1
        for point, value in self.predicted_values.iteritems():
            if count <= 8:
                self.first_8[point]=value
            else:
                self.second_8[point]=value
            count+=1

    def plot_1(self):
        '''
        This is for the first 8 points to be plotted, they are not in any
        order due to using dictionaries.
        '''
        plt.xkcd() # makes it pretty
        plt.figure(figsize=(20,25))
        count=1
        subs = {1:811, 2:812, 3:813, 4:814, 5:815, 6:816, 7:817, 8:818}
        for point, values in self.first_8.iteritems():
            if count is 9:
                break
            plt.subplot(subs[count])
            plt.plot(values,label='3 min forecast',color='k')
            plt.plot(np.insert(self.true_values[point]['y_hold_out'],0,[1.0,1.0]),label='true {}'.format(point),color='c')
            plt.legend()
            count += 1
        # plt.savefig('forecast_1.png', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_2(self):
        '''
        This is for the second 8 points to be plotted, they are not in any
        order due to using dictionaries.
        '''
        plt.xkcd() # makes it pretty
        plt.figure(figsize=(20,25))
        count=1
        subs = {1:811, 2:812, 3:813, 4:814, 5:815, 6:816, 7:817, 8:818}
        for point, values in self.second_8.iteritems():
            if count is 9:
                break
            plt.subplot(subs[count])
            plt.plot(values,label='3 min forecast',color='k')
            plt.plot(np.insert(self.true_values[point]['y_hold_out'],0,[1.0,1.0]),label='true {}'.format(point),color='c')
            plt.legend()
            count += 1
        # plt.savefig('forecast_2.png', bbox_inches='tight', dpi=300)
        plt.show()
