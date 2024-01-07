# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:36:24 2023

@author: 1055842
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 12:30:54 2021

@author: JCHAPIN
"""
########  Copy the code Below   ################################################

import csv
import numpy as np
from matplotlib import pyplot as plt


def createActuals():
    # function to create an array of x values
    # and y values with a given slope and y intercept
    #data = np.array((numPoints, 2)).astype(float)
    # no need to change
    data = np.zeros((numPoints, 3))

    for k in range(numPoints):
        data[k][0] = 1.0
        data[k][1] = float(k)
        data[k][2] = y_intercept + float(slope)*k

    x = data[:, 0:2]
    y = data[:, 2]
    print(x, y)
    return x, y


def createWeights():
    # this function creates a set of weights that you
    # will calculate the cost for
    # no need to change
    numWeights = 30
    biasWeights = np.linspace(0, y_intercept*2, numWeights)
    x1Weights = np.linspace(0, slope*2, numWeights)
    weights = np.transpose([biasWeights, x1Weights])
    return weights


def calcCost(x, weights, y):
    # =============================================================================
    # ############  YOUR CODE HERE
    # return the cost for the weights
    # =============================================================================
    pred = np.dot(x, weights)
    cost = np.mean((pred - y) ** 2)
    return cost


###########################  Main     ######################
numPoints = 4
slope = 8.0
y_intercept = 10.0
x, y = createActuals()
weight_arr = createWeights()
cost_arr = []

for w in weight_arr:
    cost_arr.append(calcCost(x, w, y))
# =============================================================================
# #####   ############  YOUR CODE HERE
#  write code to calculate the cost for each set of weights
# #####  and graph it
# ============================================================================= 

x_values = weight_arr[:, 1]
y_values = cost_arr[:]
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.2,0.8,0.7])
ax1.scatter(x_values, y_values,)
ax1.set_title('Cost Funciton')
ax1.set_xlabel('Weight')
ax1.set_ylabel('Cost')
