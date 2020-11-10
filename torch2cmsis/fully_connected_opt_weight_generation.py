#!/usr/bin/env python

import numpy as np

def convert_to_x4_q7_weights(weights):
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    num_of_rows = r
    num_of_cols = h*w*c
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    counter = 0
    for i in range(int(num_of_rows/4)):
      # we only need to do the re-ordering for every 4 rows
      row_base = 4*i
      for j in range (int(num_of_cols/4)):
        # for each 4 entries
        column_base = 4*j
        new_weights[counter]   =  weights[row_base  ][column_base  ]
        new_weights[counter+1] =  weights[row_base+1][column_base  ]
        new_weights[counter+2] =  weights[row_base  ][column_base+2]
        new_weights[counter+3] =  weights[row_base+1][column_base+2]
        new_weights[counter+4] =  weights[row_base+2][column_base  ]
        new_weights[counter+5] =  weights[row_base+3][column_base  ]
        new_weights[counter+6] =  weights[row_base+2][column_base+2]
        new_weights[counter+7] =  weights[row_base+3][column_base+2]

        new_weights[counter+8] =  weights[row_base  ][column_base+1]
        new_weights[counter+9] =  weights[row_base+1][column_base+1]
        new_weights[counter+10] = weights[row_base  ][column_base+3]
        new_weights[counter+11] = weights[row_base+1][column_base+3]
        new_weights[counter+12] = weights[row_base+2][column_base+1]
        new_weights[counter+13] = weights[row_base+3][column_base+1]
        new_weights[counter+14] = weights[row_base+2][column_base+3]
        new_weights[counter+15] = weights[row_base+3][column_base+3]
        counter = counter + 16
      # the remaining ones are in order
      for j in range((int)(num_of_cols-num_of_cols%4), int(num_of_cols)):
        new_weights[counter] = weights[row_base][j]
        new_weights[counter+1] = weights[row_base+1][j]
        new_weights[counter+2] = weights[row_base+2][j]
        new_weights[counter+3] = weights[row_base+3][j]
        counter = counter + 4
    return new_weights

def convert_to_x4_q15_weights(weights):
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    num_of_rows = r
    num_of_cols = h*w*c
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    counter = 0
    for i in range(int(num_of_rows/4)):
      # we only need to do the re-ordering for every 4 rows
      row_base = 4*i
      for j in range (int(num_of_cols/2)):
        # for each 2 entries
        column_base = 2*j
        new_weights[counter]   =  weights[row_base  ][column_base  ]
        new_weights[counter+1] =  weights[row_base  ][column_base+1]
        new_weights[counter+2] =  weights[row_base+1][column_base  ]
        new_weights[counter+3] =  weights[row_base+1][column_base+1]
        new_weights[counter+4] =  weights[row_base+2][column_base  ]
        new_weights[counter+5] =  weights[row_base+2][column_base+1]
        new_weights[counter+6] =  weights[row_base+3][column_base  ]
        new_weights[counter+7] =  weights[row_base+3][column_base+1]

        counter = counter + 8
      # the remaining ones are in order
      for j in range((int)(num_of_cols-num_of_cols%2), int(num_of_cols)):
        new_weights[counter] = weights[row_base][j]
        new_weights[counter+1] = weights[row_base+1][j]
        new_weights[counter+2] = weights[row_base+2][j]
        new_weights[counter+3] = weights[row_base+3][j]
        counter = counter + 4
    return new_weights

def convert_q7_q15_weights(weights):
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    num_of_rows = r
    num_of_cols = h*w*c
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    counter = 0
    for i in range(int(num_of_rows/4)):
      # we only need to do the re-ordering for every 4 rows
      row_base = 4*i
      for j in range (int(num_of_cols/2)):
        # for each 2 entries
        column_base = 2*j
        new_weights[counter]   =  weights[row_base  ][column_base  ]
        new_weights[counter+1] =  weights[row_base+1][column_base  ]
        new_weights[counter+2] =  weights[row_base  ][column_base+1]
        new_weights[counter+3] =  weights[row_base+1][column_base+1]
        new_weights[counter+4] =  weights[row_base+2][column_base  ]
        new_weights[counter+5] =  weights[row_base+3][column_base  ]
        new_weights[counter+6] =  weights[row_base+2][column_base+1]
        new_weights[counter+7] =  weights[row_base+3][column_base+1]

        counter = counter + 8
      # the remaining ones are in order
      for j in range((int)(num_of_cols-num_of_cols%2), int(num_of_cols)):
        new_weights[counter] = weights[row_base][j]
        new_weights[counter+1] = weights[row_base+1][j]
        new_weights[counter+2] = weights[row_base+2][j]
        new_weights[counter+3] = weights[row_base+3][j]
        counter = counter + 4
    return new_weights


