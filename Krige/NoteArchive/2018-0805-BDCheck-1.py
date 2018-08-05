#!/usr/bin/env python

import Krige
import pykrige
import pykrige.core as core

from scipy.spatial.distance import cdist

import numpy as np

print('\nstarting')

# Test geographic coords
# Source data locations
n_data    = 5000
x_data    = 180*np.arange(n_data)/n_data
y_data    = 90*np.arange(n_data)/n_data
xy_data_c = x_data + 1j * y_data

# Target data locations
n_points    = 1000
x_points    = 180*np.arange(n_points)/n_points
y_points    = 90*np.arange(n_points)/n_points
xy_points_c = x_points + 1j * y_points

# 3-D versions
lon_d = x_data[:, np.newaxis] * np.pi / 180.0
lat_d = y_data[:, np.newaxis] * np.pi / 180.0
xy_data = np.concatenate((np.cos(lon_d) * np.cos(lat_d),
                          np.sin(lon_d) * np.cos(lat_d),
                          np.sin(lat_d)), axis=1)

lon_p = x_points[:, np.newaxis] * np.pi / 180.0
lat_p = y_points[:, np.newaxis] * np.pi / 180.0
xy_points = np.concatenate((np.cos(lon_p) * np.cos(lat_p),
                            np.sin(lon_p) * np.cos(lat_p),
                            np.sin(lat_p)), axis=1)

bd0 = cdist(xy_points, xy_data, 'euclidean')
bd1 = core.euclid3_to_great_circle(bd0)

bd2 = core.great_circle_distance_c(xy_points_c[:,np.newaxis],xy_data_c)

delta_bd12 = np.zeros(bd2.shape)
delta_bd12[:]=bd2-bd1
print('max bd12: ',np.nanmax(np.abs(delta_bd12)))

print('done')
