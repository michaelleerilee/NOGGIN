#!/usr/bin/env python
"""

Support functions for NOGGIN.

ML Rilee, RSTLLC, mike@rilee.net for NASA/ACCESS-15/NOGGIN.

"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC
from math import *

from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
import pykrige.core as core

from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

import unittest

rad_from_deg = pi/180.0

def random_index(x0,y0,x,y,params,distribution='normal',l=-1,w=-1):

    # print 'random_index: x0,y0: '+str(x0)+', '+str(y0)\
    #    +' l,w: '+str(l)+', '+str(w)
    
    xt = x-x0; yt = y-y0

    # print 'random_index: mnmx(x):  '+str(np.nanmin(x))+', '+str(np.nanmax(x))
    # print 'random_index: mnmx(y):  '+str(np.nanmin(y))+', '+str(np.nanmax(y))
    
    # print 'random_index: mnmx(xt): '+str(np.nanmin(xt))+', '+str(np.nanmax(xt))
    # print 'random_index: mnmx(yt): '+str(np.nanmin(yt))+', '+str(np.nanmax(yt))
    
    if distribution == 'normal':
        sigma2=params*params
        f = np.exp(-(np.power(xt,2)+np.power(yt,2))/(2.0*sigma2))/sqrt(2.0*pi*sigma2)
    elif distribution == 'exponential':
        f = np.exp(-np.sqrt(np.power(xt,2)+np.power(yt,2))/params)/params
    elif distribution == 'power-law':
        f = np.power(np.sqrt(1.0+np.power(xt,2)+np.power(yt,2)),-params)
    if l > 0:
        if w < 0:
            w = l
    else:
        if w > 0:
            l = w

    if l > 0:
        f[np.where(((-l > xt)|(xt > l))|((-w > yt)|(yt > w)))] = 0.0
        
        # f[np.where((-l < xt)&(xt < l)&(-w < yt)&(yt < w))] = 0.0
    
    return (f-np.random.uniform(0.0,1.0,f.shape)) > 0.0

def adaptive_index(x0,y0,x,y,npts=100,beta0=200,frac=0.1,distribution='normal',l=-1,w=-1):
    idx_size = 0; idx = False
    beta=beta0
    iter=0
    idx = np.full(x.shape,False)
    while(True):
        iter=iter+1
        idx = idx | random_index(x0,y0,x,y,beta,distribution,l=l,w=w)
        idx_size = idx[idx].size
        if idx_size >= npts:
            break
        else:
            beta=beta*(1.0-frac)
            if beta/beta0 < 0.00001 or iter > 1000:
                # fail not so silently
                print 'noggin.adaptive_index Too many iterations, beta too small.'\
                    +' size='+str(idx_size)\
                    +' iter='+str(iter)
                return idx                
                # raise ValueError('Too many iterations, beta too small.'\
                #                      +' size='+str(idx_size)\
                #                      +' iter='+str(iter)\
                #                      )
    return idx

def span_array(a):
    return (np.nanmax(a) - np.nanmin(a))

def center_array(a):
    return 0.5*(np.nanmax(a) + np.nanmin(a))

###########################################################################

class krigePlotConfiguration(object):
    def __init__(self\
                 ,kriged                  = True\
                 ,kriged_data             = True\
                 ,kriged_outline          = False\
                 ,source_data             = True\
                 ,source_data_last_sample = True\
                 ,variogram               = False\
                 ,debug                   = True\
                 ,verbose                 = True\
                 ,marker_size             = 0.5\
                 ,m_alpha                 = 1.0\
                 ,colormap                = plt.cm.rainbow\
                 ,vmin                    = -2.0\
                 ,vmax                    = 1.25\
                 ,title                   = 'title'\
                 ,zVariableName           = 'z'\
                 ):
        self.kriged                  = kriged
        self.kriged_data             = kriged_data
        self.kriged_outline          = kriged_outline
        self.source_data             = source_data
        self.source_data_last_sample = source_data_last_sample
        self.variogram               = variogram
        self.debug                   = debug
        self.verbose                 = verbose
        self.marker_size             = marker_size
        self.m_alpha                 = m_alpha
        self.colormap                = colormap
        self.vmin                    = vmin
        self.vmax                    = vmax
        self.title                   = title
        self.zVariableName           = zVariableName
        

class krigeResults(object):
    """Capture the result of the kriging calculation and provide a place
to store debugging information as well."""
    def __init__(self\
                 ,s=None\
                 ,z=None,x=None,y=None\
                 ,src_z=None,src_x=None,src_y=None\
                 ,dbg_z=None,dbg_x=None,dbg_y=None\
                 ,hull=None\
                 ,box=None\
                 ,zVariableName="z"\
                 ,title='KrigeResultTitle'\
                 ,vg_function=None\
                 ,vg_parameters=None\
                 ,note='Default note for krigeResults'):
        self.clear()
        if x is not None:
            self.x=x.copy()
        if y is not None:
            self.y=y.copy()
        if z is not None:
            self.z=z.copy()
        if s is not None:
            self.s=s.copy()
        if src_x is not None:
            self.src_x=src_x.copy()
        if src_y is not None:
            self.src_y=src_y.copy()
        if src_z is not None:
            self.src_z=src_z.copy()
        if dbg_x is not None:
            self.dbg_x=dbg_x.copy()
        if dbg_y is not None:
            self.dbg_y=dbg_y.copy()
        if dbg_z is not None:
            self.dbg_z=dbg_z.copy()
        if dbg_x is not None \
           and dbg_y is not None \
           and dbg_z is not None:
            self.dbg = True
        else:
            self.dbg = False
        self.title = title
        if hull is not None:
            self.hull = hull
        if box is not None:
            self.box = box.copy()
        else:
            self.box = None
        if vg_parameters is not None:
            self.vg_parameters = vg_parameters
        if vg_function is not None:
            self.vg_function = vg_function
        self.note = str(note)
        self.sort_on_longitude()
        self.construct_hull()
    def sort_on_longitude_xyz(self):
        if self.x is not None \
           and self.y is not None \
           and self.z is not None:
            idx = self.x.argsort()
            self.y = self.y[idx[::-1]]
            self.z = self.z[idx[::-1]]
            self.x = self.x[idx[::-1]]
    def sort_on_longitude_src_xyz(self):
        if self.src_x is not None \
           and self.src_y is not None \
           and self.src_z is not None:
            idx = self.src_x.argsort()
            self.src_y = self.src_y[idx[::-1]]
            self.src_z = self.src_z[idx[::-1]]
            self.src_x = self.src_x[idx[::-1]]
    def sort_on_longitude_dbg_xyz(self):
        if self.dbg_x is not None \
           and self.dbg_y is not None \
           and self.dbg_z is not None:
            idx = self.dbg_x.argsort()
            self.dbg_y = self.dbg_y[idx[::-1]]
            self.dbg_z = self.dbg_z[idx[::-1]]
            self.dbg_x = self.dbg_x[idx[::-1]]
    def sort_on_longitude(self):
        """Sort data to avoid bug in basemap. TODO: move into __init__."""
        self.sort_on_longitude_xyz()
        self.sort_on_longitude_src_xyz()
        self.sort_on_longitude_dbg_xyz()
        # else fail silently
    def clear(self):
        self.x     = None
        self.y     = None
        self.z     = None
        self.s     = None
        self.src_x = None
        self.src_y = None
        self.src_z = None
        self.dbg_x = None
        self.dbg_y = None
        self.dbg_z = None
        self.hull  = None
        self.box   = None
        self.note  = 'Default note for krigeResults'
    def construct_hull(self):
        """Construct the hull from z,x,y. z is used to get the shape of the data, 
        so it could be replaced using x and y alone."""
        if self.z is not None \
           and self.x is not None \
           and self.y is not None:
            xy1 = np.zeros((self.z.shape[0],2))
            xy1[:,0] = self.x
            xy1[:,1] = self.y
            self.hull = ConvexHull(xy1)

###########################################################################

# https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap

def draw_screen_poly( lons, lats, m, facecolor='black', edgecolor='black' ):
    """Draw a polygon based on sequences of lons & lats and a Basemap m. Note len(lons) = len(lats)."""
    x, y = m( lons, lats )
    plt.gca().add_patch( Polygon( zip(x,y)\
                                  ,facecolor=facecolor\
                                  ,edgecolor=edgecolor\
                                  ,alpha=0.8\
                                  ,fill=False\
                                  ))

###########################################################################

#### mlr #### # /Users/mrilee/src/python/PyKrige-1.3.2.tar.gz
#### mlr #### # core.variogram_function_error
#### mlr #### def variogram_function_error(params, x, y, variogram_function, weight):
#### mlr ####     """Function used to in fitting of variogram model.
#### mlr ####     Returns RMSE between calculated fit and actual data."""
#### mlr #### 
#### mlr ####     diff = variogram_function(params, x) - y
#### mlr #### 
#### mlr ####     if weight:
#### mlr ####         weights = np.arange(x.size, 0.0, -1.0)
#### mlr ####         weights /= np.sum(weights)
#### mlr ####         rmse = np.sqrt(np.average(diff**2, weights=weights))
#### mlr ####     else:
#### mlr ####         rmse = np.sqrt(np.mean(diff**2))
#### mlr #### 
#### mlr ####     return rmse
#### mlr #### 
#### mlr #### # /Users/mrilee/src/python/PyKrige-1.3.2.tar.gz
#### mlr #### # core.calculate_variogram_model
#### mlr #### 
#### mlr #### def calculate_variogram_model(lags, semivariance, variogram_model, variogram_function, weight):
#### mlr ####     """Function that fits a variogram model when parameters are not specified."""
#### mlr #### 
#### mlr ####     ## TODO Fix hardcoding
#### mlr ####     # x0 = [2.6,0.2,0.75]
#### mlr ####     x0 = [2.0,0.01,0.0001]
#### mlr ####     
#### mlr ####     ## PyKrige 1.3.2 core._variogram_function_error
#### mlr ####     if False:
#### mlr ####         bnds = ((0.0, 1000), (0.0, 10.0), (0.0, 10.0))
#### mlr ####         res = minimize(variogram_function_error\
#### mlr ####                        ,x0, args=(lags, semivariance, variogram_function, weight),\
#### mlr ####                        method='SLSQP', bounds=bnds)
#### mlr ####     
#### mlr ####     ## PyKrige 1.4 now uses least_squares with a soft L1-norm to minimize outliers...
#### mlr ####     if True:
#### mlr ####         bnds = ([0.0,0.0,0.0], [1000,10,10])
#### mlr ####         res = least_squares(core._variogram_residuals, x0, bounds=bnds, loss='soft_l1',\
#### mlr ####                             args=(lags, semivariance, variogram_function, weight))
#### mlr #### 
#### mlr ####     return res.x
#### mlr #### 
#### mlr #### 
#### mlr #### # PyKrige-1.4.0.tar.gz
#### mlr #### # core._initialize_variogram_model
#### mlr #### 
#### mlr #### def initialize_variogram_model140(X, y, variogram_model,
#### mlr ####                                   variogram_model_parameters, variogram_function,
#### mlr ####                                   nlags, weight, coordinates_type='geographic'):
#### mlr ####     """Initializes the variogram model for kriging. If user does not specify
#### mlr ####     parameters, calls automatic variogram estimation routine.
#### mlr ####     Returns lags, semivariance, and variogram model parameters.
#### mlr #### 
#### mlr ####     Parameters
#### mlr ####     ----------
#### mlr ####     X: ndarray
#### mlr ####         float array [n_samples, n_dim], the input array of coordinates
#### mlr ####     y: ndarray
#### mlr ####         float array [n_samples], the input array of values to be kriged
#### mlr ####     variogram_model: str
#### mlr ####         user-specified variogram model to use
#### mlr ####     variogram_model_parameters: list
#### mlr ####         user-specified parameters for variogram model
#### mlr ####     variogram_function: callable
#### mlr ####         function that will be called to evaluate variogram model
#### mlr ####         (only used if user does not specify variogram model parameters)
#### mlr ####     nlags: int
#### mlr ####         integer scalar, number of bins into which to group inter-point distances
#### mlr ####     weight: bool
#### mlr ####         boolean flag that indicates whether the semivariances at smaller lags
#### mlr ####         should be weighted more heavily in the automatic variogram estimation
#### mlr ####     coordinates_type: str
#### mlr ####         type of coordinates in X array, can be 'euclidean' for standard
#### mlr ####         rectangular coordinates or 'geographic' if the coordinates are lat/lon
#### mlr #### 
#### mlr ####     Returns
#### mlr ####     -------
#### mlr ####     lags: ndarray
#### mlr ####         float array [nlags], distance values for bins into which the
#### mlr ####         semivariances were grouped
#### mlr ####     semivariance: ndarray
#### mlr ####         float array [nlags], averaged semivariance for each bin
#### mlr ####     variogram_model_parameters: list
#### mlr ####         parameters for the variogram model, either returned unaffected if the
#### mlr ####         user specified them or returned from the automatic variogram
#### mlr ####         estimation routine
#### mlr ####     """
#### mlr #### 
#### mlr ####     # distance calculation for rectangular coords now leverages
#### mlr ####     # scipy.spatial.distance's pdist function, which gives pairwise distances
#### mlr ####     # in a condensed distance vector (distance matrix flattened to a vector)
#### mlr ####     # to calculate semivariances...
#### mlr ####     if coordinates_type == 'euclidean':
#### mlr ####         d = pdist(X, metric='euclidean')
#### mlr ####         g = 0.5 * pdist(y[:, None], metric='sqeuclidean')
#### mlr #### 
#### mlr ####     # geographic coordinates only accepted if the problem is 2D
#### mlr ####     # assume X[:, 0] ('x') => lon, X[:, 1] ('y') => lat
#### mlr ####     # old method of distance calculation is retained here...
#### mlr ####     # could be improved in the future
#### mlr ####     elif coordinates_type == 'geographic':
#### mlr ####         if X.shape[1] != 2:
#### mlr ####             raise ValueError('Geographic coordinate type only '
#### mlr ####                              'supported for 2D datasets.')
#### mlr ####         x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
#### mlr ####         y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
#### mlr ####         z1, z2 = np.meshgrid(y, y, sparse=True)
#### mlr ####         d = core.great_circle_distance(x1, y1, x2, y2)
#### mlr ####         g = 0.5 * (z1 - z2)**2.
#### mlr ####         indices = np.indices(d.shape)
#### mlr ####         d = d[(indices[0, :, :] > indices[1, :, :])]
#### mlr ####         g = g[(indices[0, :, :] > indices[1, :, :])]
#### mlr #### 
#### mlr ####     else:
#### mlr ####         raise ValueError("Specified coordinate type '%s' "
#### mlr ####                          "is not supported." % coordinates_type)
#### mlr #### 
#### mlr ####     # Equal-sized bins are now implemented. The upper limit on the bins
#### mlr ####     # is appended to the list (instead of calculated as part of the
#### mlr ####     # list comprehension) to avoid any numerical oddities
#### mlr ####     # (specifically, say, ending up as 0.99999999999999 instead of 1.0).
#### mlr ####     # Appending dmax + 0.001 ensures that the largest distance value
#### mlr ####     # is included in the semivariogram calculation.
#### mlr ####     dmax = np.amax(d)
#### mlr ####     dmin = np.amin(d)
#### mlr ####     dd = (dmax - dmin) / nlags
#### mlr ####     bins = [dmin + n * dd for n in range(nlags)]
#### mlr ####     dmax += 0.001
#### mlr ####     bins.append(dmax)
#### mlr #### 
#### mlr ####     # This old binning method was experimental and doesn't seem
#### mlr ####     # to work too well. Bins were computed such that there are more
#### mlr ####     # at shorter lags. This effectively weights smaller distances more
#### mlr ####     # highly in determining the variogram. As Kitanidis points out,
#### mlr ####     # the variogram fit to the data at smaller lag distances is more
#### mlr ####     # important. However, the value at the largest lag probably ends up
#### mlr ####     # being biased too high for the larger values and thereby throws off
#### mlr ####     # automatic variogram calculation and confuses comparison of the
#### mlr ####     # semivariogram with the variogram model.
#### mlr ####     #
#### mlr ####     # dmax = np.amax(d)
#### mlr ####     # dmin = np.amin(d)
#### mlr ####     # dd = dmax - dmin
#### mlr ####     # bins = [dd*(0.5**n) + dmin for n in range(nlags, 1, -1)]
#### mlr ####     # bins.insert(0, dmin)
#### mlr ####     # bins.append(dmax)
#### mlr #### 
#### mlr ####     lags = np.zeros(nlags)
#### mlr ####     semivariance = np.zeros(nlags)
#### mlr #### 
#### mlr ####     for n in range(nlags):
#### mlr ####         # This 'if... else...' statement ensures that there are data
#### mlr ####         # in the bin so that numpy can actually find the mean. If we
#### mlr ####         # don't test this first, then Python kicks out an annoying warning
#### mlr ####         # message when there is an empty bin and we try to calculate the mean.
#### mlr ####         if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
#### mlr ####             lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
#### mlr ####             semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
#### mlr ####         else:
#### mlr ####             lags[n] = np.nan
#### mlr ####             semivariance[n] = np.nan
#### mlr #### 
#### mlr ####     lags = lags[~np.isnan(semivariance)]
#### mlr ####     semivariance = semivariance[~np.isnan(semivariance)]
#### mlr #### 
#### mlr ####     # We only use our custom model, and then calculate the parameters.
#### mlr ####     
#### mlr ####     # a few tests the make sure that, if the variogram_model_parameters
#### mlr ####     # are supplied, they have been supplied as expected...
#### mlr ####     # if variogram_model_parameters was not defined, then estimate the variogram
#### mlr ####     if variogram_model_parameters is not None:
#### mlr ####         if variogram_model == 'linear' and len(variogram_model_parameters) != 2:
#### mlr ####             raise ValueError("Exactly two parameters required "
#### mlr ####                              "for linear variogram model.")
#### mlr ####         elif variogram_model in ['power', 'spherical', 'exponential',
#### mlr ####                                  'gaussian', 'hole-effect'] \
#### mlr ####                 and len(variogram_model_parameters) != 3:
#### mlr ####             raise ValueError("Exactly three parameters required for "
#### mlr ####                              "%s variogram model" % variogram_model)
#### mlr ####         # Nothing to do! elif variogram_model == 'custom':
#### mlr ####     else:
#### mlr ####         # if variogram_model == 'custom':
#### mlr ####         #     raise ValueError("Variogram parameters must be specified when "
#### mlr ####         #                      "implementing custom variogram model.")
#### mlr ####         # else:
#### mlr ####         variogram_model_parameters = \
#### mlr ####             calculate_variogram_model(lags, semivariance, variogram_model,
#### mlr ####                                        variogram_function, weight)
#### mlr #### 
#### mlr ####     # # TODO HACK -- Refactor and do better... 
#### mlr ####     # variogram_model_parameters = \
#### mlr ####     #                              calculate_variogram_model(lags, semivariance, variogram_model,
#### mlr ####     #                                                        variogram_function, weight)
#### mlr #### 
#### mlr ####     # 2018-0620-1252-45-EDT MLR The following does not work.
#### mlr ####     #                             core._calculate_variogram_model(lags, semivariance, variogram_model,
#### mlr #### 
#### mlr ####     return lags, semivariance, variogram_model_parameters

def fit_variogram(x,y,z
                  ,variogram_model
                  ,variogram_parameters,variogram_function
                  ,anisotropy_scaling=1.0
                  ,anisotropy_angle=0.0
                  ,nlags=12,weight=False
                  ,coordinates_type='geographic'
):
    """Fit the variogram, based on code from PyKrige 1.4. Try to elide asap.
"""
    print('fit_variogram')
    # Code assumes 1D input arrays. Ensures that any extraneous dimensions
    # don't get in the way. Copies are created to avoid any problems with
    # referencing the original passed arguments.
    self_X_ORIG = np.atleast_1d(np.squeeze(np.array(x, copy=True)))
    self_Y_ORIG = np.atleast_1d(np.squeeze(np.array(y, copy=True)))
    self_Z = np.atleast_1d(np.squeeze(np.array(z, copy=True)))
    self_XCENTER = (np.amax(self_X_ORIG) + np.amin(self_X_ORIG))/2.0
    self_YCENTER = (np.amax(self_Y_ORIG) + np.amin(self_Y_ORIG))/2.0
    
    if coordinates_type == 'euclidean':
        raise NotImplementedError('coordinates_type euclidean NOT IMPLEMENTED.')
        self_anisotropy_scaling = anisotropy_scaling
        self_anisotropy_angle = anisotropy_angle
        #self_X_ADJUSTED, self_Y_ADJUSTED = \
        #                                   core.adjust_for_anisotropy(np.copy(self_X_ORIG), np.copy(self_Y_ORIG),
        #                                                              self_XCENTER, self_YCENTER,
        #                                                              self_anisotropy_scaling, self_anisotropy_angle)
    elif coordinates_type == 'geographic':
        if anisotropy_scaling != 1.0:
            warnings.warn("Anisotropy is not compatible with geographic "
                          "coordinates. Ignoring user set anisotropy.",
                          UserWarning)
        # Anisotropy not applicable to geographic case
        # From PyKrige. Many of these parameters feed _adjust_for_anisotropy
        self_XCENTER= 0.0
        self_YCENTER= 0.0
        self_anisotropy_scaling = 1.0
        self_anisotropy_angle = 0.0
        self_X_ADJUSTED = self_X_ORIG
        self_Y_ADJUSTED = self_Y_ORIG
        
    self_variogram_model    = variogram_model
    self_variogram_function = variogram_function
    
    # core.initialize_variogram_model(self_X_ADJUSTED, self_Y_ADJUSTED, self_Z,
    # 132
    # self_lags, self_semivariance, self_variogram_model_parameters = \
    #     initialize_variogram_model132(self_X_ADJUSTED, self_Y_ADJUSTED, self_Z,
    #                                self_variogram_model, variogram_parameters,
    #                                self_variogram_function, nlags, weight
    #     )


    # # Note: also consider np.vstack((x,y)).T
    # self_lags, self_semivariance, self_variogram_model_parameters = \
    #     initialize_variogram_model140(\
    #                                   np.column_stack((self_X_ORIG,self_Y_ORIG))\
    #                                   ,self_Z\
    #                                   ,self_variogram_model, variogram_parameters\
    #                                   ,self_variogram_function, nlags, weight\
    #                                   ,coordinates_type=coordinates_type\
    #     )

    # core._
    
    self_lags, self_semivariance, self_variogram_model_parameters = \
        core._initialize_variogram_model(\
                                      np.column_stack((self_X_ORIG,self_Y_ORIG))\
                                      ,self_Z\
                                      ,self_variogram_model, variogram_parameters\
                                      ,self_variogram_function, nlags, weight\
                                      ,coordinates_type=coordinates_type\
        )
    
# This only used for euclidean...
#                                          np.column_stack((self_X_ADJUSTED, self_Y_ADJUSTED))\

    
    # self_display_variogram_model()

    return self_lags, self_semivariance, self_variogram_model_parameters


# lags,semivar,parms = initialize_variogram_model(data_x1,data_y,data_z1,'custom',custom_args,custom_vg,nlags=12,weight=False)

###########################################################################

def drive_OKrige(
        x,y
        ,src_x,src_y,src_z
        ,grid_stride=1
        ,random_permute=False
        ,variogram_model=None
        ,variogram_parameters=None
        ,variogram_function=None
        ,nlags=12
        ,weight=False
        ,enable_plotting=False
        ,enable_statistics=False
        ,npts=5000
        ,beta0=1.5
        ,frac=0.0
        ,l=-1,w=-1
        ,verbose=False
        ,eps=1.0e-10
        ,backend='vectorized'
        ):
    """Krige from src_ arguments to x,y returning the kriged result gridz, gridss
and the data_ and the variogram_parameters of the last sub-calculation.

'Geographic' (lon-lat) coordinate type is assumed.

"""
    if variogram_parameters is None:
        calculate_parms = True
    else:
        calculate_parms = False

    variogram_parameters_used = []
    
    gridz  = np.zeros(x.shape)
    gridss = np.zeros(x.shape)
    if random_permute:
        ipermute = np.random.permutation(x.size)
    for i in range(0,x.size,grid_stride):
        if random_permute:
            # Krig to a random permutation of the input positions
            isample = ipermute[i:i+grid_stride]
        else:
            # Krig to a array-index-contiguous segment of the input positions
            isample = range(i,min(i+grid_stride,x.size))
        g_lon = x[isample]
        g_lat = y[isample]
        # l,w form a window from which to sample.
        idx = adaptive_index(
            center_array(g_lon)
            ,center_array(g_lat)
            ,src_x,src_y
            ,npts=npts,beta0=beta0,frac=frac,l=l,w=w
            ,distribution='normal')
        print('processing '+str(i)\
                  +' of '+str(x.size)\
                  +',  '+str(int(100*float(i)/x.size))\
                  +'% done, '+str(idx[idx].size))
        data_x = src_x[idx];
        ## Adapt to geographic coordinates used by PyKrige
        data_x1        = np.copy(data_x)
        idltz          = np.where(data_x < 0)
        data_x1[idltz] = data_x1[idltz] + 360.0
        data_y = src_y[idx]
        data_z = src_z[idx]
        data_z1= np.log(data_z)

        if calculate_parms:
            # Find parms
            lags,semivar,parms = fit_variogram(
                data_x1,data_y,data_z1
                ,variogram_model=variogram_model
                ,variogram_parameters=variogram_parameters
                ,variogram_function=variogram_function
                ,nlags=nlags
                ,weight=weight
                ,coordinates_type='geographic'
                )
            print('parms: ',parms)
            variogram_parameters = list(parms)

	    # OK = OrdinaryKriging( data_x1, data_y, data_z, variogram_model='exponential',
	    # OK = OrdinaryKriging( data_x1, data_y, data_z, nlags=nlags\
	    #                           ,variogram_model='exponential'\
	    #                           ,enable_plotting=True\
	    #                           )
	    #                           ,enable_plotting=True
	    OK = OrdinaryKriging(     data_x1, data_y, data_z1\
                                      ,variogram_model=variogram_model
                                      ,variogram_parameters=variogram_parameters
                                      ,variogram_function=variogram_function
                                      ,nlags=nlags
                                      ,weight=weight
                                      ,enable_plotting=enable_plotting
                                      ,enable_statistics=enable_statistics
                                      ,coordinates_type='geographic'
                                      ,verbose=verbose
                                      )
            
            #                          ,eps=eps
            # 0..360,-90..90
            #,coordinates_type='geographic'
            #,coordinates_type='euclidean'
        
        #                            ,nlags=nlags\
        #                            ,variogram_model='custom'\
        #                            ,variogram_function=custom_vg\
        #                            ,variogram_parameters=custom_args\
        #                            ,weight=True\
        #                            )
        #                          ,enable_plotting=True\
        #                         ,variogram_model='exponential'\
        #                         ,variogram_model='gaussian'\
        # OK = OrdinaryKriging(     data_x1, data_y, data_z1\
        #                           ,variogram_model='spherical'\
        #                           ,nlags=nlags\
        #                           ,enable_plotting=True\
        #                           ,weight=True
        #                           )
	    z, ss = OK.execute('points',g_lon,g_lat\
                               ,backend=backend\
            )
            #                   ,backend='vectorized'\
            #                   ,backend='loop'\
            #                   ,backend='C'\
	    # gridz[i:i+grid_stride]  = np.exp(z[:])
	    # gridss[i:i+grid_stride] = ss[:]
            if verbose:
                print('driveOKrige isample,mnmx(ln(z)): ',isample,np.nanmin(z),np.nanmax(z))
	    gridz [isample] = np.exp(z[:])
	    gridss[isample] = ss[:]
    # TODO Need to return gridss
    return krigeResults(s              = gridss\
                        ,z             = gridz\
                        ,x             = x\
                        ,y             = y\
                        ,src_z         = data_z\
                        ,src_x         = data_x\
                        ,src_y         = data_y\
                        ,vg_function   = variogram_function\
                        ,vg_parameters = variogram_parameters)
    # return gridz, data_x, data_y, data_z, variogram_parameters
    
####

class rectangular_grid():
    x0 = -180.0
    x1 =  180.0
    dx =    1.0
    y0 =  -90.0
    y1 =   90.0
    dy =    1.0
    x  = None
    y  = None
    xy = None

    def __init__(self\
                     ,x0 = -180.0\
                     ,x1 =  180.0\
                     ,dx =    1.0\
                     ,y0 =  -90.0\
                     ,y1 =   90.0\
                     ,dy =    1.0\
                     ):
        self.x0 = x0
        self.x1 = x1
        self.dx = dx
        self.y0 = y0
        self.y1 = y1
        self.dy = dy

        self.xy = np.meshgrid(     np.arange(self.x0,self.x1,self.dx)\
                                  ,np.arange(self.y0,self.y1,self.dy))
        # xy = np.meshgrid(np.arange(-120.0,-105.0,0.25),np.arange(  20.0,  25.0,0.25))
        self.x = np.ravel(self.xy[0])
        self.y = np.ravel(self.xy[1])

    
    def in_grid(self,longitude,latitude):
        return \
          np.where(      (self.x0 < longitude) & (longitude < self.x1)\
                             & (self.y0 < latitude ) & (latitude  < self.y1)\
                            )
    def ex_grid(self,longitude,latitude):
        return \
          np.where(      (self.x0 > longitude) | (longitude > self.x1)\
                            | (self.y0 > latitude ) | (latitude  > self.y1)\
                            )

    def gridxy(self):
        return self.x,self.y
          
####

class fig_generator():
    iSubplot       = -1
    irowSubplots   = -1
    icolSubplots   = -1
    nrowSubplots   =  1
    ncolSubplots   =  1
    nTotalSubplots =  1
    fig  = None
    axes = None
    
    def __init__(self,nrow,ncol,figsize=(2560.0/110.0,1400.0/110.0),dpi=110.0):
        self.nrowSubplots   =  nrow
        self.ncolSubplots   =  ncol
        self.nTotalSubplots =  nrow*ncol
        self.figsize=figsize
        self.dpi=dpi
        # if figsize is None:
        #     plt.figure()
        # else:
        #     plt.figure(figsize=figsize,dpi=dpi)
        #     self.figsize=figsize
        #     self.dpi=dpi
        self.increment_figure()
        # fig, axes = plt.subplots(ncol,nrow)
        # if self.nTotalSubplots == 1:
        #     axes = [axes]
        # self.fig = fig; self.axes = axes

    def iSubIndex(self):
        if self.nrowSubplots == 1 and self.ncolSubplots == 1 :
            iSub = 0
        elif self.nrowSubplots == 1 :
            iSub = self.icolSubplots
        elif self.ncolSubplots == 1 :
            iSub = self.irowSubplots
        else:
            iSub = (self.icolSubplots,self.irowSubplots)
        return iSub

    def increment_figure(self):
        self.iSubplot=self.iSubplot+1
        self.icolSubplots = self.iSubplot % self.ncolSubplots
        self.irowSubplots = self.iSubplot/self.ncolSubplots
        iSub = self.iSubIndex()
        if self.iSubplot % self.nTotalSubplots == 0:            
            fig, axes = plt.subplots(self.ncolSubplots,self.nrowSubplots\
                                     ,figsize=self.figsize,dpi=self.dpi)
            if self.nTotalSubplots == 1:
                axes = [axes]
            self.fig = fig; self.axes = axes

    def get_fig_axes(self,):
        return self.fig, self.axes

###########################################################################

class bounding_box_latlon():
    """Provide a way to determine if latlon bounding boxes overlap."""
    def __init__(self,lat_min,lat_max,lon_min,lon_max,lat_center=None,lon_center=None\
                     ,label=None):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        if lat_center is None:
            self.lat_center = 0.5*(lat_min+lat_max)
        else:
            self.lat_center = lat_center
        if lon_center is None:
            self.lon_center = 0.5*(lon_min+lon_max)
        else:
            self.lon_center = lon_center

    def overlap(self,other):
        return \
          (((self.lat_min < other.lat_max) & (other.lat_max < self.lat_max))\
               |((self.lat_min < other.lat_min) & (other.lat_min < self.lat_max)))\
               &(((self.lon_min < other.lon_max) & (other.lon_max < self.lon_max))\
                |((self.lon_min < other.lon_min) & (other.lon_min < self.lon_max)))


class Test_bounding_box(unittest.TestCase):

    def setUp(self):
        self.bb0=bounding_box_latlon(10, 20, 15, 25)
        self.bb1=bounding_box_latlon(10, 20, 35, 45)
        self.bb2=bounding_box_latlon( 5, 15,  5, 20)
        self.bb3=bounding_box_latlon(10, 15, 35, 45)
        self.bb4=bounding_box_latlon
    # def test_center_calc_1(self):

    def test_overlap_empty_1(self):
        self.assertEqual(self.bb0.overlap(self.bb1), False)

    def test_overlap_empty_2(self):
        self.assertEqual(self.bb0.overlap(self.bb3), False)

    def test_ovelap_low_1(self):
        self.assertEqual(self.bb0.overlap(self.bb2), True)        
        
    def test_ovelap_high_1(self):
        self.assertEqual(self.bb2.overlap(self.bb0), True)        
    
###########################################################################

def data_src_directory():
    """Determine the directory containing the data to be processed."""
    if('NOGGIN_DATA_SRC_DIRECTORY' in os.environ):
        SRC_DIRECTORY_BASE=os.environ['NOGGIN_DATA_SRC_DIRECTORY']
    else:
        SRC_DIRECTORY_BASE='./'
    return SRC_DIRECTORY_BASE
        
###########################################################################
    

if __name__ == '__main__':
    if False:
        fig_gen = fig_generator(1,1)
        fig_gen.increment_figure()
        fig, ax = fig_gen.get_fig_axes()

    unittest.main()

    
