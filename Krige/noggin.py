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
from pykrige import core
from scipy.optimize import minimize

import unittest

rad_from_deg = pi/180.0

def random_index(x0,y0,x,y,params,distribution='normal',l=-1,w=-1):
    xt = x-x0; yt = y-y0
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
                # fail silently
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

# https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap

def draw_screen_poly( lons, lats, m, facecolor='black', edgecolor='black' ):
    x, y = m( lons, lats )
    plt.gca().add_patch( Polygon( zip(x,y)\
                                  ,facecolor=facecolor\
                                  ,edgecolor=edgecolor\
                                  ,alpha=0.8\
                                  ,fill=False\
                                  ))

###########################################################################

# /Users/mrilee/src/python/PyKrige-1.3.2.tar.gz
# core.calculate_variogram_model

def calculate_variogram_model(lags, semivariance, variogram_model, variogram_function, weight):
    """Function that fits a variogram model when parameters are not specified."""

    ## TODO Fix hardcoding
    # x0 = [2.6,0.2,0.75]
    x0 = [2.0,0.01,0.0001]
    bnds = ((0.0, 100), (0.0, 10.0), (0.0, 10.0))
    res = minimize(core.variogram_function_error, x0, args=(lags, semivariance, variogram_function, weight),
                   method='SLSQP', bounds=bnds)
    return res.x


# /Users/mrilee/src/python/PyKrige-1.3.2.tar.gz
# core.initialize_variogram_model

def initialize_variogram_model(x, y, z, variogram_model, variogram_model_parameters,
                               variogram_function, nlags, weight):
    """Initializes the variogram model for kriging according
    to user specifications or to defaults"""

    print('initialize_variogram_model')

    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    z1, z2 = np.meshgrid(z, z)

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    d = np.sqrt(dx**2 + dy**2)
    g = 0.5 * dz**2

    indices = np.indices(d.shape)
    d = d[(indices[0, :, :] > indices[1, :, :])]
    g = g[(indices[0, :, :] > indices[1, :, :])]

    # Equal-sized bins are now implemented. The upper limit on the bins
    # is appended to the list (instead of calculated as part of the
    # list comprehension) to avoid any numerical oddities
    # (specifically, say, ending up as 0.99999999999999 instead of 1.0).
    # Appending dmax + 0.001 ensures that the largest distance value
    # is included in the semivariogram calculation.
    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin)/nlags
    bins = [dmin + n*dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    # This old binning method was experimental and doesn't seem
    # to work too well. Bins were computed such that there are more
    # at shorter lags. This effectively weights smaller distances more
    # highly in determining the variogram. As Kitanidis points out,
    # the variogram fit to the data at smaller lag distances is more
    # important. However, the value at the largest lag probably ends up
    # being biased too high for the larger values and thereby throws off
    # automatic variogram calculation and confuses comparison of the
    # semivariogram with the variogram model.
    #
    # dmax = np.amax(d)
    # dmin = np.amin(d)
    # dd = dmax - dmin
    # bins = [dd*(0.5**n) + dmin for n in range(nlags, 1, -1)]
    # bins.insert(0, dmin)
    # bins.append(dmax)

    lags = np.zeros(nlags)
    semivariance = np.zeros(nlags)

    for n in range(nlags):
        # This 'if... else...' statement ensures that there are data
        # in the bin so that numpy can actually find the mean. If we
        # don't test this first, then Python kicks out an annoying warning
        # message when there is an empty bin and we try to calculate the mean.
        if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
            lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
            semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
        else:
            lags[n] = np.nan
            semivariance[n] = np.nan

    lags = lags[~np.isnan(semivariance)]
    semivariance = semivariance[~np.isnan(semivariance)]

    #if variogram_model_parameters is not None:
    #    if variogram_model == 'linear' and len(variogram_model_parameters) != 2:
    #        raise ValueError("Exactly two parameters required "
    #                         "for linear variogram model")
    #    elif (variogram_model == 'power' or variogram_model == 'spherical' or variogram_model == 'exponential'
    #          or variogram_model == 'gaussian') and len(variogram_model_parameters) != 3:
    #        raise ValueError("Exactly three parameters required "
    #                         "for %s variogram model" % variogram_model)
    #else:
    #    if variogram_model == 'custom':
    #        raise ValueError("Variogram parameters must be specified when implementing custom variogram model.")
    #    else:
    #        variogram_model_parameters = calculate_variogram_model(lags, semivariance, variogram_model,
    #                                                               variogram_function, weight)

    variogram_model_parameters = calculate_variogram_model(lags, semivariance, variogram_model,
                                                                    variogram_function, weight)

    return lags, semivariance, variogram_model_parameters


def fit_variogram(x,y,z
                      ,variogram_model
                      ,variogram_parameters,variogram_function
                      ,anisotropy_scaling=1.0
                      ,anisotropy_angle=0.0
                      ,nlags=12,weight=False):
    """ """
    print('fit_variogram')
    # Code assumes 1D input arrays. Ensures that any extraneous dimensions
    # don't get in the way. Copies are created to avoid any problems with
    # referencing the original passed arguments.
    self_X_ORIG = np.atleast_1d(np.squeeze(np.array(x, copy=True)))
    self_Y_ORIG = np.atleast_1d(np.squeeze(np.array(y, copy=True)))
    self_Z = np.atleast_1d(np.squeeze(np.array(z, copy=True)))

    self_XCENTER = (np.amax(self_X_ORIG) + np.amin(self_X_ORIG))/2.0
    self_YCENTER = (np.amax(self_Y_ORIG) + np.amin(self_Y_ORIG))/2.0
    self_anisotropy_scaling = anisotropy_scaling
    self_anisotropy_angle = anisotropy_angle
    self_X_ADJUSTED, self_Y_ADJUSTED = \
        core.adjust_for_anisotropy(np.copy(self_X_ORIG), np.copy(self_Y_ORIG),
                                   self_XCENTER, self_YCENTER,
                                   self_anisotropy_scaling, self_anisotropy_angle)

    self_variogram_model = variogram_model
    self_variogram_function = variogram_function
    
    # core.initialize_variogram_model(self_X_ADJUSTED, self_Y_ADJUSTED, self_Z,
    self_lags, self_semivariance, self_variogram_model_parameters = \
        initialize_variogram_model(self_X_ADJUSTED, self_Y_ADJUSTED, self_Z,
                                        self_variogram_model, variogram_parameters,
                                        self_variogram_function, nlags, weight)
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
        ,weight=True
        ,enable_plotting=False
        ,npts=5000
        ,beta0=1.5
        ,frac=0.0
        ,l=-1,w=-1
        ):
    """ """
    if variogram_parameters is None:
        calculate_parms = True
    else:
        calculate_parms = False
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
        data_x = src_x[idx]; data_x1 = data_x
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
                )
            print('parms: ',parms)
            variogram_parameters = parms
        
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
                                      )
        
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
	    z, ss = OK.execute('points',g_lon,g_lat)
	    # gridz[i:i+grid_stride]  = np.exp(z[:])
	    # gridss[i:i+grid_stride] = ss[:]
	    gridz [isample] = np.exp(z[:])
	    gridss[isample] = ss[:]
    
    return gridz, data_x, data_y, data_z
    
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
    
    def __init__(self,nrow,ncol):
        self.nrowSubplots   =  nrow
        self.ncolSubplots   =  ncol
        self.nTotalSubplots =  nrow*ncol
        plt.figure()
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
            fig, axes = plt.subplots(self.ncolSubplots,self.nrowSubplots)
            if self.nTotalSubplots == 1:
                axes = [axes]
            self.fig = fig; self.axes = axes

    def get_fig_axes(self,):
        return self.fig, self.axes

###########################################################################

class bounding_box_latlon():
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

    
