#!/usr/bin/env python
"""

Kriging support functions for NOGGIN.

ML Rilee, RSTLLC, mike@rilee.net for NASA/ACCESS-15/NOGGIN.

"""
import Krige

import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC
from math import *

import pykrige
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
import pykrige.core as core

from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

import h5py
import json

import unittest

###########################################################################

def log10_map(x,eps=1.0e-9,inverse=False):
    """Return the log10 (or 10**) of the argument with a floor of eps."""
    if not inverse:
        return np.log10(x+eps)
    else:
        tmp = np.power(10.0,x)
        if isinstance(tmp, (list, tuple, np.ndarray)):
            tmp[np.where(tmp < eps)] = eps
        elif tmp < eps:
            tmp = eps
        return tmp

###########################################################################

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
                print 'Krige.adaptive_index Too many iterations, beta too small.'\
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
                 ,vmap                    = None\
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
        self.vmap                    = vmap
        self.title                   = title
        self.zVariableName           = zVariableName
        

class krigeResults(object):
    """Capture the result of the kriging calculation and provide a place
to store debugging information as well."""
    def __init__(self\
                 ,z=None,s=None,x=None,y=None\
                 ,z2=None,s2=None,x2=None,y2=None\
                 ,src_z=None,src_x=None,src_y=None\
                 ,dbg_z=None,dbg_x=None,dbg_y=None\
                 ,hull=None\
                 ,box=None\
                 ,zVariableName="z"\
                 ,title='KrigeResultTitle'\
                 ,vg_name=None\
                 ,vg_function=None\
                 ,vg_parameters=None\
                 ,npts=None\
                 ,log_calc=None\
                 ,note='Default note for krigeResults'\
                 ,config=None\
    ):
        self.clear()
        self.config = config
        if x is not None:
            self.x=x.copy()
        if y is not None:
            self.y=y.copy()
        if z is not None:
            self.z=z.copy()
        if x2 is not None:
            self.x2 = x2.copy()
        if y2 is not None:
            self.y2 = y2.copy()
        if z2 is not None:
            self.z2 = z2.copy()
        if z2 is not None:
            self.z2 = z2.copy()
        if x2 is not None:
            self.s2 = s2.copy()
            
        self.zVariableName = zVariableName
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
        self.vg_name = vg_name
        if vg_parameters is not None:
            self.vg_parameters = vg_parameters
        if vg_function is not None:
            self.vg_function = vg_function
        # Number of points sampled
        self.npts = npts
        self.log_calc = log_calc
        self.note = str(note)
        self.sort_on_longitude()
        self.construct_hull()
    def sort_on_longitude_xyz(self):
        if self.x is not None \
           and self.y is not None \
           and self.z is not None:
            idx = self.x.argsort()
            # print '100: x.shape: ',self.x.shape
            # print '110: y.shape: ',self.y.shape
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
        self.config = None
        self.x      = None
        self.y      = None
        self.z      = None
        self.s      = None
        self.src_x  = None
        self.src_y  = None
        self.src_z  = None
        self.dbg_x  = None
        self.dbg_y  = None
        self.dbg_z  = None
        self.hull   = None
        self.box    = None
        self.note   = 'Default note for krigeResults'
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

class krigeHDF(object):
    def __init__(self\
                 ,krg_z=None,krg_s=None,krg_x=None,krg_y=None\
                 ,src_z=None,src_x=None,src_y=None\
                 ,orig_z=None,orig_x=None,orig_y=None\
                 ,krg_name='krg_name'\
                 ,src_name='src_name'\
                 ,orig_name='orig_name'\
                 ,krg_units='kr_units'\
                 ,src_units='kr_units'\
                 ,orig_units='kr_units'\
                 ,output_filename ='krigeHDF-test.hdf'\
                 ,config = None\
                 ,redimension = True\
                 ):
        """Support saving krige results in an HDF (EOS) compatible way, so that tools like Panoply might be used.

Krige results are to be stored in the krig_? arguments.

Krige inputs are to be stored in src_? arguments, which may have been sampled from the original data.

The true input data is to be stored in orig_? arguments. 

The shape of the orig_? arrays is used to format the datasets written to output file, and np.nan is expected as a fill value.
"""
        self.krg_z           = krg_z
        self.krg_s           = krg_s
        self.krg_x           = krg_x
        self.krg_y           = krg_y
        self.krg_name        = krg_name
        self.krg_units       = krg_units
        self.src_z           = src_z
        self.src_x           = src_x
        self.src_y           = src_y
        self.src_name        = src_name
        self.src_units       = src_units
        self.orig_z          = orig_z
        self.orig_x          = orig_x
        self.orig_y          = orig_y
        self.orig_name       = orig_name
        self.orig_units      = orig_units
        self.output_filename = output_filename
        self.config          = config

        if (self.orig_z is None):
            #  or (self.krg_z is None):
            orig_and_krg = None
        else:
            orig_and_krg = np.zeros(self.orig_x.shape)
        if self.krg_z is not None:
            krg = np.zeros(self.orig_x.shape)
            krg[:,:] = np.nan
        else:
            krg = None
        if self.krg_s is not None:
            s = np.zeros(self.orig_x.shape)
            s[:,:] = np.nan
        else:
            s = None
        if self.orig_z is not None:
            orig_and_krg[:,:] = self.orig_z

        if redimension:
            if (self.krg_x is not None) and (self.orig_x is not None):
                for i in range(len(self.krg_x)):
                    x = self.krg_x[i]
                    y = self.krg_y[i]
                    idx = np.where( (self.orig_x == x) & (self.orig_y == y) )
                    # If len(idx) != 2 error.
                    orig_and_krg[idx[0],idx[1]] = self.krg_z[i]
                    krg[idx[0],idx[1]] = self.krg_z[i]
                    s  [idx[0],idx[1]] = self.krg_s[i]
        else:
            x = self.krg_x
            y = self.krg_y
            idx = np.where( orig_and_krg == np.nan )
            # TODO check to see if krg_z and orig_z have the same shape
            orig_and_krg[idx] = self.krg_z[idx]
            krg = self.krg_z
            s   = self.krg_s
                
                    
        self.x2 = self.orig_x
        self.y2 = self.orig_y
        self.z2 = orig_and_krg
        self.orig_and_krg = orig_and_krg
        self.krg = krg
        self.s   = s
        
    def save(self):
        """Save the krige, source, and krige+source data to a file. Should figure out a way to control the format of the file and provide information about the 'grid.' This is necessary if we want the krige output to be compatible with NOGGIn regridding."""
        with h5py.File(self.output_filename,'w') as f:

            nx,ny = self.x2.shape

            # Group: /HDFEOS
            grp_1 = f.create_group('HDFEOS')
            
            # Group: /HDFEOS/NOGGIN
            grp_2 = grp_1.create_group('NOGGIN')
            
            # Group: /HDFEOS/NOGGIN/KrigeResult1
            grp_3 = grp_2.create_group('KrigeResult1')

            if self.config is not None:
                grp_4 = grp_3.create_group('KrigeCalculationConfiguration')
                # TODO: Encapsulate this logic in a configuration object.
                dset = grp_4.create_dataset('configuration.json',data=self.config.as_json())
                dset.attrs['json'] = self.config.as_json()
            
            # Group: /HDFEOS/SWATHS/Swath2953/Geolocation Fields (HPD original)
            # Group: /HDFEOS/NOGGIN/KrigeResult1/Geolocation Fields
            grp_4 = grp_3.create_group('Geolocation Fields')

# TODO need to go back to some sort of 2D array.
            # Dataset: /HDFEOS/NOGGIN/KrigeResult1/Geolocation Fields/Latitude
            dt = np.dtype('<f8')
            dset = grp_4.create_dataset('Latitude', (nx,ny), maxshape=(nx,ny), dtype=dt)
            # initialize dataset values here
            dset[:,:] = self.y2
            
            # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Geolocation Fields/Latitude
            dset.attrs['units'] = "degrees_north"
            
            # Dataset: /HDFEOS/NOGGIN/KrigeResult1/Geolocation Fields/Longitude
            dt = np.dtype('<f8')
            dset = grp_4.create_dataset('Longitude', (nx,ny), maxshape=(nx,ny), dtype=dt)
            # initialize dataset values here
            dset[:,:] = self.x2
            
            # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Geolocation Fields/Longitude
            dset.attrs['units'] = "degrees_east"
            
            # Dataset: /HDFEOS/NOGGIN/KrigeResult1/Geolocation Fields/Time
            dt = np.dtype('<f8')
            dset = grp_4.create_dataset('Time', (1,), maxshape=(1,), dtype=dt)
            # initialize dataset values here
            dset[:] = 0.0
                        
            # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Geolocation Fields/Time
            # TODO This is probably incorrect
            dset.attrs['units'] = "seconds since 1993-01-01 00:00:00.000000Z"
            
            # Group: /HDFEOS/NOGGIN/KrigeResult1/Data Fields
            grp_4 = grp_3.create_group('Data Fields')

            if self.orig_and_krg is not None:
                # Dataset: /HDFEOS/NOGGIN/KrigeResult1/Data Fields/temperature
                dt = np.dtype('<f8')
                dset = grp_4.create_dataset(self.orig_name.split('/')[-1]+'_orig_and_krg', (nx,ny), maxshape=(nx,ny), dtype=dt)
                # initialize dataset values here
                dset[:,:] = self.orig_and_krg
                # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Data Fields/temperature
                dset.attrs['units'] = self.krg_units
                dset.attrs['coordinates'] = "latitude longitude"
                dset.attrs['source_variable'] = self.orig_name

            if self.orig_z is not None:
                dset = grp_4.create_dataset(self.orig_name.split('/')[-1], (nx,ny), maxshape=(nx,ny), dtype=dt)
                # initialize dataset values here
                dset[:,:] = self.orig_z
                # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Data Fields/temperature
                dset.attrs['units'] = self.orig_units
                dset.attrs['coordinates'] = "latitude longitude"
                dset.attrs['source_variable'] = self.orig_name

            if self.krg is not None:
                dset = grp_4.create_dataset(self.krg_name.split('/')[-1], (nx,ny), maxshape=(nx,ny), dtype=dt)
                # initialize dataset values here
                dset[:,:] = self.krg
                # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Data Fields/temperature
                dset.attrs['units'] = self.krg_units
                dset.attrs['coordinates'] = "latitude longitude"
                dset.attrs['source_variable'] = self.orig_name

            if self.s is not None:
                dset = grp_4.create_dataset(self.krg_name.split('/')[-1]+"_uncertainty", (nx,ny), maxshape=(nx,ny), dtype=dt)
                # initialize dataset values here
                dset[:,:] = np.sqrt(self.s)
                # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Data Fields/temperature
                dset.attrs['units'] = self.krg_units
                dset.attrs['coordinates'] = "latitude longitude"
                dset.attrs['source_variable'] = self.orig_name

            if (self.s is not None) and (self.krg is not None):
                dset = grp_4.create_dataset(self.krg_name.split('/')[-1]+"_relative_uncertainty", (nx,ny), maxshape=(nx,ny), dtype=dt)
                # initialize dataset values here
                tmp = np.zeros(self.s.shape)
                tmp[:,:] = np.sqrt(self.s)/self.krg
                tmp[np.where(tmp == np.inf)] = np.nan
                dset[:,:] = tmp 
                # Creating attributes for /HDFEOS/NOGGIN/KrigeResult1/Data Fields/temperature
                dset.attrs['units'] = 'dimensionless'
                dset.attrs['coordinates'] = "latitude longitude"
                dset.attrs['source_variable'] = self.orig_name
            
            # Group: /HDFEOS INFORMATION
            grp_1 = f.create_group('HDFEOS INFORMATION')
            
            # Dataset: /HDFEOS INFORMATION/StructMetadata.0
            dt = np.dtype('S1')
            dset = grp_1.create_dataset('StructMetadata.0', (), dtype=dt)
            # initialize dataset values here
            
            # METADATA STRUCTURE
            
            # Close the file
            # f.close()
            
            # dset = f.create_dataset('metadata.json'\
            #                         ,data=json.dumps(\
            #                                          {"npts":self.npts\
            #                                           ,"log_calc":self.log_calc\
            #                                           ,"note":self.note\
            #                                           ,"vg_name":self.vg_name\
            #                                          }\
            #                                          ,sort_keys=True))
            
            # dset = f.create_dataset(self.zVariableName,data=self.z)
            # dset = f.create_dataset(self.zVariableName+'_s',data=self.s)
            # dset = f.create_dataset('x',data=self.x)
            # dset = f.create_dataset('y',data=self.y)


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

class krigeConfig(object):
    def __init__(self\
                 ,npts                 = None\
                 ,beta0                = None\
                 ,frac                 = None\
                 ,l                    = None\
                 ,w                    = None\
                 ,model_name           = None\
                 ,nlags                = None\
                 ,weight               = None\
                 ,parameters           = None\
                 ,grid_stride          = None\
                 ,random_permute       = None\
                 ,coordinates_type     = None\
    ):
        self.noggin_krige_version = Krige.__version__
        self.pykrige_version      = pykrige.__version__
        self.npts                 = npts
        self.beta0                = beta0
        self.frac                 = frac
        self.l                    = l
        self.w                    = w
        self.model_name           = model_name
        self.nlags                = nlags
        self.weight               = weight
        self.parameters           = parameters
        self.grid_stride          = grid_stride
        self.random_permute       = random_permute
        self.coordinates_type     = coordinates_type
    def as_dict(self):
        return \
            {\
             'noggin_krige_version': self.noggin_krige_version\
             ,'pykrige_version'    : self.pykrige_version\
             ,'coordinates_type'   : self.coordinates_type\
             ,'adaptive_sampling'  : \
             {\
              'npts'   : self.npts\
              ,'beta0' : self.beta0\
              ,'frac'  : self.frac\
              ,'l'     : self.l\
              ,'w'     : self.w\
             }\
             ,'variogram':\
             {\
              'model_name' : self.model_name\
              ,'nlags'     : self.nlags\
              ,'weight'    : self.weight\
              ,'parameters': self.parameters\
             }\
             ,'calculation':\
             {\
              'grid_stride'     : self.grid_stride\
              ,'random_permute' : self.random_permute\
              }}
    def as_json(self):
        return json.dumps(self.as_dict())
    # TODO def from_json(self)...


###########################################################################

# TODO Add option to turn off adaptive sampling. Notes this would affect
# TODO the bounding box (l,w) that limits the data used.

def drive_OKrige(
        x,y
        ,src_x,src_y,src_z
        ,log_calc=True
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
        if log_calc:
            data_z1= np.log(data_z)
        else:
            data_z1= np.copy(data_z)

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
            if log_calc:
	        gridz [isample] = np.exp(z[:])
            else:
	        gridz [isample] = z[:]
	    gridss[isample] = ss[:]
    # TODO Need to return gridss
    print 1000
    config = krigeConfig(\
                         npts                  = npts\
                         ,beta0                = beta0\
                         ,frac                 = frac\
                         ,l                    = l\
                         ,w                    = w\
                         ,model_name           = variogram_model\
                         ,nlags                = nlags\
                         ,weight               = weight\
                         ,parameters           = variogram_parameters\
                         ,grid_stride          = grid_stride\
                         ,random_permute       = random_permute\
                         ,coordinates_type     = 'geographic'\
    )
    return krigeResults(s              = gridss\
                        ,z             = gridz\
                        ,x             = x\
                        ,y             = y\
                        ,src_z         = data_z\
                        ,src_x         = data_x\
                        ,src_y         = data_y\
                        ,vg_function   = variogram_function\
                        ,vg_parameters = variogram_parameters\
                        ,config        = config\
                        )
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

class Test_log10_map(unittest.TestCase):
    
    def test_log10_map(self):
        eps = 1.0e-10
        # eps = 0
        self.assertLess(np.abs(log10_map(10)-1),eps)
        self.assertLess(np.abs(log10_map(100)-2),eps)
        self.assertLess(np.abs(log10_map(2,inverse=True)-100),eps)
        self.assertLess(np.abs(log10_map(3,inverse=True)-1000),eps)
        self.assertLess(np.sum((log10_map(np.arange(3),inverse=True)-[1,10,100])**2),eps**2)
        self.assertLess(np.sum((log10_map(np.power(10.0,np.arange(3)))-[0,1,2])**2),100.0*eps**2)


class Test_krigeConfig(unittest.TestCase):
    def test_json(self):
        i = 0
        npts                 = i; i = i+1
        beta0                = i; i = i+1
        frac                 = i; i = i+1
        l                    = i; i = i+1
        w                    = i; i = i+1
        model_name           = i; i = i+1
        nlags                = i; i = i+1
        weight               = i; i = i+1
        parameters           = i; i = i+1
        grid_stride          = i; i = i+1
        random_permute       = i; i = i+1
        
        config = krigeConfig(\
                             npts                  = npts\
                             ,beta0                = beta0\
                             ,frac                 = frac\
                             ,l                    = l\
                             ,w                    = w\
                             ,model_name           = model_name\
                             ,nlags                = nlags\
                             ,weight               = weight\
                             ,parameters           = parameters\
                             ,grid_stride          = grid_stride\
                             ,random_permute       = random_permute\
                             ,coordinates_type     ='geographic'\
        )
        self.assertEqual(config.as_json()\
                         ,'{"variogram": {"parameters": 8, "nlags": 6, "model_name": 5, "weight": 7}, "calculation": {"grid_stride": 9, "random_permute": 10}, "pykrige_version": "1.4.0.NOGGIn.0", "noggin_krige_version": "0.0.1", "adaptive_sampling": {"npts": 0, "frac": 2, "l": 3, "w": 4, "beta0": 1}, "coordinates_type": "geographic"}')
        
### delete #### class Test_krigeResult(unittest.TestCase):
### delete #### # Need to use a SWATH mode for the krigeResult
### delete ####     
### delete ####     def test_save(self):
### delete ####         output_filename = 'test.hdf5'
### delete ####         x,y = np.meshgrid(np.arange(-180,180),np.arange(-90,90))
### delete ####         x = x.reshape(180/4,360*4)
### delete ####         y = y.reshape(180/4,360*4)
### delete ####         z   = x**2 +y**2
### delete ####         x1 = np.ravel(x)
### delete ####         y1 = np.ravel(y)
### delete ####         z1 = np.ravel(z)
### delete ####         print 'x.shape:  ',x.shape
### delete ####         print 'y.shape:  ',y.shape
### delete ####         print 'z.shape:  ',z.shape
### delete ####         print 'x1.shape: ',x1.shape
### delete ####         print 'y1.shape: ',y1.shape
### delete ####         print 'z1.shape: ',z1.shape
### delete ####         kr=krigeResults(vg_name        = 'vg_name'\
### delete ####                         ,zVariableName = 'z1'\
### delete ####                         ,x             = x1\
### delete ####                         ,y             = y1\
### delete ####                         ,z             = z1\
### delete ####                         ,s             = -z1\
### delete ####                         ,x2            = x\
### delete ####                         ,y2            = y\
### delete ####                         ,z2            = z\
### delete ####                         ,s2            = -z
### delete ####                         )
### delete ####         kr.save(output_filename=output_filename)
### delete #### 
### delete ####         # print 'kr.z: ',kr.z
### delete ####         
### delete ####         metadata = ''
### delete ####         with h5py.File(output_filename,'r') as f:
### delete ####             metadata = json.loads(f['metadata.json'][()])
### delete ####             self.assertEqual(metadata['vg_name']   ,kr.vg_name)
### delete ####             self.assertEqual(metadata['npts']      ,kr.npts)
### delete ####             self.assertEqual(metadata['log_calc']  ,kr.log_calc)
### delete ####             # z = f['z'][:]
### delete ####             # self.assertEqual(z[0],3.0)
### delete ####             # self.assertEqual(z[1],2.0)
### delete ####             # self.assertEqual(z[3],1.0)
    
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
    print
    print 'Krige version:   ',Krige.__version__
    print 'PyKrige version: ',pykrige.__version__
    if False:
        fig_gen = fig_generator(1,1)
        fig_gen.increment_figure()
        fig, ax = fig_gen.get_fig_axes()

    unittest.main()

    
