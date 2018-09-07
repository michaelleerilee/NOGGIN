#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare a (MODIS, etc.) datafield for further processing using python tools. Provide a basic viewing capability.

2018-0423-1352-44-EDT ML Rilee, RSTLLC, mike@rilee.net.

------


Copyright © 2018 Michael Lee Rilee, mike@rilee.net, Rilee Systems Technologies LLC

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

"""

import json
import os
import numpy as np
from pyhdf.SD import SD, SDC
import h5py as h5

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from Krige import fig_generator, data_src_directory

import unittest

class Interval(object):
    def __init__(self):
        """An interval."""
        self.x0 = None
        self.x1 = None
    def __init__(self,lst=None):
        if lst is None:
            self.x0 = None
            self.x1 = None
        else:            
            self.x0 = lst[0]
            self.x1 = lst[1]
    def tuple(self):
        return (self.x0,self.x1)
    def list(self):
        return [self.x0,self.x1]
    def centroid(self):
        return 0.5*(self.x0+self.x1)
    def emptyp(self):
        return (self.x0 is None) and (self.x1 is None)
    def contains(self,x):
        return (self.x0 <= x) and (x <= self.x1)
    def overlap(self,z):
        a0 = self.x0
        a1 = self.x1
        b0 = z.x0
        b1 = z.x1

        l0 = float('-inf')
        l1 = float( 'inf')
        if (b1 < a0) or (a1 < b0):
            return Interval()
        else:
            if a0 >= b0:
                l0 = a0
            else:
                l0 = b0
            if a1 <= b1:
                l1 = a1
            else:
                l1 = b1
        return Interval((l0,l1))


class LonInterval(Interval):
    """Intervals in longitude, always ordered northward, i.e. from low to
    high longitude (right-hand-rule).

    The ordering, from west to east, implies that if a southward
    interval is found, it crosses the international date line.

    Currently we presume were on the -180..180 branch and 'split'
    intervals that cross the date line, when needed.

    TODO: Value checking, maybe branch definition.

    """
    def __init__(self,lst=None):
        if lst is None:
            self.x0 = None
            self.x1 = None
        else:
            self.x0 = lst[0]
            self.x1 = lst[1]
    def to_Intervals(self):
        if self.x1 >= self.x0:
            return [Interval([self.x0,self.x1])]
        else:
            return \
                [ Interval([-180,self.x1]) \
                  ,Interval([self.x0,180]) ]
    def centroid(self):
        if self.x1 >= self.x0:
            return super(LonInterval,self).centroid()
        else:
            return Interval([self.x0,self.x1+360]).centroid()
        
    def contains(self,x):
        if self.x1 >= self.x0:
            return super(LonInterval,self).contains(x)
        else:
            return \
                Interval([self.x0,180]).contains(x) \
                or Interval([-180,self.x1]).contains(x)

    def overlap(self,z):
        """Return a list of *Interval* containing the overlap of this and another LonInterval.

        Returns [] if there is no overlap.

        TODO Maybe return an empty LonInterval?
        """
        lst = [ i0.overlap(i1) for i0 in self.to_Intervals() for i1 in z.to_Intervals() ]
        return [ i for i in lst if not i.emptyp() ]


class Point(object):
    """A point on the sphere with coordinates in degrees longitude and latitude."""
    lat_degrees=0.0
    lon_degrees=0.0
    def __init__(self):
        """A point on the sphere."""
    def __init__(self,lonlat_degrees=None,latlon_degrees=None):
        """Make a point from a tuple or the individual lat and lon"""
        assert not isinstance(lonlat_degrees,basestring)
        assert not isinstance(latlon_degrees,basestring)
        if lonlat_degrees is not None:
            self.lon_degrees = lonlat_degrees[0]
            self.lat_degrees  = lonlat_degrees[1]
        elif latlon_degrees is not None:
            self.lon_degrees = latlon_degrees[1]
            self.lat_degrees  = latlon_degrees[0]
        else:
            self.lon_degrees = None
            self.lat_degrees = None
    def inDegrees(self,order="lonlat"):
        if   order == "lonlat":
            return (self.lon_degrees,self.lat_degrees)
        elif order == "latlon":
            return (self.lat_degrees,self.lon_degrees)
        return "Point.inDegrees: unkown argument: order = "+order
    def to_xml(self):
        return "<Point lon_degrees="+str(self.lon_degrees)+" lat_degrees="+str(self.lat_degrees)+"/>"
    def copy(self):
        return Point((self.lon_degrees,self.lat_degrees))

def box_covering(lon,lat,hack_branchcut_threshold=-1):
    """Make a BB covering, arguments (lon,lat) are arrays of lat and lon in degrees.

    Make the assumption that we are dealing with compact clouds of
    points like you'll get in a granule. We'll detect those that cross
    the dateline by checking if the extent of the cloud exceeds some
    threshold.

    hack_branchcut_threshold is a start at handling the lon=+/-180
    periodic boundary.  If hack_branchcut_threshold > 0 and the extent
    of lon exceeds this threshold, then add 360 to bring the minimum
    around to the right sheet.

    """
    if hack_branchcut_threshold > 0:
        lon0=np.nanmin(lon)
        lon1=np.nanmax(lon)
        if lon1-lon0 > hack_branchcut_threshold:
            _lon = np.copy(lon)
            idltz=np.where(_lon < 0)
            _lon[idltz] = _lon[idltz] + 360.0
            _lon_mn = np.nanmin(_lon)
            if _lon_mn > 180:
                _lon_mn = _lon_mn - 360
            _lon_mx = np.nanmax(_lon)
            if _lon_mx > 180:
                _lon_mx = _lon_mx - 360
            # print 100,np.nanmin(_lon)
            # print 101,np.nanmax(_lon)
            # print 102,lon
            # print 103,lat
            # print 104,_lon
            return BoundingBox(( Point((_lon_mn,np.nanmin(lat)))\
                                 ,Point((_lon_mx,np.nanmax(lat))) ))
        
    return BoundingBox(( Point((np.nanmin(lon),np.nanmin(lat)))\
                         ,Point((np.nanmax(lon),np.nanmax(lat))) ))

class Polygon(object):
    def __init__(self):
        self.lons=[]
        self.lats=[]
    def __init__(self,lons_lats):
        self.lons=lons_lats[0]
        self.lats=lons_lats[1]
    def add_point(self,lon_lat):
        self.lons.add(lon_lat[0])
        self.lats.add(lon_lat[1])
    def polygon(self):
        return self.lons,self.lats

class BoundingBox(object):
    p0 = None; p1 = None
    def __init__(self):
        """A box."""
    def __init__(self,p=(Point(),Point())):
        self.init(p)
        # self.p0 = p[0]
        # self.p1 = p[1]
        # self.lon_interval = Interval([self.p0.lon_degrees, self.p1.lon_degrees])
        # self.lat_interval = Interval([self.p0.lat_degrees, self.p1.lat_degrees])
    def init(self,p=(Point(),Point())):
        self.p0 = p[0]
        self.p1 = p[1]
        self.lon_interval = LonInterval([self.p0.lon_degrees, self.p1.lon_degrees])
        self.lat_interval = Interval([self.p0.lat_degrees, self.p1.lat_degrees])
    def copy(self):
        """Return a deep copy of this object."""
        return BoundingBox((self.p0.copy(),self.p1.copy()))
    def centroid(self):
        lon0 = self.lon_interval.centroid()
        lat0 = self.lat_interval.centroid()
        return Point((lon0,lat0))
    def emptyp(self):
        return (self.lon_interval.emptyp() or self.lat_interval.emptyp())
    def to_xml(self):
        return \
            "<BoundingBox>\n"\
            "  "+self.p0.to_xml()+"\n"\
            "  "+self.p1.to_xml()+"\n"\
            +"</BoundingBox>\n"
    def to_json(self):
        return json.dumps(['BoundingBox',{ 'p0':tuple(map(float,self.p0.inDegrees()))\
                                          ,'p1':tuple(map(float,self.p1.inDegrees())) }])
        # p0lon,p0lat = tuple(map(float,self.p0.inDegrees()))
        # p1lon,p1lat = tuple(map(float,self.p1.inDegrees()))
        # # print 'type-p0lon: ',type(p0lon)
        # return json.dumps(['BoundingBox',{'p0': (p0lon,p0lat),'p1': (p1lon,p1lat) }])
        # return json.dumps(['BoundingBox',{'p0':self.p0.inDegrees(),'p1':self.p1.inDegrees()}])
        # return json.dumps(['BoundingBox',{'p0':str(type(self.p0.inDegrees())),'p1':str(self.p1.inDegrees())}])
    def from_json(self,json_str):
        res = json.loads(json_str)
        points = res[1]
        # Can we just call __init__ again?
        self.init((Point(points['p0']),Point(points['p1'])))
        # self.p0 = Point(points['p0'])
        # self.p1 = Point(points['p1'])
        # self.lon_interval = Interval([self.p0.lon_degrees, self.p1.lon_degrees])
        # self.lat_interval = Interval([self.p0.lat_degrees, self.p1.lat_degrees])
        return self
    
    def overlap(self,other_box):
        """Intersection
        
        Returns a list of BoundingBoxes corresponding to intersections
        of this and the other_box.
        """

        if self.emptyp():
            return []

        if other_box.emptyp():
            return []
        
        lon_overlaps = self.lon_interval.overlap(other_box.lon_interval)
        if len(lon_overlaps) == 0:
            return []
        elif lon_overlaps[0].emptyp():
            return []
        
        lat_overlap = self.lat_interval.overlap(other_box.lat_interval)
        if lat_overlap.emptyp():
            return []

        o_lon_interval = other_box.lon_interval
        if o_lon_interval.emptyp():
            return []

        o_lat_interval = other_box.lat_interval
        if o_lat_interval.emptyp():
            return []

        return \
              [ BoundingBox( (Point((lon.x0,lat_overlap.x0))\
                             ,Point((lon.x1,lat_overlap.x1))) ) for lon in lon_overlaps ]
               
#        return BoundingBox(( Point((lon_overlaps.x0,lat_overlap.x0))\
#                           ,Point((lon_overlaps.x1,lat_overlap.x1)) ))

    def lons_lats(self):
        if self.emptyp():
            return [],[]
        else:
            return [self.p0.lon_degrees,self.p1.lon_degrees],[ self.p0.lat_degrees,self.p1.lat_degrees]

    def polygon(self):
        """
        Return a list of polygons corresponding to the bounding box in -180..180.
        """
        if self.p1.lon_degrees > self.p0.lon_degrees:
            return \
                [Polygon((\
                         [\
                          self.p0.lon_degrees
                          ,self.p1.lon_degrees
                          ,self.p1.lon_degrees
                          ,self.p0.lon_degrees
                         ]\
                         ,\
                         [\
                          self.p0.lat_degrees
                          ,self.p0.lat_degrees
                          ,self.p1.lat_degrees
                          ,self.p1.lat_degrees
                         ]))]
        else:
            return \
                [Polygon((\
                          [\
                           self.p0.lon_degrees
                           ,180
                           ,180
                           ,self.p0.lon_degrees
                          ]\
                          ,\
                          [\
                           self.p0.lat_degrees
                           ,self.p0.lat_degrees
                           ,self.p1.lat_degrees
                           ,self.p1.lat_degrees
                          ])),
                 Polygon((\
                          [\
                           -180
                           ,self.p1.lon_degrees
                           ,self.p1.lon_degrees
                           ,-180
                          ]\
                          ,\
                          [\
                           self.p0.lat_degrees
                           ,self.p0.lat_degrees
                           ,self.p1.lat_degrees
                           ,self.p1.lat_degrees
                          ]))
                ]

            
    def union(self,other_box,hack_branchcut_threshold=-1):
        lons,lats   = self.lons_lats()
        olons,olats = other_box.lons_lats()
        return box_covering(np.array(lons+olons),np.array(lats+olats)\
                            ,hack_branchcut_threshold=hack_branchcut_threshold)

    def contains_point(self,p):
        lon,lat=p.inDegrees()
        return self.lon_interval.contains(lon) and self.lat_interval.contains(lat)

class DataField(object):
    """Access a datafield in an HDFEOS (was MODIS) file"""
    # TODO Add an object or classes that can handle swath or grid
    # TODO rename key_*...
    # key along, across are now dated...
    key_along  ='Cell_Along_Swath_5km:mod05'
    key_across ='Cell_Across_Swath_5km:mod05'
    key_units  ='units'
    units      ='units'
    nAlong  = 0
    nAcross = 0
    m = None
    colormesh_title = 'colormesh_title'
    long_name = 'long_name'

    def __init__(self):
        """Default constructor"""

    def __init__(self,datafilename=None,datafieldname=None,srcdirname="./",geofile=""\
                 ,data=None,latitude=None,longitude=None\
                 ,title='title'
                 ,colormesh_title='colormesh_title'\
                 ,long_name='long_name'\
                 ,hack_branchcut_threshold=0\
                 ,custom_loader=None\
    ):
        """

custom_loader=None.  A callable(self) that allows a user to write a
  custom loader using a visitor pattern
  (https://en.wikipedia.org/wiki/Visitor_pattern). The methods load05
  and load08 can be used as patterns for writing the callable.

        """
        self.title                    = title
        self.colormesh_title          = colormesh_title
        self.long_name                = long_name
        self.datafilename             = datafilename
        self.datafieldname            = datafieldname
        self.data                     = data
        self.latitude                 = latitude
        self.longitude                = longitude
        self.srcdirname               = srcdirname
        self.geofile                  = geofile
        self.hack_branchcut_threshold = hack_branchcut_threshold
        self.custom_loader            = custom_loader
        # Should test that either datafilename or data is set.
        if self.datafilename is not None:
            self.load()
        else:
            self.plot_lat_m_center = np.nanmean(self.latitude)
            self.plot_lon_m_center = np.nanmean(self.longitude)

    def load(self):
        # TODO MLR Verify this use of callable works.
        if callable(self.custom_loader):
            custom_loader(self)
            return
        if self.datafilename[0:3] in ["MOD","MYD"]:
            if self.datafilename[3:5] == '05':
                self.load05()
            elif self.datafilename[3:5] == '08':
                self.load08()
            else:
                print 'failed loading '+self.datafilename
                print 'loading type '+self.datafilename[0:5]+' not supported. returning...'
                return
        elif self.datafilename[0:3] in ["OMI"]:
            # How did OMI get in a MODIS object?
            self.loadOMI_L3()
        else:
            print 'failed loading '+self.datafilename
            print 'loading type '+self.datafilename[0:3]+' not supported. returning...'


    def load05(self):
        hdf = SD(self.srcdirname+self.datafilename, SDC.READ)
        ds = hdf.select(self.datafieldname)

            # ds_dims = [ds.dimensions()[i] for i in ds.dimensions().keys()]

        self.colormesh_title = self.datafilename

        # TODO MLR There must be a better way to get this information!!!
        # print 'ds-keys: ',ds.dimensions().keys()
        if "Cell_Along_Swath_1km:mod05" in ds.dimensions().keys():
            # Need to load geolocation data from MOD03
            if self.geofile == "":
                raise ValueError("Determined external geolocation data needed, but geofile not set to MOD03...")
            geo = SD(self.srcdirname+self.geofile,SDC.READ)
            self.key_along  = "Cell_Along_Swath_1km:mod05"
            self.key_across = "Cell_Across_Swath_1km:mod05"
            self.key_units  = 'unit'
        elif "Cell_Along_Swath_5km:mod05" in ds.dimensions().keys():
            self.key_along  = "Cell_Along_Swath_5km:mod05"
            self.key_across = "Cell_Across_Swath_5km:mod05"
            self.key_units  = 'units'
        else:
            raise ValueError("Can't determine geolocation information from 'Cell_Along_Swath_?km'.")
            
        nAlong  = ds.dimensions()[self.key_along]
        nAcross = ds.dimensions()[self.key_across]

        data = np.zeros((nAlong,nAcross))
        data = ds[0:nAlong,0:nAcross].astype(np.double)
        
        attrs        = ds.attributes(full=1)
        lna          = attrs["long_name"]
        self.long_name    = lna[0]
        aoa          = attrs["add_offset"]
        add_offset   = aoa[0]
        fva          = attrs["_FillValue"]
        _FillValue   = fva[0]
        sfa          = attrs["scale_factor"]
        scale_factor = sfa[0]
        vra          = attrs["valid_range"]
        valid_min    = vra[0][0]
        valid_max    = vra[0][1]
        ua           = attrs[self.key_units]
        self.units        = ua[0]

        # This sequence follows https://hdfeos.org/zoo/LAADS/MOD05_L2.A2010001.0000.006.2015041171924.hdf.py
        invalid = np.logical_or(data > valid_max, data < valid_min)
        invalid = np.logical_or(invalid,data == _FillValue)
        data[invalid] = np.nan
        data = (data - add_offset) * scale_factor
        self.data = np.ma.masked_array(data, np.isnan(data))

        if self.geofile == "":
            lat = hdf.select('Latitude')
            lon = hdf.select('Longitude')
        else:
            # raise NotImplementedError('separate geofile with location information not implemented!')
            # TODO implement
            lat = geo.select('Latitude')
            lon = geo.select('Longitude')
        self.latitude = lat[:,:]
        self.longitude = lon[:,:]
        self.bbox = box_covering(self.longitude,self.latitude\
                                 ,hack_branchcut_threshold=self.hack_branchcut_threshold\
        )
            
    def load08(self):
        hdf = SD(self.srcdirname+self.datafilename, SDC.READ)
        ds  = hdf.select(self.datafieldname)

        print 'load08'
        print 'ds.dimensions(): ',ds.dimensions()

        # long_name="Atmospheric_Water_Vapor_Mean"
        ### TODO Maybe these do not need to be fields, just local vars
        self.key_along  = 'YDim:mod08'
        self.key_across = 'XDim:mod08'
        
        self.colormesh_title = self.datafilename
        
        nAlong  = ds.dimensions()[self.key_along]
        nAcross = ds.dimensions()[self.key_across]
        print '0 nAlong,nAcross: '+str(nAlong)+', '+str(nAcross)

        data = np.zeros((nAlong,nAcross))
        print 'a data.shape:     '+str(data.shape)
        print 'd ds.data.shape:  '+str(ds[:,:].astype(np.double).shape)
        
        data = ds[0:nAlong,0:nAcross].astype(np.double)

        print '1 nAlong,nAcross: '+str(nAlong)+', '+str(nAcross)
        print '2 data.shape:     '+str(data.shape)
        
        attrs        = ds.attributes(full=1)
        lna          = attrs["long_name"]
        self.long_name    = lna[0]
        aoa          = attrs["add_offset"]
        add_offset   = aoa[0]
        fva          = attrs["_FillValue"]
        _FillValue   = fva[0]
        sfa          = attrs["scale_factor"]
        scale_factor = sfa[0]
        vra          = attrs["valid_range"]
        valid_min    = vra[0][0]
        valid_max    = vra[0][1]
        ua           = attrs[self.key_units]
        self.units        = ua[0]

        invalid = np.logical_or(data > valid_max, data < valid_min)
        invalid = np.logical_or(invalid,data == _FillValue)
        
        data[invalid] = np.nan
        data = (data - add_offset) * scale_factor
        self.data = np.ma.masked_array(data, np.isnan(data))

        if self.geofile == "":
            lat = hdf.select('YDim')
            # lat is 1d, since 08 is a grid
            # self.latitude = lat[:,:]
            lon = hdf.select('XDim')
            # lon is 1d, since 08 is a grid
            # self.longitude = lon[:,:]
            print 'type(lon): '+str(type(lon))
            print 'type(lat): '+str(type(lat))
            print 'lon.shape: '+str(lon[:].shape)
            print 'lat.shape: '+str(lat[:].shape)
            
            self.longitude,self.latitude = np.meshgrid(lon[:],lat[:])
                                           
            self.bbox = box_covering(self.longitude,self.latitude\
                                     ,hack_branchcut_threshold=self.hack_branchcut_threshold\
                                     )
        else:
            raise NotImplementedError('separate geofile with location information not implemented!')
            # TODO implement
        
    def loadOMI_L3(self):

        # e.g. http://hdfeos.org/software/h5py.php
        # TODO: Consider netCDF4 vs. h5py
        print 'loadOMI_L3'
        
        with h5.File(self.srcdirname+self.datafilename, mode='r') as hdf:

            if self.geofile != "":
                raise NotImplementedError('separate geofile with location information not implemented!')
            
            ds  = hdf[self.datafieldname]

            # print 'hdf.keys: ',hdf.keys()
            print 'ds.attrs.keys: ',ds.attrs.keys()

            # long_name="Atmospheric_Water_Vapor_Mean"
            ### TODO Maybe these do not need to be fields, just local vars
                        
            # :NumberOfLongitudesInGrid

            self.key_along  = 'lat'
            self.key_across = 'lon'
            self.colormesh_title = self.datafilename
            self.key_units = 'Units'

            # lat = hdf['lat']
            # lon = hdf['lon']
            lat = np.arange( -90.0, 90.0, 1.0)
            lon = np.arange(-180.0, 180.0, 1.0)

            print 'type(lon): '+str(type(lon))
            print 'type(lat): '+str(type(lat))
            print 'lon.shape: '+str(lon[:].shape)
            print 'lat.shape: '+str(lat[:].shape)
            
            self.longitude,self.latitude = np.meshgrid(lon[:],lat[:])
            
            self.bbox = box_covering(self.longitude,self.latitude\
                                     ,hack_branchcut_threshold=self.hack_branchcut_threshold\
            )
            print 'longitude.shape: ',self.longitude.shape
            print 'latitude.shape:  ',self.latitude.shape
            
            nAlong  = len(lat)
            nAcross = len(lon)
            print '0 nAlong,nAcross: '+str(nAlong)+', '+str(nAcross)

            data = np.zeros((nAlong,nAcross))
            print 'a data.shape:     '+str(data.shape)
            print 'd ds.data.shape:  '+str(ds[:,:].astype(np.double).shape)
            
            data = ds[0:nAlong,0:nAcross].astype(np.double)

            print '1 nAlong,nAcross: '+str(nAlong)+', '+str(nAcross)
            print '2 data.shape:     '+str(data.shape)
        
            # attrs        = ds.attributes(full=1)
            attrs        = ds.attrs
            aoa          = attrs["Offset"]
            add_offset   = aoa[0]
            fva          = attrs["_FillValue"]
            _FillValue   = fva[0]
            sfa          = attrs["ScaleFactor"]
            scale_factor = sfa[0]
            vra          = attrs["ValidRange"]
            valid_min    = vra[0]
            valid_max    = vra[1]
            ua           = attrs[self.key_units]
            self.units        = ua[0]

            print 'aoa: ',aoa
            print 'fva: ',fva
            print 'sfa: ',sfa
            print 'vra: ',vra
            print 'ua:  ',ua

            print '0 mnmx(data): ',np.nanmin(data),np.nanmax(data)

            invalid = np.logical_or(data > valid_max, data < valid_min)
            invalid = np.logical_or(invalid,data == _FillValue)

            data[invalid] = np.nan
            print '1 mnmx(data): ',np.nanmin(data),np.nanmax(data)

            # Not sure about the following. Cribbed from MODIS approach.
            data = (data - add_offset) * scale_factor
            self.data = np.ma.masked_array(data, np.isnan(data))
            print '2 mnmx(self.data): ',np.nanmin(self.data),np.nanmax(self.data)

    def init_basemap(self,ax=None,wh_scale=None\
                         ,lat_center=None, lon_center=None
                         ):
        """
        Initialize basemap visualization.
        """
        if lat_center is None:
            self.plot_lat_m_center = np.nanmean(self.latitude)
        else:
            self.plot_lat_m_center = lat_center
        if lon_center is None:
            self.plot_lon_m_center = np.nanmean(self.longitude)
        else:
            self.plot_lon_m_center = lon_center

        if wh_scale is None:
            self.m = Basemap(projection='cyl', resolution='l'\
                             ,ax=ax\
                             ,lat_0=self.plot_lat_m_center, lon_0=self.plot_lon_m_center)
        else:
            self.m = Basemap(projection='laea', resolution='l', lat_ts=65\
                             ,ax=ax\
                             ,lat_0=self.plot_lat_m_center, lon_0=self.plot_lon_m_center\
                             ,width=wh_scale[0]*3000000,height=wh_scale[1]*2500000)
        
        self.m.drawcoastlines(linewidth=0.5)
        self.m.drawparallels(np.arange(-90.0, 91., 10.), labels=[1, 0, 0, 0])
        self.m.drawmeridians(np.arange(-180, 181., 30), labels=[0, 0, 0, 1])

    def show(self):
        plt.show()
        
    def colormesh(self,m=None,vmin=np.nan,vmax=np.nan\
                      ,plt_show=False,ax=None\
                      ,colorbar=False,title=None):
        # Render the plot in a lambert equal area projection.
        save_m = None
        if m is None:
            if self.m is None:
                self.init_basemap(ax=ax)
        else:
            save_m = self.m
            self.m = m

        if(np.isnan(vmin)):
            vmin = np.nanmin(self.data)
        if(np.isnan(vmax)):
            vmax = np.nanmax(self.data)
        
        # m.scatter(longitude, latitude, c=data, latlon=True)
        self.m.pcolormesh(self.longitude, self.latitude, self.data, latlon=True\
                         ,vmin=vmin,vmax=vmax)
        if colorbar:
            cb=self.m.colorbar()
            cb.set_label(self.units, fontsize=8)

        if title is not None:
            title_ = title
        else:
            title_ = self.colormesh_title
            
        # basename = os.path.basename(self.datafilename),
        basename = os.path.basename(title_)
        plt.title('{0}\n{1}'.format(basename, self.long_name))
        fig = plt.gcf()
        # pngfile = "{0}.py.png".format(basename)
        # fig.savefig(pngfile)
        if plt_show:
            plt.show()
        if save_m is not None:
            self.m = save_m

    def scatterplot(self,m=None,vmin=np.nan,vmax=np.nan\
                    ,plt_show=True,ax=None\
                    ,marker_size=1\
                    ,colorbar=False\
                    ,title=None\
                    ,cmap=None\
                    ,value_map=None\
                    ,edgecolors=None\
    ):
        # Render the plot in a lambert equal area projection.
        save_m = None
        if m is None:
            if self.m is None:
                self.init_basemap(ax=ax)
        else:
            save_m = self.m
            self.m = m

        if(np.isnan(vmin)):
            vmin = np.nanmin(self.data)
        if(np.isnan(vmax)):
            vmax = np.nanmax(self.data)

        sc = None
        if value_map is None:
            if cmap is None:
                sc = self.m.scatter(self.longitude, self.latitude, c=self.data, latlon=True\
                                    ,vmin=vmin,vmax=vmax\
                                    ,s=marker_size\
                                    ,edgecolors=edgecolors\
                )
            else:
                sc = self.m.scatter(self.longitude, self.latitude, c=self.data, latlon=True\
                                    ,vmin=vmin,vmax=vmax\
                                    ,s=marker_size\
                                    ,cmap=cmap
                                    ,edgecolors=edgecolors\
                )
        else:
            if cmap is None:
                sc = self.m.scatter(self.longitude, self.latitude\
                                    ,c=value_map(self.data)\
                                    ,latlon=True\
                                    ,vmin=vmin,vmax=vmax\
                                    ,s=marker_size\
                                    ,edgecolors=edgecolors\
                )
            else:
                sc = self.m.scatter(self.longitude, self.latitude\
                                    ,c=value_map(self.data)\
                                    ,latlon=True\
                                    ,vmin=vmin,vmax=vmax\
                                    ,s=marker_size\
                                    ,cmap=cmap
                                    ,edgecolors=edgecolors\
                )
            
        # self.m.pcolormesh(self.longitude, self.latitude, self.data, latlon=True\
        #                 ,vmin=vmin,vmax=vmax)
        if colorbar:
            # fig=plt.gcf()
            # cb = fig.colorbar(sc,ax=ax)
            cb = self.m.colorbar(sc)
            cb.set_label(self.units, fontsize=8)

        if title is not None:
            title_ = title
        else:
            title_ = self.colormesh_title
            
        # basename = os.path.basename(self.datafilename),
        basename = os.path.basename(title_)
        plt.title('{0}\n{1}'.format(basename, self.long_name))
        fig = plt.gcf()
        # pngfile = "{0}.py.png".format(basename)
        # fig.savefig(pngfile)
        if plt_show:
            plt.show()
        if save_m is not None:
            self.m = save_m

    def get_m(self):
        return self.m
        
    def ravel(self):
        return \
          np.ravel(self.data),np.ravel(self.latitude),np.ravel(self.longitude)

    def data_mnmx(self):
        return (np.nanmin(self.data),np.nanmax(self.data))


class TestPoint(unittest.TestCase):

    def test_Point(self):
        self.assertEqual((None,None),Point().inDegrees())
        self.assertEqual((1,0),Point(latlon_degrees=(0,1)).inDegrees())
        self.assertEqual((2,3),Point((2,3)).inDegrees())
        self.assertEqual('<Point lon_degrees=None lat_degrees=None/>',Point().to_xml())
        self.assertEqual('<Point lon_degrees=0 lat_degrees=1/>',Point((0,1)).to_xml())

    def test_Interval(self):
        self.assertEqual((None,None),Interval().tuple())
        self.assertEqual((0,1),Interval([0,1]).tuple())
        self.assertEqual([1,2],Interval(lst=[1,2]).list())
        self.assertEqual([2,1],Interval([2,1]).list())

    def test_IntervalEmpty(self):
        self.assertEqual(False,Interval([0,1]).emptyp())
        self.assertEqual(True,Interval().emptyp())

    def test_IntervalOverlap(self):
        self.assertEqual([None,None],Interval([0,1]).overlap(Interval([2,3])).list())
        self.assertEqual([0,0.5],Interval([0,1]).overlap(Interval([-0.5,0.5])).list())
        self.assertEqual([0,0],Interval([0,1]).overlap(Interval([-0.5,0])).list())
        self.assertEqual([1,1],Interval([0,1]).overlap(Interval([1,1.5])).list())
        self.assertEqual([0.25,0.75],Interval([0,1]).overlap(Interval([0.25,0.75])).list())
        self.assertEqual([0.25,0.75],Interval([0.25,0.75]).overlap(Interval([0,1])).list())
        self.assertEqual([0,1],Interval([0,1]).overlap(Interval([0,1])).list())
        self.assertEqual([0.5,0.5],Interval([0.5,0.5]).overlap(Interval([0,1])).list())

    def test_contains(self):
        self.assertEqual(True,Interval([0.0,1.0]).contains(0.5))
        self.assertEqual(False,Interval([0.0,1.0]).contains(1.5))
        self.assertEqual(True,Interval([175,180]).contains(180))
        self.assertEqual(True,Interval([-180,-175]).contains(-180))

    def test_LonInterval(self):
        self.assertEqual([0,1],LonInterval([0,1]).list())
        self.assertEqual([175,-175],LonInterval([175,-175]).list())
        self.assertEqual([-180,-175],LonInterval([175,-175]).to_Intervals()[0].list())
        self.assertEqual([175,180],LonInterval([175,-175]).to_Intervals()[1].list())
        self.assertEqual(180,LonInterval([175,-175]).centroid())
        self.assertEqual(True,LonInterval([175,-175]).contains(180))
        self.assertEqual(True,LonInterval([175,-175]).contains(-178))
        self.assertEqual(False,LonInterval([175,-175]).contains(-174))

    def test_LonIntervalOverlap(self):
        self.assertEqual(1,len(LonInterval([175.0,-175.0]).overlap(LonInterval([-177,-170]))))
        self.assertEqual([-177,-175.0],LonInterval([175.0,-175.0]).overlap(LonInterval([-177,-170]))[0].list())
        self.assertEqual(0,len(LonInterval([175.0,-175.0]).overlap(LonInterval([-173,-170]))))
        self.assertEqual(1,len(LonInterval([175.0,-175.0]).overlap(LonInterval([ 170, 176]))))
        self.assertEqual([175.0,176],LonInterval([175.0,-175.0]).overlap(LonInterval([ 170, 176]))[0].list())
        self.assertEqual(2,len(LonInterval([175.0,-175.0]).overlap(LonInterval([ 177,-178]))))
        self.assertEqual([-180,-178],LonInterval([175.0,-175.0]).overlap(LonInterval([ 177,-178]))[0].list())
        self.assertEqual([177,180],LonInterval([175.0,-175.0]).overlap(LonInterval([ 177,-178]))[1].list())
        self.assertEqual([-180,-175.0],LonInterval([175.0,-175.0]).overlap(LonInterval([ 170,-170]))[0].list())
        self.assertEqual([175.0,180],LonInterval([175.0,-175.0]).overlap(LonInterval([ 170,-170]))[1].list())
        self.assertEqual(2,len(LonInterval([175.0,-175.0]).overlap(LonInterval([-178, 178]))))
        self.assertEqual([-178,-175.0],LonInterval([175.0,-175.0]).overlap(LonInterval([-178, 178]))[0].list())
        self.assertEqual([175.0,178],LonInterval([175.0,-175.0]).overlap(LonInterval([-178, 178]))[1].list())
        
    def test_BoundingBox(self):
        # print '\n'+BoundingBox().to_xml()
        self.assertEqual('<BoundingBox>\n  <Point lon_degrees=None lat_degrees=None/>\n  '\
                         +'<Point lon_degrees=None lat_degrees=None/>\n</BoundingBox>\n'\
                         ,BoundingBox().to_xml())
        self.assertEqual(True,BoundingBox().emptyp())
        self.assertEqual(False,BoundingBox((Point((0,0)),Point((1,1)))).emptyp())
        self.assertEqual(([0,1],[0,1]),BoundingBox((Point((0,0)),Point((1,1)))).lons_lats())
        # print '\n'+BoundingBox((Point((0,0)),Point((1,1)))).to_xml()
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=0 lat_degrees=0/>\n'\
                         +'  <Point lon_degrees=1 lat_degrees=1/>\n'\
                         +'</BoundingBox>\n'\
                         ,BoundingBox((Point((0,0)),Point((1,1)))).to_xml())
        # print '\n'+BoundingBox((Point((0.5,0.5)),Point((1.5,1.5)))).to_xml()
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=0.5 lat_degrees=0.25/>\n'\
                         +'  <Point lon_degrees=1 lat_degrees=1/>\n'\
                         +'</BoundingBox>\n'\
                         ,BoundingBox((Point((0,0)),Point((1,1))))\
                         .overlap(BoundingBox((Point((0.5,0.25)),Point((1.5,1.5)))))\
                         [0].to_xml())
        self.assertEqual('<Point lon_degrees=0.5 lat_degrees=0.5/>'\
                         ,BoundingBox((Point((0,0)),Point((1,1)))).centroid().to_xml())

    def test_bbox_contains(self):
        self.assertEqual(True\
                         ,BoundingBox((Point((0,0)),Point((1,1))))\
                         .contains_point(Point((0.5,0.5))))
        self.assertEqual(False\
                         ,BoundingBox((Point((0,0)),Point((1,1))))\
                         .contains_point(Point((1.1,0.5))))
        self.assertEqual(True\
                         ,BoundingBox((Point((0,0)),Point((1,1))))\
                         .contains_point(Point((1.0,0.5))))

    def test_Covering(self):
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=0.0 lat_degrees=0.0/>\n'\
                         +'  <Point lon_degrees=1.0 lat_degrees=3.0/>\n'\
                         +'</BoundingBox>\n'\
                         ,box_covering([0.0,0.5,1.0],[0.0,1.0,1.5,3.0])\
                         .to_xml())
        bb0 = BoundingBox((Point((0.0,0.0)),Point((1.0,1.0))))
        bb1 = BoundingBox((Point((2.0,2.0)),Point((3.0,3.0))))
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=0.0 lat_degrees=0.0/>\n'\
                         +'  <Point lon_degrees=3.0 lat_degrees=3.0/>\n'\
                         +'</BoundingBox>\n'\
                         ,bb0.union(bb1).to_xml())
        bbz = BoundingBox()
        self.assertEqual(bb0.to_xml(),bb0.union(bbz).to_xml())
        self.assertEqual(bb0.to_xml(),bbz.union(bb0).to_xml())
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=-165.0 lat_degrees=40.0/>\n'\
                         +'  <Point lon_degrees=165.0 lat_degrees=50.0/>\n'\
                         +'</BoundingBox>\n'\
                         ,box_covering([-165.0,165],[40.0,50.0]).to_xml())
        
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=165.0 lat_degrees=40.0/>\n'\
                         +'  <Point lon_degrees=-175.0 lat_degrees=50.0/>\n'\
                         +'</BoundingBox>\n'\
                         ,box_covering([165.0,-175.0],[40.0,50.0]\
                         ,hack_branchcut_threshold=250).to_xml())

        # Note that the other_box below is not running hack_branchcut, because other_box is not presumed
        # compact.
        self.assertEqual(2,\
                         len(
                             box_covering([175.0,-175.0],[40.0,50.0],hack_branchcut_threshold=250)\
                             .overlap(box_covering([-177.0,177.0],[40.0,50.0]))))
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=-177.0 lat_degrees=40.0/>\n'\
                         +'  <Point lon_degrees=-175.0 lat_degrees=50.0/>\n'\
                         +'</BoundingBox>\n'\
                         ,box_covering([175.0,-175.0],[40.0,50.0],hack_branchcut_threshold=250)\
                         .overlap(box_covering([-177.0,177.0],[40.0,50.0]))[0].to_xml())
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=175.0 lat_degrees=40.0/>\n'\
                         +'  <Point lon_degrees=177.0 lat_degrees=50.0/>\n'\
                         +'</BoundingBox>\n'\
                         ,box_covering([175.0,-175.0],[40.0,50.0],hack_branchcut_threshold=250)\
                         .overlap(box_covering([-177.0,177.0],[40.0,50.0]))[1].to_xml())
        

    def test_bbox(self):
        SRC_DIRECTORY=data_src_directory()+'MODIS/'
        test_obj_0 = DataField(\
                                        datafilename='MYD05_L2.A2015304.2125.006.2015305175459.hdf'\
                                        ,datafieldname='Water_Vapor_Infrared'\
                                        ,srcdirname=SRC_DIRECTORY\
                                        )
        test_obj_1 = DataField(\
                                         datafilename='MOD05_L2.A2015304.1815.006.2015308155414.hdf'\
                                         ,datafieldname='Water_Vapor_Infrared'\
                                         ,srcdirname=SRC_DIRECTORY\
                                         )
        # print test_obj_0.bbox.to_xml()
        # print test_obj_1.bbox.to_xml()
        # print test_obj_0.bbox.overlap(test_obj_1.bbox).to_xml()
        # print test_obj_1.bbox.overlap(test_obj_0.bbox).to_xml()
        self.assertEqual( '<BoundingBox>\n'\
                          +'  <Point lon_degrees=-125.55471 lat_degrees=16.427057/>\n'\
                          +'  <Point lon_degrees=-108.90769 lat_degrees=34.175877/>\n'\
                          +'</BoundingBox>\n'\
                          ,test_obj_1.bbox.overlap(test_obj_0.bbox)[0].to_xml() )

        self.assertEqual( '<BoundingBox>\n'\
                          +'  <Point lon_degrees=-135.69511 lat_degrees=13.147603/>\n'\
                          +'  <Point lon_degrees=-98.01473 lat_degrees=37.473373/>\n'\
                          +'</BoundingBox>\n'\
                          ,test_obj_1.bbox.union(test_obj_0.bbox).to_xml() )
        
    def test_json(self):
        self.assertEqual('["BoundingBox", {"p0": [0.0, 0.0], "p1": [1.0, 1.0]}]'\
                         ,BoundingBox((Point((0,0)),Point((1,1)))).to_json())
        self.assertEqual('["BoundingBox", {"p0": [0.0, 0.1], "p1": [-1.7, 1.0]}]'\
                         ,BoundingBox((Point((0,0.1)),Point((-1.7,1)))).to_json())
        self.assertEqual('["BoundingBox", {"p0": [0.0, 0.0], "p1": [1.0, 1.0]}]'\
                         ,BoundingBox()\
                         .from_json(BoundingBox((Point((0,0)),Point((1,1)))).to_json()).to_json())

    def test_overlap_empty_bug(self):
        """BoundinbBox.from_json was not initializing the intervals."""
        a=BoundingBox((Point((0,0)),Point((1.0,1.0))))\
                         .overlap(BoundingBox((Point((0,0)),Point((1.0,1.0)))))
        # print 'a: ',a.to_xml()
        
        self.assertEqual(False,a[0].emptyp())

        self.assertEqual(False\
                         ,BoundingBox((Point((0,0)),Point((1.0,1.0))))\
                         .overlap(BoundingBox((Point((0,0)),Point((1.0,1.0)))))[0].emptyp())

        globeBox=BoundingBox((Point((-180.0,-90.0)),Point((180.0,90.0))))
        SRC_DIRECTORY=data_src_directory()+'MODIS/'
        test_obj_0 = DataField(\
                                        datafilename='MYD05_L2.A2015304.2125.006.2015305175459.hdf'\
                                        ,datafieldname='Water_Vapor_Infrared'\
                                        ,srcdirname=SRC_DIRECTORY\
                                        )
        b=globeBox.overlap(test_obj_0.bbox)
        self.assertEqual(False,b[0].emptyp())
        self.assertEqual(False,globeBox.overlap(test_obj_0.bbox)[0].emptyp())
        # print 't',test_obj_0.bbox.to_xml()
        # print 'g',globeBox.to_xml()
        # print 'b',b.to_xml()
        # The following is the key to the bug.
        obj_bb=BoundingBox().from_json(test_obj_0.bbox.to_json())
        self.assertEqual(False,obj_bb.emptyp())
        self.assertEqual(False,globeBox.overlap(obj_bb)[0].emptyp())
        
def demo_obj(show=False):
    if not show:
        return
    
    fig_gen = fig_generator(1,1)

    # if('NOGGIN_DATA_SRC_DIRECTORY' in os.environ):
    #     SRC_DIRECTORY_BASE=os.environ['NOGGIN_DATA_SRC_DIRECTORY']
    # else:
    #     SRC_DIRECTORY_BASE='./'
    # SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS/'
    SRC_DIRECTORY=data_src_directory()+'MODIS/'

    print('loading: ',SRC_DIRECTORY+'MYD05_L2.A2015304.2125.006.2015305175459.hdf\n')
    
    test_obj = DataField(\
                                        datafilename='MYD05_L2.A2015304.2125.006.2015305175459.hdf'\
                                        ,datafieldname='Water_Vapor_Infrared'\
                                        ,srcdirname=SRC_DIRECTORY\
                                        )
    test_obj.colormesh(vmin=1.0,vmax=3.0\
                                 ,colorbar=True\
                                 )

    fig_gen.increment_figure()
    # test_obj.colormesh()
    test_obj = DataField(\
                                         datafilename='MOD05_L2.A2015304.1815.006.2015308155414.hdf'\
                                         ,datafieldname='Water_Vapor_Infrared'\
                                         ,srcdirname=SRC_DIRECTORY\
                                         )
    test_obj.colormesh(vmin=1.0,vmax=3.0)

    # print test_obj.bbox.to_xml()

    fig_gen.increment_figure()
    test_obj_1 = DataField(\
                                           data       = test_obj.data\
                                           ,latitude  = test_obj.latitude\
                                           ,longitude = test_obj.longitude\
                                           )
    # test_obj_1.colormesh(vmin=1.0,vmax=3.0)                                           
    test_obj_1.scatterplot(vmin=1.0,vmax=3.0,title='scatter',colorbar=True)

    plt.show()


    
if __name__ == '__main__':

    demo_obj(False)

    if True:
        unittest.main()
        
