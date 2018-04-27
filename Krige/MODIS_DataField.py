#!/usr/bin/env python
"""
Prepare a MODIS datafield for further processing using python tools. Provide a basic viewing capability.

2018-0423-1352-44-EDT ML Rilee, RSTLLC, mike@rilee.net.
"""

import os
import numpy as np
from pyhdf.SD import SD, SDC
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from noggin import fig_generator, data_src_directory

import unittest

class Interval():
    def __init__(self):
        """An interval."""
        self.x0 = None
        self.x1 = None
    def __init__(self,lst=None):
        if lst is None:
            self.x0 = None
            self.x1 = None
        else:            
            self.x0 = min(lst)
            self.x1 = max(lst)
    def tuple(self):
        return (self.x0,self.x1)
    def list(self):
        return [self.x0,self.x1]
    def emptyp(self):
        return (self.x0 is None) and (self.x1 is None)
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

class Point():
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
    def str(self):
        return "<Point lon_degrees="+str(self.lon_degrees)+" lat_degrees="+str(self.lat_degrees)+"/>"


def box_covering(lon,lat):
    """Make a BB covering, arguments (lon,lat) are arrays of lat and lon in degrees."""
    return BoundingBox(( Point((np.nanmin(lon),np.nanmin(lat)))\
                         ,Point((np.nanmax(lon),np.nanmax(lat))) ))
    
class BoundingBox():
    p0 = None; p1 = None
    def __init__(self):
        """A box."""
    def __init__(self,p=(Point(),Point())):
        self.p0 = p[0]
        self.p1 = p[1]
        self.lon_interval = Interval([self.p0.lon_degrees, self.p1.lon_degrees])
        self.lat_interval = Interval([self.p0.lat_degrees, self.p1.lat_degrees])
    def emptyp(self):
        return (self.lon_interval.emptyp() or self.lat_interval.emptyp())
    def str(self):
        return \
            "<BoundingBox>\n"\
            "  "+self.p0.str()+"\n"\
            "  "+self.p1.str()+"\n"\
            +"</BoundingBox>\n"
    
    def overlap(self,other_box):
        if self.emptyp() or other_box.emptyp():
            return BoundingBox()
        
        lon_overlap = self.lon_interval.overlap(other_box.lon_interval)
        if lon_overlap.emptyp():
            return BoundingBox()
        lat_overlap = self.lat_interval.overlap(other_box.lat_interval)
        if lat_overlap.emptyp():
            return BoundingBox()
        
        o_lon_interval = other_box.lon_interval
        if o_lon_interval.emptyp():
            return BoundingBox()
        o_lat_interval = other_box.lat_interval
        if o_lat_interval.emptyp():
            return BoundingBox()

        return BoundingBox(( Point((lon_overlap.x0,lat_overlap.x0))\
                           ,Point((lon_overlap.x1,lat_overlap.x1)) ))
        
class MODIS_DataField():
    """Access a datafield in a MODIS file"""
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
                     ,colormesh_title='colormesh_title'\
                     ,long_name='long_name'\
                     ):
        self.datafilename  = datafilename
        self.datafieldname = datafieldname
        self.data          = data
        self.latitude      = latitude
        self.longitude     = longitude
        self.srcdirname    = srcdirname
        self.geofile       = geofile
        # Should test that either datafilename or data is set.
        if self.datafilename is not None:
            self.load()
        else:
            self.plot_lat_m_center = np.nanmean(self.latitude)
            self.plot_lon_m_center = np.nanmean(self.longitude)


    def load(self):
        hdf = SD(self.srcdirname+self.datafilename, SDC.READ)
        ds  = hdf.select(self.datafieldname)
        # ds_dims = [ds.dimensions()[i] for i in ds.dimensions().keys()]

        self.colormesh_title = self.datafilename
        
        nAlong  = ds.dimensions()[self.key_along]
        nAcross = ds.dimensions()[self.key_across]

        data = np.zeros((nAlong,nAcross))
        data = ds[0:nAlong,0:nAcross].astype(np.double)
        
        attrs        = ds.attributes(full=1)
        lna          = attrs["long_name"]
        self.long_name    = lna[0]
        aoa          = attrs["add_offset"]
        add_offset   = aoa[0]
        fva          = attrs["add_offset"]
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
            lat = hdf.select('Latitude')
            self.latitude = lat[:,:]
            lon = hdf.select('Longitude')
            self.longitude = lon[:,:]
            self.bbox = box_covering(self.longitude,self.latitude)
        
    def init_basemap(self,ax=None,wh_scale=(1.5,1.5)\
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
        
        self.m = Basemap(projection='laea', resolution='l', lat_ts=65\
                        ,ax=ax\
                        ,lat_0=self.plot_lat_m_center, lon_0=self.plot_lon_m_center\
                        ,width=wh_scale[0]*3000000,height=wh_scale[1]*2500000)
        self.m.drawcoastlines(linewidth=0.5)
        self.m.drawparallels(np.arange(50., 91., 10.), labels=[1, 0, 0, 0])
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
        basename = os.path.basename(title_),
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
                        ,colorbar=False,title=None\
                        ,cmap=None\
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
        if cmap is None:
            sc = self.m.scatter(self.longitude, self.latitude, c=self.data, latlon=True\
                               ,vmin=vmin,vmax=vmax\
                               ,s=marker_size\
                               )
        else:
            sc = self.m.scatter(self.longitude, self.latitude, c=self.data, latlon=True\
                               ,vmin=vmin,vmax=vmax\
                               ,s=marker_size\
                               ,cmap=cmap
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



class TestPoint(unittest.TestCase):

    def test_Point(self):
        self.assertEqual((None,None),Point().inDegrees())
        self.assertEqual((1,0),Point(latlon_degrees=(0,1)).inDegrees())
        self.assertEqual((2,3),Point((2,3)).inDegrees())
        self.assertEqual('<Point lon_degrees=None lat_degrees=None/>',Point().str())
        self.assertEqual('<Point lon_degrees=0 lat_degrees=1/>',Point((0,1)).str())

    def test_Interval(self):
        self.assertEqual((None,None),Interval().tuple())
        self.assertEqual((0,1),Interval([0,1]).tuple())
        self.assertEqual([1,2],Interval(lst=[1,2]).list())
        self.assertEqual([1,2],Interval([2,1]).list())

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
        
    def test_BoundingBox(self):
        # print '\n'+BoundingBox().str()
        self.assertEqual('<BoundingBox>\n  <Point lon_degrees=None lat_degrees=None/>\n  '\
                         +'<Point lon_degrees=None lat_degrees=None/>\n</BoundingBox>\n'\
                         ,BoundingBox().str())
        self.assertEqual(True,BoundingBox().emptyp())
        self.assertEqual(False,BoundingBox((Point((0,0)),Point((1,1)))).emptyp())
        # print '\n'+BoundingBox((Point((0,0)),Point((1,1)))).str()
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=0 lat_degrees=0/>\n'\
                         +'  <Point lon_degrees=1 lat_degrees=1/>\n'\
                         +'</BoundingBox>\n'\
                         ,BoundingBox((Point((0,0)),Point((1,1)))).str())
        # print '\n'+BoundingBox((Point((0.5,0.5)),Point((1.5,1.5)))).str()
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=0.5 lat_degrees=0.25/>\n'\
                         +'  <Point lon_degrees=1 lat_degrees=1/>\n'\
                         +'</BoundingBox>\n'\
                         ,BoundingBox((Point((0,0)),Point((1,1))))\
                         .overlap(BoundingBox((Point((0.5,0.25)),Point((1.5,1.5)))))\
                         .str())

    def test_Covering(self):
        self.assertEqual('<BoundingBox>\n'\
                         +'  <Point lon_degrees=0.0 lat_degrees=0.0/>\n'\
                         +'  <Point lon_degrees=1.0 lat_degrees=3.0/>\n'\
                         +'</BoundingBox>\n'\
                         ,box_covering([0.0,0.5,1.0],[0.0,1.0,1.5,3.0])\
                         .str())

    def test_MODIS_bbox(self):
        SRC_DIRECTORY=data_src_directory()+'MODIS/'
        test_modis_obj_0 = MODIS_DataField(\
                                        datafilename='MYD05_L2.A2015304.2125.006.2015305175459.hdf'\
                                        ,datafieldname='Water_Vapor_Infrared'\
                                        ,srcdirname=SRC_DIRECTORY\
                                        )
        test_modis_obj_1 = MODIS_DataField(\
                                         datafilename='MOD05_L2.A2015304.1815.006.2015308155414.hdf'\
                                         ,datafieldname='Water_Vapor_Infrared'\
                                         ,srcdirname=SRC_DIRECTORY\
                                         )
        # print test_modis_obj_0.bbox.str()
        # print test_modis_obj_1.bbox.str()
        # print test_modis_obj_0.bbox.overlap(test_modis_obj_1.bbox).str()
        # print test_modis_obj_1.bbox.overlap(test_modis_obj_0.bbox).str()
        self.assertEqual( '<BoundingBox>\n'\
                          +'  <Point lon_degrees=-125.555 lat_degrees=16.4271/>\n'\
                          +'  <Point lon_degrees=-108.908 lat_degrees=34.1759/>\n'\
                          +'</BoundingBox>\n'\
                          ,test_modis_obj_1.bbox.overlap(test_modis_obj_0.bbox).str() )

        
def demo_modis_obj(show=False):
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
    
    test_modis_obj = MODIS_DataField(\
                                        datafilename='MYD05_L2.A2015304.2125.006.2015305175459.hdf'\
                                        ,datafieldname='Water_Vapor_Infrared'\
                                        ,srcdirname=SRC_DIRECTORY\
                                        )
    test_modis_obj.colormesh(vmin=1.0,vmax=3.0\
                                 ,colorbar=True\
                                 )

    fig_gen.increment_figure()
    # test_modis_obj.colormesh()
    test_modis_obj = MODIS_DataField(\
                                         datafilename='MOD05_L2.A2015304.1815.006.2015308155414.hdf'\
                                         ,datafieldname='Water_Vapor_Infrared'\
                                         ,srcdirname=SRC_DIRECTORY\
                                         )
    test_modis_obj.colormesh(vmin=1.0,vmax=3.0)

    # print test_modis_obj.bbox.str()

    fig_gen.increment_figure()
    test_modis_obj_1 = MODIS_DataField(\
                                           data       = test_modis_obj.data\
                                           ,latitude  = test_modis_obj.latitude\
                                           ,longitude = test_modis_obj.longitude\
                                           )
    # test_modis_obj_1.colormesh(vmin=1.0,vmax=3.0)                                           
    test_modis_obj_1.scatterplot(vmin=1.0,vmax=3.0,title='scatter',colorbar=True)

    plt.show()
    
if __name__ == '__main__':

    demo_modis_obj(False)

    if True:
        unittest.main()
        
