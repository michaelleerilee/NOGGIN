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

if __name__ == '__main__':
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

    fig_gen.increment_figure()
    test_modis_obj_1 = MODIS_DataField(\
                                           data       = test_modis_obj.data\
                                           ,latitude  = test_modis_obj.latitude\
                                           ,longitude = test_modis_obj.longitude\
                                           )
    # test_modis_obj_1.colormesh(vmin=1.0,vmax=3.0)                                           
    test_modis_obj_1.scatterplot(vmin=1.0,vmax=3.0,title='scatter',colorbar=True)

    plt.show()
