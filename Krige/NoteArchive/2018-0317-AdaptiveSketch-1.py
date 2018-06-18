#!/usr/bin/env python

# (find-file-other-frame "2018-0317-AdaptiveSketchNotes-1.org")

import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC
from math import *
from noggin import *

# from operator import add
from MODIS_DataField import MODIS_DataField

# for kriging, plotting
from scipy.spatial import ConvexHull

SRC_DIRECTORY=data_src_directory()

FILE_NAMES=[
'MYD05_L2.A2015304.1945.006.2015305180304.hdf',
'MYD05_L2.A2015304.2125.006.2015305175459.hdf'
]
MODIS_DIR=SRC_DIRECTORY+"MODIS/"

FILE_NAMES=[
'MYD05_L2.A2015304.2125.061.2018054031117.hdf',
'MYD05_L2.A2015304.1945.061.2018054025242.hdf'
]
MODIS_DIR=SRC_DIRECTORY+"MODIS-61/"

# COMPARISON_FILE='MOD05_L2.A2015304.1815.006.2015308155414.hdf'
COMPARISON_FILE='MOD05_L2.A2015304.1815.061.2017323092403.hdf'
COMPARISON_DIR=MODIS_DIR

f0=0; f1=len(FILE_NAMES)

FILE_NAMES_1=FILE_NAMES[f0:f1]
firstFlag=True

n=len(FILE_NAMES_1)

modis_objs = []
for i in range(f0,f1):
    print('i: ',i)
    modis_obj = MODIS_DataField(\
                                         datafilename=FILE_NAMES_1[i]\
                                         ,datafieldname='Water_Vapor_Infrared'\
                                         ,srcdirname=MODIS_DIR\
                                         )
    # modis_obj.colormesh(vmin=1.0,vmax=3.0)
    print('data.shape: ',modis_obj.data.shape)
    print('lat.shape:  ',modis_obj.latitude.shape)
    print('lon.shape:  ',modis_obj.longitude.shape)
    modis_objs.append(modis_obj)

print('length(modis_objs): ',len(modis_objs))
sizes_modis_objs = [ m.data.size for m in modis_objs ]
total_size_modis_objs = sum(sizes_modis_objs)

data      = np.zeros(total_size_modis_objs)
latitude  = np.zeros(total_size_modis_objs)
longitude = np.zeros(total_size_modis_objs)

i0=0
for i in range(len(sizes_modis_objs)):
    i1 = i0 + sizes_modis_objs[i]
    print('adding indexes: ', i0, i1)
    data[i0:i1],latitude[i0:i1],longitude[i0:i1] = modis_objs[i].ravel()
    i0=i1

idx = np.where(~np.isnan(data))
data1      = data[idx]
latitude1  = latitude[idx]
longitude1 = longitude[idx]
    
modis_obj_1 = MODIS_DataField(data=data1,latitude=latitude1,longitude=longitude1)
modis_obj_1.scatterplot(vmin=1.0,vmax=4.0,title='scatter')
# modis_obj_1.colormesh(vmin=1.0,vmax=3.0,title='scatter')

# The target grid -- rectangle
# grid = rectangular_grid(\
#                              x0 = -128.0\
#                             ,x1 =  -83.0\
#                             ,dx =    0.25\
#                             ,y0 =   20.0\
#                             ,y1 =   25.0\
#                             ,dy =    0.25\
#                             )

grid = rectangular_grid(\
                             x0 = -120.0\
                            ,x1 = -105.0\
                            ,dx =    0.25\
                            ,y0 =   20.0\
                            ,y1 =   25.0\
                            ,dy =    0.25\
                            )

# gridxy = np.meshgrid(np.arange(x0,x1,dx),np.arange(y0,y1,dy))
# # gridxy = np.meshgrid(np.arange(-120.0,-105.0,0.25),np.arange(  20.0,  25.0,0.25))
# gridx = np.ravel(gridxy[0])
# gridy = np.ravel(gridxy[1])
gridx,gridy = grid.gridxy()

# in_grid = np.where(      (x0 < longitude1) & (longitude1 < x1)\
#                        & (y0 < latitude1 ) & (latitude1  < y1)\
#                        )
# ex_grid = np.where(      (x0 > longitude1) | (longitude1 > x1)\
#                        | (y0 > latitude1 ) | (latitude1  > y1)\
#                        )

in_grid = grid.in_grid(longitude1,latitude1)
ex_grid = grid.ex_grid(longitude1,latitude1)

# Target results
gridz  = np.zeros(gridx.shape)
gridss = np.zeros(gridx.shape)

nlags=12
custom_args = None

# A gamma-rayleigh distribution
def custom_vg(params,dist):
    sill    = np.float(params[0])
    falloff = np.float(params[1])
    beta    = np.float(params[2])
    fd      = falloff*dist
    omfd    = 1.0-falloff*dist
    bfd2    = beta*omfd*omfd
    return \
      sill*fd*np.exp(omfd-bfd2)

dg=gridx.size

dx = span_array(gridx)
dy = span_array(gridy)
dr = sqrt(dx*dx+dy*dy)

# smaller beta => tighter sample distribution
#+ beta0=1.5*(dr)
# beta0=0.5*(dr)
beta0=0.75*(dr)
# beta0=1.0*(dr)
l=1.2*(dx/2)
w=1.2*(dy/2)

marker_size = 3.5
m_alpha = 1.0
colormap_0 = plt.cm.rainbow
colormap_1 = plt.cm.gist_yarg
colormap_2 = plt.cm.plasma
colormap_x = colormap_0
# vmin=1.0; vmax=5.0
vmin=np.nanmin(data1); vmax=np.nanmax(data1)

gridz, data_x, data_y, data_z = drive_OKrige(\
                         grid_stride=dg\
                         ,random_permute=True\
                         ,x=gridx,y=gridy\
                         ,src_x=longitude1\
                         ,src_y=latitude1\
                         ,src_z=data1\
                         ,variogram_model='custom'\
                         ,variogram_parameters=custom_args\
                         ,variogram_function=custom_vg\
                         ,enable_plotting=True
                         ,npts=1000
                         ,beta0=beta0
                         ,frac=0.0
                         ,l=l,w=w
                         )

xy1 = np.zeros((gridz.shape[0],2))
xy1[:,0] = gridx
xy1[:,1] = gridy
grid_hull = ConvexHull(xy1)

# fig = plt.gcf()

fig_gen = fig_generator(1,1)

modis_obj_2 = MODIS_DataField(data=data1[ex_grid],latitude=latitude1[ex_grid],longitude=longitude1[ex_grid])

modis_obj_2.scatterplot(title='scatter',plt_show = False\
                            ,vmin=vmin,vmax=vmax\
                            ,cmap=colormap_x\
                            )

if True:
    m = modis_obj_2.get_m()
    m.drawmapboundary(fill_color='dimgrey')
    m.scatter(gridx,gridy,c=gridz\
                  ,cmap=colormap_x\
                  ,linewidth=0\
                  ,alpha=m_alpha\
                  ,latlon=True\
                  ,vmin=vmin, vmax=vmax\
                  ,edgecolors=None\
                  ,s=marker_size*10\
                  ,marker='s'\
                  )
                  
if True:
    draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

if True:
    fig_gen.increment_figure()
    mod05_obj = MODIS_DataField(\
                                         datafilename=COMPARISON_FILE\
                                         ,srcdirname=COMPARISON_DIR\
                                         ,datafieldname='Water_Vapor_Infrared'\
                                         )
    mod05_obj.colormesh(vmin=vmin,vmax=vmax)

if True:
    plt.show()

#### Krige from MOD05

size_mod05 = mod05_obj.data.size

data2_       = np.zeros(size_mod05)
latitude2_   = np.zeros(size_mod05)
longitude2_  = np.zeros(size_mod05)

data2_,latitude2_,longitude2_ = mod05_obj.ravel()
idx2 = np.where(~np.isnan(data2_))

data2      = data2_[idx2]
latitude2  = latitude2_[idx2]
longitude2 = longitude2_[idx2]
print('data2.shape: ',data2.shape)

# Target results
gridz2  = np.zeros(gridx.shape)
gridss2 = np.zeros(gridx.shape)

gridz2, data_x2, data_y2, data_z2 = drive_OKrige(\
                         grid_stride=dg\
                         ,random_permute=True\
                         ,x=gridx,y=gridy\
                         ,src_x=longitude2\
                         ,src_y=latitude2\
                         ,src_z=data2\
                         ,variogram_model='custom'\
                         ,variogram_parameters=custom_args\
                         ,variogram_function=custom_vg\
                         ,enable_plotting=True
                         ,npts=1000
                         ,beta0=beta0
                         ,frac=0.0
                         ,l=l,w=w
                         )

in_grid2 = grid.in_grid(longitude2,latitude2)
ex_grid2 = grid.ex_grid(longitude2,latitude2)
mod05_obj_2 = MODIS_DataField(data=data2[ex_grid2],latitude=latitude2[ex_grid2],longitude=longitude2[ex_grid2])

mod05_obj_2.scatterplot(title='scatter',plt_show = False\
                            ,vmin=vmin,vmax=vmax\
                            ,cmap=colormap_x\
                            )

if True:
    m = mod05_obj_2.get_m()
    m.drawmapboundary(fill_color='dimgrey')
    m.scatter(gridx,gridy,c=gridz\
                  ,cmap=colormap_x\
                  ,linewidth=0\
                  ,alpha=m_alpha\
                  ,latlon=True\
                  ,vmin=vmin, vmax=vmax\
                  ,edgecolors=None\
                  ,s=marker_size*10\
                  ,marker='s'\
                  )

if True:
    marker_size_factor_s = 5
    
    wh_scale = [0.6,0.6]
    lat_center = np.nanmean(gridy)
    lon_center = np.nanmean(gridx)
    # fig_gen = fig_generator(1,2)
    # fig_gen = fig_generator(1,3)
    fig_gen = fig_generator(2,3)
    fig,ax = fig_gen.get_fig_axes()
    iax = fig_gen.iSubIndex()
    print('1 iax: ',iax)
    modis_obj_1.init_basemap(ax=ax[iax],wh_scale=wh_scale\
                                 ,lat_center=lat_center\
                                 ,lon_center=lon_center\
                                 )
    # ax[iax].set_title('{0}'.format('MYD'))
    ax[iax].set_title('Kriged\n{0}..\n..{1}'.format(     modis_objs[0].datafilename\
                                           ,modis_objs[1].datafilename))
    # modis_obj_1.scatterplot(title='scatter',plt_show = False\
    #                         ,vmin=vmin,vmax=vmax\
    #                         ,cmap=colormap_x\
    #                         )
    m = modis_obj_1.get_m()
    sc = m.scatter(gridx,gridy,c=gridz\
                  ,cmap=colormap_x\
                  ,linewidth=0\
                  ,alpha=m_alpha\
                  ,latlon=True\
                  ,vmin=vmin, vmax=vmax\
                  ,edgecolors=None\
                  ,s=marker_size*marker_size_factor_s\
                  ,marker='s'\
                  ,ax=ax[iax]\
                  )
    plt.gcf().colorbar(sc,ax=ax[iax])                  

    fig_gen.increment_figure()
    fig,ax = fig_gen.get_fig_axes()
    iax = fig_gen.iSubIndex()
    print('2 iax: ',iax)
    mod05_obj_2.init_basemap(ax=ax[iax],wh_scale=wh_scale\
                                 ,lat_center=lat_center\
                                 ,lon_center=lon_center\
                                 )
    # ax[iax].set_title('{0}'.format('MOD'))
    ax[iax].set_title('{0}'.format(mod05_obj.datafilename))
    # mod05_obj_2.scatterplot(title='scatter',plt_show = False\
    #                         ,vmin=vmin,vmax=vmax\
    #                         ,cmap=colormap_x\
    #                         )
    m = mod05_obj_2.get_m()
    sc = m.scatter(gridx,gridy,c=gridz2\
                  ,cmap=colormap_x\
                  ,linewidth=0\
                  ,alpha=m_alpha\
                  ,latlon=True\
                  ,vmin=vmin, vmax=vmax\
                  ,edgecolors=None\
                  ,s=marker_size*marker_size_factor_s\
                  ,marker='s'\
                  ,ax=ax[iax]\
                  )
    plt.gcf().colorbar(sc,ax=ax[iax])

                  
    fig_gen.increment_figure()
    fig,ax = fig_gen.get_fig_axes()
    iax = fig_gen.iSubIndex()
    print('3 iax: ',iax)
    mod05_obj_2.init_basemap(ax=ax[iax],wh_scale=wh_scale\
                                 ,lat_center=lat_center\
                                 ,lon_center=lon_center\
                                 )
    # ax[iax].set_title('{0}'.format('MOD'))
    ax[iax].set_title('MYD-MOD')
    # mod05_obj_2.scatterplot(title='scatter',plt_show = False\
    #                         ,vmin=vmin,vmax=vmax\
    #                         ,cmap=colormap_x\
    #                         )
    m = mod05_obj_2.get_m()
    delta = np.subtract(gridz,gridz2)
    delta_abs = np.absolute(delta)
    delta_max = np.nanmax(delta_abs)
    delta_min = -delta_max
    sc = m.scatter(gridx,gridy,c=delta\
                  ,vmin=delta_min,vmax=delta_max\
                  ,cmap=plt.cm.bwr\
                  ,linewidth=0\
                  ,alpha=m_alpha\
                  ,latlon=True\
                  ,edgecolors=None\
                  ,s=marker_size*marker_size_factor_s\
                  ,marker='s'\
                  ,ax=ax[iax]\
                  )
    plt.gcf().colorbar(sc,ax=ax[iax])

# MYD i.e. data1
if True:
    marker_size_factor_o = 2.5
    fig_gen.increment_figure()
    fig,ax = fig_gen.get_fig_axes()
    iax = fig_gen.iSubIndex()
    print('4 iax: ',iax)
    mod05_obj_2.init_basemap(ax=ax[iax],wh_scale=wh_scale\
                                 ,lat_center=lat_center\
                                 ,lon_center=lon_center\
                                 )
    ax[iax].set_title('{0}'.format('MYD data1'))
    # ax[iax].set_title('{0}'.format(mod05_obj.datafilename))
    # mod05_obj_2.scatterplot(title='scatter',plt_show = False\
    #                         ,vmin=vmin,vmax=vmax\
    #                         ,cmap=colormap_x\
    #                         )
    m = mod05_obj_2.get_m()
    sc = m.scatter(\
                       longitude1,latitude1,c=data1\
                       ,cmap=colormap_x\
                       ,linewidth=0\
                       ,alpha=m_alpha\
                       ,latlon=True\
                       ,vmin=vmin, vmax=vmax\
                       ,edgecolors=None\
                       ,s=marker_size*marker_size_factor_o\
                       ,marker='o'\
                       ,ax=ax[iax]\
                       )
    plt.gcf().colorbar(sc,ax=ax[iax])

# MOD i.e. data2
if True:
    fig_gen.increment_figure()
    fig,ax = fig_gen.get_fig_axes()
    iax = fig_gen.iSubIndex()
    print('4 iax: ',iax)
    mod05_obj_2.init_basemap(ax=ax[iax],wh_scale=wh_scale\
                                 ,lat_center=lat_center\
                                 ,lon_center=lon_center\
                                 )
    ax[iax].set_title('{0}'.format('MOD data2'))
    # ax[iax].set_title('{0}'.format(mod05_obj.datafilename))
    # mod05_obj_2.scatterplot(title='scatter',plt_show = False\
    #                         ,vmin=vmin,vmax=vmax\
    #                         ,cmap=colormap_x\
    #                         )
    m = mod05_obj_2.get_m()
    sc = m.scatter(\
                       longitude2,latitude2,c=data2\
                       ,cmap=colormap_x\
                       ,linewidth=0\
                       ,alpha=m_alpha\
                       ,latlon=True\
                       ,vmin=vmin, vmax=vmax\
                       ,edgecolors=None\
                       ,s=marker_size*marker_size_factor_o\
                       ,marker='o'\
                       ,ax=ax[iax]\
                       )
    plt.gcf().colorbar(sc,ax=ax[iax])

# mod - data2+Krige
if True:
    fig_gen.increment_figure()
    fig,ax = fig_gen.get_fig_axes()
    iax = fig_gen.iSubIndex()
    print('4 iax: ',iax)
    mod05_obj_2.init_basemap(ax=ax[iax],wh_scale=wh_scale\
                                 ,lat_center=lat_center\
                                 ,lon_center=lon_center\
                                 )
    ax[iax].set_title('{0}'.format('MOD+Krige'))
    # ax[iax].set_title('{0}'.format(mod05_obj.datafilename))
    # mod05_obj_2.scatterplot(title='scatter',plt_show = False\
    #                         ,vmin=vmin,vmax=vmax\
    #                         ,cmap=colormap_x\
    #                         )
    m = mod05_obj_2.get_m()
    sc = m.scatter(\
                       longitude2[ex_grid2],latitude2[ex_grid2],c=data2[ex_grid2]\
                       ,cmap=colormap_x\
                       ,linewidth=0\
                       ,alpha=m_alpha\
                       ,latlon=True\
                       ,vmin=vmin, vmax=vmax\
                       ,edgecolors=None\
                       ,s=marker_size*marker_size_factor_o\
                       ,marker='o'\
                       ,ax=ax[iax]\
                       )
    sc = m.scatter(gridx,gridy,c=gridz2\
                  ,cmap=colormap_x\
                  ,linewidth=0\
                  ,alpha=m_alpha\
                  ,latlon=True\
                  ,vmin=vmin, vmax=vmax\
                  ,edgecolors=None\
                  ,s=marker_size*marker_size_factor_s\
                  ,marker='s'\
                  ,ax=ax[iax]\
                  )
    plt.gcf().colorbar(sc,ax=ax[iax])
    
if True:
    plt.show()


    
