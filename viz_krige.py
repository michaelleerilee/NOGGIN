
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import h5py as h5    


def load_and_plot_khdf(filename,vmin,vmax,rasterized):
    
    with h5.File(filename,mode='r') as hdf:
        l05_krg   = hdf['HDFEOS/NOGGIN/KrigeResult2/Data Fields/l05_krg']
        latitude  = hdf['HDFEOS/NOGGIN/KrigeResult2/Data Fields/latitude']
        longitude = hdf['HDFEOS/NOGGIN/KrigeResult2/Data Fields/longitude']
        # plt.scatter(latitude,longitude,c=l05_krg,vmin=vmin,vmax=vmax)
        plt.contourf(longitude
                     ,latitude
                     ,l05_krg
                     ,vmin=vmin
                     ,vmax=vmax
                     ,transform=ccrs.PlateCarree()
                     ,rasterized=rasterized
                     )
    return

def main():
    print('hello world')

    ax = plt.axes(projection=ccrs.PlateCarree(),transform=ccrs.Geodetic())
    ax.coastlines()
    ax.set_global()

    # files = ['noggin_krige_0019_-026_0020_-025.h5']
    files = glob.glob('noggin_krige*.h5')

    for f in files:
        load_and_plot_khdf(f
                  ,vmin= 1.0e-7
                  ,vmax= 1.0e-6
                  ,rasterized=False
                  )

    plt.show()
    return

if __name__ == '__main__':
    main()
