
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import json
# from types import SimpleNamespace

import h5py as h5

# For json decoding cf. https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object
#
def object_decoder(obj):
    "Decode an object via a hook when reading json."
    if '__type__' in obj and obj['__type__'] == 'User':
        return User(obj['name'], obj['username'])
    return obj

def load_and_plot_khdf(filename,vmin,vmax,rasterized):
    
    with h5.File(filename,mode='r') as hdf:
        l05_krg   = hdf['HDFEOS/NOGGIN/KrigeResult2/Data Fields/l05_krg']
        latitude  = hdf['HDFEOS/NOGGIN/KrigeResult2/Data Fields/latitude']
        longitude = hdf['HDFEOS/NOGGIN/KrigeResult2/Data Fields/longitude']
        config    = \
            hdf['HDFEOS/NOGGIN/KrigeResult2/KrigeCalculationConfiguration/configuration.json']
        config_ = json.loads(config.attrs['json'],object_hook=object_decoder)
        
        # plt.scatter(latitude,longitude,c=l05_krg,vmin=vmin,vmax=vmax)
        print('%s l05 mnmx: %e %e, converged: %s, diverged: %s'%(filename
                                                                 ,np.amin(l05_krg)
                                                                 ,np.amax(l05_krg)
                                                                 ,str(config_['variogram']['converged'])
                                                                 ,str(config_['variogram']['diverged']),)
                                                                 )
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
                  ,vmin= 1.0e-8
                  ,vmax= 3.0e-5
                  ,rasterized=False
                  )

    plt.show()
    return

if __name__ == '__main__':
    main()
