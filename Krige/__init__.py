
# import sys
# print(sys.path)

__author__  = 'ML Rilee'
__version__ = '0.0.1'
__doc__ = """
NOGGIn
======

NASA Open-Access Geo-Gridding Infrastructure (NOGGIn): An Integrated
Service for Next-Generation Modeling, Analysis, and Retrieval
Evaluation with Error Estimates

https://earthdata.nasa.gov/community/community-data-system-programs/access-projects/noggin

PI: T. Clune, NASA Goddard Space Flight Center

Summary
-------
Data processing supporting NOGGIn interpolation and uncertainty
estimation for data comparison goals.

Krige: Provide support functions for driving PyKrige OrdinaryKriging
(2D ordinary kriging) in the NOGGIn Earth Science context.

MODIS_DataField: Load and visualization functions for a MODIS datafield.

2018-0629-1420-40-EDT ML Rilee <mike@rilee.net> Rilee Systems Technologies LLC

"""

from . import core
# from Krige import span_array
# from Krige import drive_OKrige
# from Krige import krigePlotConfiguration
# from Krige import krigeHDF
from core import *

__all__ = ["core","DataField"]


