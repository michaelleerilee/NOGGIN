# -*- coding: utf-8 -*-
# import sys
# print(sys.path)

__author__  = 'ML Rilee'
__version__ = '0.0.1'
__doc__ = """

SUMMARY
-------
Data processing supporting NOGGIn interpolation and uncertainty
estimation for data comparison goals.

Krige: Provide support functions for driving PyKrige OrdinaryKriging
(2D ordinary kriging) in the NOGGIn Earth Science context.

MODIS_DataField: Load and visualization functions for a MODIS datafield.

------

NOGGIn
======

NASA Open-Access Geo-Gridding Infrastructure (NOGGIn): An Integrated
Service for Next-Generation Modeling, Analysis, and Retrieval
Evaluation with Error Estimates

https://earthdata.nasa.gov/community/community-data-system-programs/access-projects/noggin

PI: T. Clune, NASA Goddard Space Flight Center

------

2018-0629-1420-40-EDT ML Rilee <mike@rilee.net> Rilee Systems Technologies LLC

Copyright © 2018 Michael Lee Rilee, mike@rilee.net, Rilee Systems Technologies LLC

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

"""

from . import core
# from Krige import span_array
# from Krige import drive_OKrige
# from Krige import krigePlotConfiguration
# from Krige import krigeHDF
from core import *

__all__ = ["core","DataField"]


