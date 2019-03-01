# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import timeit
import multiprocessing
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
path_im_ms = '../data/spot6/GEBZE/S6_GEBZE_MS.tiff'
path_im_pan = '../data/spot6/GEBZE/S6_GEBZE_PAN.tiff'

i_ms = gdal.Open(path_im_ms)
i_pan = gdal.Open(path_im_pan)

np_ims1 = np.array(i_ms.GetRasterBand(1).ReadAsArray())
np_ims2 = np.array(i_ms.GetRasterBand(2).ReadAsArray())
np_ims3 = np.array(i_ms.GetRasterBand(3).ReadAsArray())
np_ipan = np.array(i_pan.GetRasterBand(1).ReadAsArray())


print("Number of cpu : ", multiprocessing.cpu_count())

