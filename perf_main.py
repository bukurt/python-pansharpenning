#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:48:00 2019

@author: burak
"""
import numpy as np
from osgeo import gdal
import time
from functions import load_datas_for_pansharpenning
from functions import pansharpenning
from functions import pansharpenning_mt
from functions import ps_quality_score
from functions import *


path_im_ms = './data/spot6/GEBZE/S6_GEBZE_MS.tiff'
path_im_pan = './data/spot6/GEBZE/S6_GEBZE_PAN.tiff'
path_xml = './data/spot6/GEBZE/S6_GEBZE_MS.XML'

pan, ms1, ms2, ms3 = load_datas_for_pansharpenning(path_im_pan, path_im_ms)
m, n = np.shape(pan)
#global vars
m, n = np.shape(pan)
f_pan1 = np.empty((m,n), dtype='float64')
f_pan2 = np.empty((m,n), dtype='float64')
f_pan3 = np.empty((m,n), dtype='float64')
f_ms1 = np.empty((m,n), dtype='complex128')
f_ms2 = np.empty((m,n), dtype='complex128')
f_ms3 = np.empty((m,n), dtype='complex128')
f_ps1 = np.empty((m,n), dtype='complex128')
f_ps2 = np.empty((m,n), dtype='complex128')
f_ps3 = np.empty((m,n), dtype='complex128')

time_scores = []
for i in range(10):
    start = time.time()
    # ps1, ps2, ps3 = pansharpenning('fft', pan, ms1, ms2, ms3, 'ideal_low', hist_m=True, cutoff_freq=.125)
    ps1, ps2, ps3 = pansharpenning_mt('fft', pan, ms1, ms2, ms3, 'ideal_low', hist_m=True, cutoff_freq=.125)
    time_scores.append(time.time() - start)
print "## ", str(np.mean(time_scores)), " seconds."
    

# Performance results
im_ps = np.empty((m,n,3), dtype='float64')
im_ref = np.empty((m,n,3), dtype='float64')
im_ps[:,:,0] = ps1; im_ps[:,:,1] = ps2; im_ps[:,:,2] = ps3
im_ref[:,:,0] = ms1; im_ref[:,:,1] = ms2; im_ref[:,:,2] = ms3
results = []
for p_method in ['SAM','RMSE','RASE','ERGAS']:
    result = ps_quality_score(p_method, im_ps, im_ref, xml_file=path_xml, ms_pan_ratio=0.25)
    results.append({p_method: result})
    print p_method, "\tscores:", str(result)



from osgeo import gdal
driver = gdal.GetDriverByName('Gtiff')
dataset = driver.Create('out.tiff', 1024, 1024, 3, gdal.GDT_Float32)
dataset.GetRasterBand(1).WriteArray(ps1)
dataset.GetRasterBand(2).WriteArray(ps2)
dataset.GetRasterBand(3).WriteArray(ps3)
dataset.FlushCache()


ds = gdal.Open('out.tiff', gdal.GA_ReadOnly)
a1 = ds.GetRasterBand(1).ReadAsArray()
a2 = ds.GetRasterBand(2).ReadAsArray()
a3 = ds.GetRasterBand(3).ReadAsArray()



fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(pan,cmap=plt.cm.gray)
ax2.imshow(pan,cmap=plt.cm.gray)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(ms1,cmap=plt.cm.Reds)
ax2.imshow(a1,cmap=plt.cm.Reds)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(ms2,cmap=plt.cm.Greens)
ax2.imshow(a2,cmap=plt.cm.Greens)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(ms3,cmap=plt.cm.Blues)
ax2.imshow(ps3.astype('float32'),cmap=plt.cm.Blues)


