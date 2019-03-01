# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from multiprocessing import Pool
from multiprocessing import cpu_count
import threading
from threading import Thread
import pyfftw
import numpy as np
from osgeo import gdal
import cv2
import time
# params

INTERPOLATION = cv2.INTER_LINEAR
gdal.UseExceptions()
path_im_ms = '../data/spot6/GEBZE/S6_GEBZE_MS.tiff'
path_im_pan = '../data/spot6/GEBZE/S6_GEBZE_PAN.tiff'

start = time.time()
ds_ms = gdal.Open(path_im_ms)
ds_pan = gdal.Open(path_im_pan)
np_pan_orj = np.array(ds_pan.GetRasterBand(1).ReadAsArray())
np_ms1_orj = np.array(ds_ms.GetRasterBand(1).ReadAsArray())
np_ms2_orj = np.array(ds_ms.GetRasterBand(2).ReadAsArray())
np_ms3_orj = np.array(ds_ms.GetRasterBand(3).ReadAsArray())
np_pan = cv2.resize(np_pan_orj, dsize=(ds_pan.RasterXSize/4,
                                       ds_pan.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
np_ms1 = cv2.resize(np_ms1_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
np_ms2 = cv2.resize(np_ms2_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
np_ms3 = cv2.resize(np_ms3_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
del np_pan_orj, np_ms1_orj, np_ms2_orj, np_ms3_orj
end1 = time.time() - start
print "## ", str(end1), " seconds for load data"

start = time.time()
pyfftw.config.NUM_THREADS = 1
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
# FFTW for PAN
fft_pan_a = pyfftw.empty_aligned(np.shape(np_pan), dtype='complex64')
fft_pan_b = pyfftw.empty_aligned(np.shape(np_pan), dtype='complex64')
fft_pan_c = pyfftw.empty_aligned(np.shape(np_pan), dtype='complex64')
# FFTW for MS
fft_ms_a = pyfftw.empty_aligned(np.shape(np_ms1), dtype='complex64')
fft_ms_b = pyfftw.empty_aligned(np.shape(np_ms1), dtype='complex64')
fft_ms_c = pyfftw.empty_aligned(np.shape(np_ms1), dtype='complex64')
# Plan an fft over the last axis
fft_obj_pan_forwd = pyfftw.FFTW(fft_pan_a, fft_pan_b, direction='FFTW_FORWARD')
fft_obj_pan_bckwd = pyfftw.FFTW(fft_pan_b, fft_pan_c, direction='FFTW_BACKWARD')
fft_obj_ms_forwd = pyfftw.FFTW(fft_ms_a, fft_ms_b, direction='FFTW_FORWARD')
fft_obj_ms_bckwd = pyfftw.FFTW(fft_ms_b, fft_ms_c, direction='FFTW_BACKWARD')
end2 = time.time() - start
print "## ", str(end2), " seconds for plan fftw"

pool_data = ([np_ms1, 'ms_forwd'], [np_ms2, 'ms_forwd'])

def fftw_pan_forwd(np_in):
    np_out = fft_obj_pan_forwd(np_in)
    # return np_out
def fftw_pan_bckwd(np_in):
    np_out = fft_obj_pan_bckwd(np_in)
    # return np_out
def fftw_ms_forwd(np_in):
    np_out = fft_obj_ms_forwd(np_in)
    # return np_out
def fftw_ms_bcjwd(np_in):
    np_out = fft_obj_ms_bckwd(np_in)
    # return np_out

def npfft2(np_in):
    np_out = np.fft.fft2(np_in)
    #print("FINISH %s " % threading.current_thread())

def phandler_pan_forwd():
    start = time.time()
    p = Pool(1)
    print "## ", str(time.time() - start), " seconds for initialize pool"
    #p.map(do_fftw, ((i1,i2) for i1, i2 in pool_data))
    start = time.time()
    p.map(fft_obj_ms_forwd, (np_ms1))
    print "## ", str(time.time() - start), " seconds for fftw"


if __name__ == '__main__':
    pool_handler()

p = Pool(1)
start = time.time()
p.map(npfft2, np_pan_orj)
print "## ", str(time.time() - start), " seconds for fftw"
p.close
start = time.time()
fft_obj_pan_forwd(np_pan)
print "## ", str(time.time() - start), " seconds for fftw"

start = time.time()
npfft2(np_pan_orj)
npfft2(np_pan_orj)
print "## ", str(time.time() - start), " seconds for fftw"



#Creating only four threads for now
start = time.time()
for i in (np_pan_orj, np_pan_orj):
    th = Thread(target=npfft2, args=(i, ))
    th.start()
print "## ", str(time.time() - start), " seconds for fftw"


