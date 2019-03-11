# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def load_test_data(dset='pan'):
    import numpy as np
    from osgeo import gdal
    # import time
    # start = time.time()
    try:
        if dset == 'pan':
            path_im = './data/spot6/GEBZE/S6_GEBZE_PAN.tiff'
        elif dset == 'ms':
            path_im = './data/spot6/GEBZE/S6_GEBZE_MS.tiff'
        else:
            return 1
        ds = gdal.Open(path_im, gdal.GA_ReadOnly)
        np_im = np.empty((ds.RasterXSize, ds.RasterYSize, ds.RasterCount), dtype='uint16')
        for i in range(ds.RasterCount):
            np_im[:, :, i] = ds.GetRasterBand(i+1).ReadAsArray()
        np_im = np.concatenate([np_im, np_im], axis=0)
        np_im = np.concatenate([np_im, np_im], axis=1)
        return np_im
    except Exception as e:
        print e.message, e.args

def load_datas_for_pansharpenning(pan_path, ms_path):
    from osgeo import gdal
    import cv2
    INTERPOLATION = cv2.INTER_LINEAR
    SCALE = 4
    gdal.UseExceptions()
    try:
        ds_pan = gdal.Open(pan_path, gdal.GA_ReadOnly)
        ds_ms = gdal.Open(ms_path, gdal.GA_ReadOnly)        
        pan = ds_pan.GetRasterBand(1).ReadAsArray()
        ms1 = ds_ms.GetRasterBand(1).ReadAsArray().astype('float64')
        ms2 = ds_ms.GetRasterBand(2).ReadAsArray().astype('float64')
        ms3 = ds_ms.GetRasterBand(3).ReadAsArray().astype('float64')
        pan = pan[0:-1:SCALE, 0:-1:SCALE]
        ms1 = ms1[0:-1:SCALE, 0:-1:SCALE]
        ms2 = ms2[0:-1:SCALE, 0:-1:SCALE]
        ms3 = ms3[0:-1:SCALE, 0:-1:SCALE]
        ms1 = cv2.resize(ms1, (0,0), fx=SCALE, fy=SCALE, interpolation=INTERPOLATION)
        ms2 = cv2.resize(ms2, (0,0), fx=SCALE, fy=SCALE, interpolation=INTERPOLATION)
        ms3 = cv2.resize(ms3, (0,0), fx=SCALE, fy=SCALE, interpolation=INTERPOLATION)
        return pan, ms1, ms2, ms3
    except Exception as e:
        print e.message, e.args

def dftuv(m, n):
    import numpy as np
    u = np.arange(m)
    v = np.arange(n)
    np.putmask(u, u > (m/2 -1 ), u - m +1)
    np.putmask(v, v > (n/2 -1 ), v - n +1)
    uu, vv = np.meshgrid(u, v, sparse=False)
    return uu, vv

def ffilters(filter_name, m, n, d0=.125, k=1):
    import numpy as np
    u, v = dftuv(m, n)
    d0 = d0 * (max(m,n)/2)
    d = np.sqrt(u**2 + v**2)
    if filter_name == 'ideal_low':
        h = (d <= d0).astype('float64')
    elif filter_name == 'ideal_high':
        h = (d >= d0).astype('float64')
    elif filter_name == 'hamming':
        h = (.54 + .46 * np.cos(np.pi * (d/d0) )) * (d <= d0)
    elif filter_name == 'hanning':
        h = .5 * (1 + np.cos(np.pi * d/d0) ) * (d <= d0)
    elif filter_name == 'lbtw':
        h = 1 / (1 + (d / d0) ** (2 * k))
    elif filter_name == 'gauss_low':
        h = np.exp(-(d**2) / (2*(d0**2)))
    else:
        print "Unknown filter name."
        exit(1)
    return h

def hist_match(im, im_ref):
    import numpy as np
    m, n = np.shape(im)
    m_ref, n_ref = np.shape(im_ref)
    im_mean = np.mean(im)
    im_ref_mean = np.mean(im_ref)
    im_std = np.std(im)
    im_ref_std = np.std(im_ref)
    im = (im - im_mean) * (im_ref_std / im_std) + im_ref_mean;
    return im

def mean_rad(xml_file):
    import xml.etree.ElementTree as et
    root = et.parse(xml_file).getroot()
    g1 = float(root[8][4][0][0][4][5].text)
    g2 = float(root[8][4][0][0][5][5].text)
    g3 = float(root[8][4][0][0][6][5].text)
    return g1, g2, g3

def ps_quality_score(p_method, im_ps, im_ref, xml_file='none', ms_pan_ratio=0.25):
    import numpy as np
    m, n, k = np.shape(im_ps)
    if p_method == 'SAM':
        p1 = np.empty((k), dtype='float64')
        p2 = np.empty((k), dtype='float64')
        for i in range(k):
            p1[i] = np.dot(np.reshape(im_ps[:,:,i], (m*n)), np.reshape(im_ref[:,:,i], (m*n)))
            p2[i] = np.sqrt(np.sum(im_ps[:,:,i] ** 2.0)) * np.sqrt(np.sum(im_ref[:,:,i] ** 2.0))
        result = (np.arccos(p1 / p2) * (180.0 / np.pi))
    elif p_method == 'RMSE':
        p1 = np.empty((k), dtype='float64')
        p2 = np.empty((k), dtype='float64')
        for i in range(k):
            p1[i] = np.sum((im_ref[:,:,i] - im_ps[:,:,i]) ** 2.0)
        result = ((1.0 / (m*n)) * (np.sqrt(p1)))
    elif p_method == 'RASE':
        p1 = np.empty((k), dtype='float64')
        p2 = np.empty((k), dtype='float64')
        rmse = np.empty((k), dtype='float64')
        for i in range(k):
            p1[i] = np.sum((im_ref[:,:,i] - im_ps[:,:,i]) ** 2.0)
            rmse[i] = (1.0 / (m*n)) * (np.sqrt(p1[i]))
        gain = np.array(mean_rad(xml_file))
        p2 = np.sum((rmse ** 2.0) / gain)
        result = (100.0 * np.sqrt((1.0 / k) * p2))
    elif p_method == 'ERGAS':
        p1 = np.empty((k), dtype='float64')
        p2 = np.empty((k), dtype='float64')
        rmse = np.empty((k), dtype='float64')
        for i in range(k):
            p1[i] = np.sum((im_ref[:,:,i] - im_ps[:,:,i]) ** 2.0)
            rmse[i] = (1.0 / (m*n)) * (np.sqrt(p1[i]))
        gain = np.array(mean_rad(xml_file))
        p2 = np.sum((rmse ** 2.0) / gain)
        result = (100.0 * ms_pan_ratio * np.sqrt((1.0 / k) * p2))
    return result

def pansharpenning(ps_method, pan, ms1, ms2, ms3, filter_name, cutoff_freq=.125, hist_m=True):
    import numpy as np
    if ps_method == 'fft':
        m, n = np.shape(pan)
        h_low = ffilters(filter_name, m, n, cutoff_freq, 1)
        h_high = 1 - h_low
        if hist_m:
            f_pan1 = np.fft.fft2(hist_match(pan, ms1))
            f_pan2 = np.fft.fft2(hist_match(pan, ms2))
            f_pan3 = np.fft.fft2(hist_match(pan, ms3))
            g_pan1 = f_pan1 * h_high
            g_pan2 = f_pan2 * h_high
            g_pan3 = f_pan3 * h_high
        else:
            f_pan = np.fft.fft2(pan)
            g_pan = f_pan * h_high
        f_ms1 = np.fft.fft2(ms1)
        f_ms2 = np.fft.fft2(ms2)
        f_ms3 = np.fft.fft2(ms3)
        g_ms1 = f_ms1 * h_low
        g_ms2 = f_ms2 * h_low
        g_ms3 = f_ms3 * h_low
        if hist_m:
            f_ps1 = g_pan1 + g_ms1
            f_ps2 = g_pan2 + g_ms2
            f_ps3 = g_pan3 + g_ms3
        else:
            f_ps1 = g_pan + g_ms1
            f_ps2 = g_pan + g_ms2
            f_ps3 = g_pan + g_ms3
        ps1 = np.fft.ifft2(f_ps1)
        ps2 = np.fft.ifft2(f_ps2)
        ps3 = np.fft.ifft2(f_ps3)
    elif ps_method == '':
        pass
    return ps1, ps2, ps3

############################################################################### 
# Multi-theaded functions
###############################################################################
    
def hist_match_mt(band):
    import numpy as np
    global pan
    global ms1
    global ms2
    global ms3
    global f_pan1
    global f_pan2
    global f_pan3
    m, n = np.shape(pan)
    im_mean = np.mean(pan)
    im_std = np.std(pan)
    if band == 1:
        im_ref_mean = np.mean(ms1)
        im_ref_std = np.std(ms1)
        f_pan1 = (pan - im_mean) * (im_ref_std / im_std) + im_ref_mean;
    elif band == 2:
        im_ref_mean = np.mean(ms2)
        im_ref_std = np.std(ms2)
        f_pan2 = (pan - im_mean) * (im_ref_std / im_std) + im_ref_mean;
    elif band == 3:
        im_ref_mean = np.mean(ms3)
        im_ref_std = np.std(ms3)
        f_pan3 = (pan - im_mean) * (im_ref_std / im_std) + im_ref_mean;

def fft2_pan_mt(band):
    import numpy as np
    global f_pan1
    global f_pan2
    global f_pan3
    if band == 1:
        f_pan1 = np.fft.fft2(f_pan1)
    elif band == 2:
        f_pan2 = np.fft.fft2(f_pan2)
    elif band == 3:
        f_pan3 = np.fft.fft2(f_pan3)

def fft2_ms_mt(band):
    import numpy as np
    global ms1
    global ms2
    global ms3
    global f_ms1
    global f_ms2
    global f_ms3
    if band == 1:
        f_ms1 = np.fft.fft2(ms1)
    elif band == 2:
        f_ms2 = np.fft.fft2(ms2)
    elif band == 3:
        f_ms3 = np.fft.fft2(ms3)


def filter_pan_mt(band):
    global f_pan1
    global f_pan2
    global f_pan3
    global h_high
    if band == 1:
        f_pan1 = f_pan1 * h_high
    elif band == 2:
        f_pan2 = f_pan2 * h_high
    elif band == 3:
        f_pan3 = f_pan3 * h_high

def filter_ms_mt(band):
    global f_ms1
    global f_ms2
    global f_ms3
    global h_low
    if band == 1:
        f_ms1 = f_ms1 * h_low
    elif band == 2:
        f_ms2 = f_ms2 * h_low
    elif band == 3:
        f_ms3 = f_ms3 * h_low
        
def create_f_ps_im_mt(band, hist_m):
    global f_pan
    global f_pan1
    global f_ms1
    global f_ps1    
    global f_pan2    
    global f_ms2
    global f_ps2    
    global f_pan3
    global f_ms3
    global f_ps3    
    if hist_m:
        if band == 1:
            f_ps1 = f_pan1 + f_ms1
        elif band == 2:
            f_ps2 = f_pan2 + f_ms2
        elif band == 3:
            f_ps3 = f_pan3 + f_ms3
    else:
        if band == 1:
            f_ps1 = f_pan + f_ms1
        elif band ==2:
            f_ps2 = f_pan + f_ms2
        elif band == 3:
            f_ps3 = f_pan + f_ms3
            
def ifft2_mt(band):
    import numpy as np
    global f_ps1
    global f_ps2
    global f_ps3
    global ps1
    global ps2
    global ps3
    if band == 1:
        ps1 = np.fft.ifft2(f_ps1)
    elif band == 2:
        ps2 = np.fft.ifft2(f_ps2)
    elif band == 3:
        ps3 = np.fft.ifft2(f_ps3)
    
        
def pansharpenning_mt(ps_method, pan, ms1, ms2, ms3, filter_name, cutoff_freq=.125, hist_m=True):
    import threading as th
    import numpy as np
    if ps_method == 'fft':
        m, n = np.shape(pan)
        h_low = ffilters(filter_name, m, n, cutoff_freq, 1)
        h_high = 1 - h_low
        f_pan1 = np.empty((m,n), dtype='float64')
        f_pan2 = np.empty((m,n), dtype='float64')
        f_pan3 = np.empty((m,n), dtype='float64')
        f_ms1 = np.empty((m,n), dtype='complex128')
        f_ms2 = np.empty((m,n), dtype='complex128')
        f_ms3 = np.empty((m,n), dtype='complex128')
        f_ps1 = np.empty((m,n), dtype='complex128')
        f_ps2 = np.empty((m,n), dtype='complex128')
        f_ps3 = np.empty((m,n), dtype='complex128')
        ps1 = np.empty((m,n), dtype='float64')
        ps2 = np.empty((m,n), dtype='float64')
        ps3 = np.empty((m,n), dtype='float64')
        # Initialize threads
        # Part 1 - Histogram matching
        th1 = th.Thread(target=hist_match_mt, args=(1,))
        th2 = th.Thread(target=hist_match_mt, args=(2,))
        th3 = th.Thread(target=hist_match_mt, args=(3,))
        # Part 2 - FFT PAN
        th4 = th.Thread(target=fft2_pan_mt, args=(1,))
        th5 = th.Thread(target=fft2_pan_mt, args=(2,))
        th6 = th.Thread(target=fft2_pan_mt, args=(3,))
        # Part 3 - Filtering PAN
        th7 = th.Thread(target=filter_pan_mt, args=(1,))
        th8 = th.Thread(target=filter_pan_mt, args=(2,))
        th9 = th.Thread(target=filter_pan_mt, args=(3,))
        # Part 4 - FFT MS
        th10 = th.Thread(target=fft2_ms_mt, args=(1,))
        th11 = th.Thread(target=fft2_ms_mt, args=(2,))
        th12 = th.Thread(target=fft2_ms_mt, args=(3,))
        # Part 5 - Filtering MS
        th13 = th.Thread(target=filter_ms_mt, args=(1,))
        th14 = th.Thread(target=filter_ms_mt, args=(2,))
        th15 = th.Thread(target=filter_ms_mt, args=(3,))
        # Part 6 - Creating Pan-Sharpenned Image
        th16 = th.Thread(target=create_f_ps_im_mt, args=(1, hist_m))
        th17 = th.Thread(target=create_f_ps_im_mt, args=(2, hist_m))
        th18 = th.Thread(target=create_f_ps_im_mt, args=(3, hist_m))
        # Part7 - I-FFT of PS Image
        th19 = th.Thread(target=ifft2_mt, args=(1,))
        th20 = th.Thread(target=ifft2_mt, args=(2,))
        th21 = th.Thread(target=ifft2_mt, args=(3,))
        if hist_m:            
            # Run theareds
            th1.start()
            th2.start()
            th3.start()
            
            th1.join()
            th4.start()
            
            th2.join()
            th5.start()
            
            th3.join()
            th6.start()

            th4.join()
            th7.start()
            th5.join()
            th8.start()
            th6.join()
            th9.start()
            
            th10.start()
            th11.start()
            th12.start()
            
            th10.join()
            th13.start()
            th11.join()
            th14.start()
            th12.join()
            th15.start()
            
            th7.join()
            th13.join()
            th16.start()
            th8.join()
            th14.join()
            th17.start()
            th9.join()
            th15.join()
            th18.start()
            
            th16.join()
            th19.start()
            th17.join()
            th20.start()
            th18.join()
            th21.start()
        else:
            # before run Threads 14-15-16
            f_pan = np.fft.fft2(pan)
            f_pan = f_pan * h_high
            # Run threads
            th10.start()
            th11.start()
            th12.start()
            
            th10.join()
            th13.start()
            th11.join()
            th14.start()
            th12.join()
            th15.start()
            
            th13.join()
            th16.start()
            th14.join()
            th17.start()
            th15.join()
            th18.start()
            
            th16.join()
            th19.start()
            th17.join()
            th20.start()
            th18.join()
            th21.start()
    elif ps_method == '':
        pass
    return ps1, ps2, ps3










def npfft2(np_in):
    import numpy as np
    return np.fft.fft2(np_in)
    #print("FINISH %s " % threading.current_thread())

def np_fft2_v2(np_in,):
    import numpy as np
    return np.fft.fft2(np_in)

def np_ifft2_v2(np_in,):
    import numpy as np
    return np.fft.ifft2(np_in)

def np_ifft2_v2(np_in, np_out, xy):
    import numpy as np
    np_out[xy[0]:xy[1], xy[2]:xy[3]] = np.fft.ifft2(np_in)

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def test_fft_ifft_single(np_im):
    import numpy as np
    # import time
    # single core
    x, y, z = np.shape(np_im)
    try:
        fft_im = np.empty((np.shape(np_im)), dtype='complex128')
        ifft_im = np.empty((np.shape(np_im)), dtype='float64')
        for i in range(z):
            fft_im[:, :, i] = np.fft.fft2(np_im[:, :, i])
        for i in range(z):
            ifft_im[:, :, i] = np.fft.ifft2(fft_im[:, :, i])
        return ifft_im
    except Exception as e:
        print e.message, e.args
        
def test_fft_ifft_mp_pool(np_im, pool_size=2):
    from multiprocessing import Pool
    import numpy as np
    x, y, z = np.shape(np_im)
    try:
        np_ims = np_im.reshape((z*(pool_size**pool_size), x/pool_size, y/pool_size))
        fft_ims = np.empty((z*(pool_size**pool_size), x/pool_size, y/pool_size), dtype='complex128')
        ifft_ims = np.empty((z*(pool_size**pool_size), x/pool_size, y/pool_size), dtype='float64')
        pool = Pool(pool_size)
        fft_ims = pool.map(np.fft.fft2, np_ims)
        ifft_ims = pool.map(np.fft.ifft2, fft_ims)
        return ifft_ims
    except Exception as e:
        print e.message, e.args

