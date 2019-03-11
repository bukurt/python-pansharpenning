from functions import load_datas_for_pansharpenning
from functions import pansharpenning

path_im_ms = './data/spot6/GEBZE/S6_GEBZE_MS.tiff'
path_im_pan = './data/spot6/GEBZE/S6_GEBZE_PAN.tiff'
path_xml = './data/spot6/GEBZE/S6_GEBZE_MS.XML'

pan, ms1, ms2, ms3 = load_datas_for_pansharpenning(path_im_pan, path_im_ms)
ps1, ps2, ps3 = pansharpenning('fft', pan, ms1, ms2, ms3, 'ideal_low', hist_m=False, cutoff_freq=.125)
