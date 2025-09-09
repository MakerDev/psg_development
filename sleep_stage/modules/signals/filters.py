import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert
from scipy.interpolate import interp1d

from mne.filter import filter_data

def band_pass_filter(x, lf, hf, fs=100, tb='auto'):

    return filter_data(
        x, sfreq=fs, l_freq=lf, h_freq=hf, verbose=False,
        l_trans_bandwidth=tb, h_trans_bandwidth=tb)

def trim_sigma(x, sigma=0.999):

    x = x.copy()
    pq = np.quantile(x, sigma)
    mask_positive = x > pq
    x[mask_positive] = pq

    nq = np.quantile(x, 1 - sigma)
    mask_negative = x < nq
    x[mask_negative] = nq 

    return x 

def rolling(x, window=100, min_periods=1):

    return np.squeeze(
        pd.DataFrame(x).rolling(
            window=100, min_periods=1).mean().values)

def subtract_rolling(x, window=100, min_periods=1):
    
    return x-rolling(x, window=100, min_periods=1)

def peakline(x, distance=10, height=0, with_neg=False, upper=False):

    if upper :
        height = np.quantile(np.abs(x), height)
        
    peaks, _ = find_peaks(x, height=height, distance=distance)

    if with_neg:
        peaks_neg, _ = find_peaks(-x, height=height, distance=distance)
        peaks = np.sort(np.concatenate([peaks, peaks_neg]))

    f_interp1d = interp1d(peaks, x[peaks], fill_value="extrapolate")
    
    return f_interp1d(
        np.linspace(0, len(x), num=len(x), endpoint=True))

def zero_interpolate(data):

    x = np.where(data > 1000)[0]
    y = data[x]
    f = interp1d(x, y, fill_value='extrapolate')
    data = f(np.arange(0, len(data), 1))

    return data 

def envelope(x):
    return np.abs(hilbert(x))

def add_noise(x):

    return x + np.random.normal(
        abs(np.min(x))/10000,abs(np.min(x))/10001,len(x))

def mm_scaling(x):

    return (x - np.min(x))/ (np.max(x) - np.min(x))

def zero_symmetry_scaling(x):

    pos_mask = x >= 0
    neg_mask = x <  0
    x = np.abs(x)
    
    if len(x[pos_mask]) > 0:
        x[pos_mask] = np.apply_along_axis(func1d=mm_scaling, axis=0, arr=x[pos_mask])

    if len(x[neg_mask]) > 0:
        x[neg_mask] = np.apply_along_axis(func1d=mm_scaling, axis=0, arr=x[neg_mask])

    x[neg_mask] = x[neg_mask] * -1

    return x

def log_quantize(x, base_n=16):
    # Ony zero to one scaled sequence can be input x.
    x = x*2-1
    neg_mask  = (x<0)
    zeor_mask = (x==0)
    y = np.round(
        np.log2(np.abs(x)*np.exp2(int(base_n/2))+np.finfo(float).eps))
    
    y[y <= 0] =0
    y[neg_mask] =-y[neg_mask]
    y[zeor_mask] = 0

    return y

def norm_quantize(x, base_n=16):
    # Ony zero to one scaled sequence can be input x.
    x = x * base_n
    return np.round(x)

def pad(x, size, pad_num=0):
  
    shape_front = x.shape[0] 
    shape_back  = x.shape[1:]

    n_pad = ((int(int(shape_front)/size)+1) * size) - shape_front

    x = np.concatenate([ x, np.zeros(((n_pad,)+ shape_back))+pad_num ])
    shape = (-1, size,) + shape_back
    x = x.reshape(*shape)
        
    return x

def zero_padding(size=120):

    def wrap(func):
        def wrap_f(*arg, **kargs):

            x = func(*arg,**kargs)
            x = pad(x, size=size)

            return x 
        return wrap_f
    return wrap 

def activity(x, window_size):

    return pd.Series(x).rolling(window_size, min_periods=1).var().values 

def mobility(x, window_size):

    w  = window_size
    dx = pd.Series(x).diff().values

    return np.sqrt(activity(dx,w)/activity(x,w))

def complexity(x, window_size):

    w  = window_size
    dx = pd.Series(x).diff().values

    return mobility(dx,w)/mobility(x,w)