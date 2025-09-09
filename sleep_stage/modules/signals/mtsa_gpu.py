import cupy as cp
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from mne.time_frequency.multitaper import _compute_mt_params

memfree = cp.get_default_memory_pool().free_all_blocks
pmemfree = cp.get_default_pinned_memory_pool().free_all_blocks

"""
THIS IS MTSA PREPROCESSIONG MODULE. 
Implemented by Paul Kim from HoneyNaps.
"""

def psd_array_multitaper(x, sfreq, mt_params):
    """ calculate multitaper fft for psd array.

    Caution:
        Before call this function, you should get mt_params 
        from get_mt_params function with same sfreq and padding length.
    
    Arguments:
        x         {array}      -- shape: (..., n_dim) 
        sfreq     {float}      -- sample frequency e.g. 500.o
        mt_params {dictionary} -- return value from get_mt_params function. 

    return: psd_array
    """

    if mt_params is None:
        raise ValueError(
            'mt_params shold be filled with dpss, eigvals')
    
    # Get parameter for MTSA 
    ndim_in = x.ndim
    dshape = x.shape[:-1]
    
    dpss = mt_params['dpss']
    eigvals = mt_params['eigvals']
    freq_mask = mt_params['freq_mask']
    
    # Calculate PSD MTSA . 
    x_mt = mt_spectra(x, dpss, sfreq)[0]
    weights = cp.sqrt(eigvals)[cp.newaxis, :, cp.newaxis]
    psd = psd_from_mt(x_mt[:, :, freq_mask], weights)
    
    # Combining/reshaping to original data shape
    psd.shape = dshape + (-1, )
    if ndim_in == 1:
        psd = psd[0]

    return psd


def psd_from_mt(x_mt, weights):
    """Compute PSD from tapered spectra.

    Parameters
    ----------
    x_mt : array
        Tapered spectra
    weights : array
        Weights used to combine the tapered spectra

    Returns
    -------
    psd : array
        The computed PSD
    """
    psd = weights * x_mt
    psd *= psd.conj()
    psd = psd.real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd


def mt_spectra(x, dpss, sfreq, n_fft=None):
    """Compute tapered spectra.

    Parameters
    ----------
    x : array, shape=(..., n_times)
        Input signal
    dpss : array, shape=(n_tapers, n_times)
        The tapers
    sfreq : float
        The sampling frequency
    n_fft : int | None
        Length of the FFT. If None, the number of samples in the input signal
        will be used.

    Returns
    -------
    x_mt : array, shape=(..., n_tapers, n_times)
        The tapered spectra
    freqs : array
        The frequency points in Hz of the spectra
    """


    # x = cp.asarray(x)
    if n_fft is None:
        n_fft = x.shape[1]

    # remove mean (do not use in-place subtraction as it may modify input x)
    x = x - cp.mean(x, axis=-1, keepdims=True)

    # only keep positive frequencies
    freqs = cp.fft.rfftfreq(n_fft, 1. / sfreq)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = cp.zeros(x.shape[:-1] + (n_tapers, len(freqs)),
                    dtype=cp.complex128)
    for idx, sig in enumerate(x):
        x_mt[idx] = cp.fft.rfft(sig[..., cp.newaxis, :] * dpss, n=n_fft)

    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[:, :, 0] /= cp.sqrt(2.)
    if x.shape[1] % 2 == 0:
        x_mt[:, :, -1] /= cp.sqrt(2.)

    return x_mt, freqs


def get_mt_params(sfreq=500.0, padding=2, fmin=0, fmax=None):
    """This function generate parameters for psd_array_multitaper.
    
    Caution:
        Before call this function, you should get mt_params 
        from get_mt_params function with same sfreq and padding length.
    
    Keyword Arguments:
        sfreq {float} -- The sampling frequency (default: {500.0})
        padding {int} -- The seconds of terget length of spectrum. (default: {2})
        fmin {int} -- Minimum value of interesting frequncy area. (default: {0})
        fmax {int} -- Maximum value of interesting frequncy area. (default: {None})
                    - If you set fmax is none, sfreq/2 will be fmax. 
                    - Over sfreq/2 can't be a fmax value.
    
    Return: mt_prams {dict}
    """
    if fmax == None: 
        fmax = cp.array([float('inf')], dtype=np.float32).astype(np.int32)
    
    # 2. Get MTSA parameters 
    n_times = padding*sfreq 
    
    mt_params = dict()
    dpss, eigvals, _ = _compute_mt_params(
            n_times=n_times,
            sfreq=sfreq,
            bandwidth=None,
            low_bias=True,
            adaptive=False,
            verbose=False
    )
    
    mt_params['dpss'] = cp.asarray(dpss)
    mt_params['eigvals'] = cp.asarray(eigvals)
    
    freqs = cp.fft.rfftfreq(n_times, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = cp.asnumpy(freqs[freq_mask])
    mt_params['freq_mask'] = cp.asarray(freq_mask)
    
    return mt_params, freqs
    

def conv_spectrogram(epochs, mt_params, 
    sfreq=500.0, n_epoch=30 ):
    """ Spectrogram converter. 
    - This function desigened only for StageNet preprocessing. 
    - Recommend only input 30s of epochs for Argument epochs. 
    - You can get mt_params by calling get_mt_params function.

    Arguments:
        epochs {array} -- (..., n_epoch*sfreq) e.g. (..., 15000)
        mt_params {dictionary} -- return values of get_mt_params functinon.
    
    Keyword Arguments:
        sfreq {float} -- [description] (default: {500.0})
        n_epoch {int} -- [description] (default: {30})
    """
    epochs = cp.asarray(epochs)

    spectogram = []
    for epoch in epochs:
        image = []
        for i in range(n_epoch-1):
            psd_array_multitaper(
                epoch[i*int(sfreq): (i+2)*int(sfreq)].reshape(1,-1), sfreq, mt_params)
            
        spectogram.append(cp.concatenate(image))
    
    spectogram = cp.asnumpy(cp.stack(spectogram))
   
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    # def mtsa(epoch):

    #     image = [
    #         psd_array_multitaper(
    #             epoch[i*int(sfreq): (i+2)*int(sfreq)].reshape(1,-1), sfreq, mt_params)
    #         for i in range(n_epoch-1)
    #     ]

    #     return cp.concatenate(image)
        
    # spectogram = cp.stack(list(map(mtsa, epochs)))

    # spectogram = cp.stack([ 
    #     cp.concatenate([ 
    #         10*cp.log(
    #             psd_array_multitaper(
    #                 epoch[i*int(sfreq): (i+2)*int(sfreq)].reshape(1,-1), 
    #                 sfreq, mt_params) )
    #         for i in range(n_epoch-1)])
    #     for epoch in tqdm(epochs, position=0)
    # ])
    
    return spectogram
    
    

    
    
    


def wraper_conv_spectrogram(args):
    """[summary]
    
    Arguments:
        args {[type]} -- [description]
    """
    epoch = args[0]
    mt_params =  args[1]
    
    return conv_spectrogram(epoch, mt_params)


def build_spectrogram(dict_epoch,
    sfreq=500, n_epoch = 30, n_job=6 ,
    CH_EEG = ['C3-M2', 'C4-M1', 'F3-M2', 'F4-M1', 'O1-M2', 'O2-M1']):
    """[summary]
    
    Arguments:
        dict_epoch {[type]} -- [description]
    
    Keyword Arguments:
        sfreq   {int} -- [description] (default: {500})
        n_epoch {int} -- [description] (default: {30})
        n_job   {int} -- [description] (default: {6})
        CH_EEG {list} -- [description] (default: {['C3-M2', 'C4-M1', 'F3-M2', 'F4-M1', 'O1-M2', 'O2-M1']})
    """

    mt_params1, _ = get_mt_params(sfreq=500, padding=2, fmin=0, fmax=20)
    mt_params2, _ = get_mt_params(sfreq=500, padding=2, fmin=0, fmax=100)

    dict_parameter = [
        (dict_epoch[signal], mt_params1) if signal in CH_EEG else 
        (dict_epoch[signal], mt_params2) for signal in dict_epoch
    ]


    results = [
        conv_spectrogram(epoch, params)
        for epoch, params in dict_parameter
    ]

    dict_spectrogram = {
        signal:result
        for signal, result in zip(dict_epoch, results)
    }
    
    return dict_spectrogram