import mne 
import numpy as np
from mne.time_frequency.multitaper import _compute_mt_params

def conv_spectrogram(epochs, fmin=0, fmax=200, sfreq=500.0, n_epoch=30 ):

    return np.array(list(map(
            lambda epoch: np.array([
                mne.time_frequency.psd_array_multitaper(
                    epoch[i*int(sfreq): (i+2)*int(sfreq)].reshape(1,-1), 
                    sfreq, fmin, fmax, verbose=False)[0].reshape(-1)
                for i in range(n_epoch-1)]), 
            epochs)))



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
        fmax = np.array([float('inf')], dtype=np.float32).astype(np.int32)
    
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
    
    mt_params['dpss'] = dpss
    mt_params['eigvals'] = eigvals
    
    freqs = np.fft.rfftfreq(n_times, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    mt_params['freq_mask'] = freq_mask
    
    return mt_params, freqs