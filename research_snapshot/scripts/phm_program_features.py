"""Shared actual-signal feature helpers,available identically to both arms."""
import numpy as np


def time_stats(waves):
    x=np.asarray(waves,dtype=float);assert x.ndim==3 and x.shape[2]==6
    centered=x-x.mean(axis=1,keepdims=True);std=np.sqrt(np.mean(centered**2,axis=1));rms=np.sqrt(np.mean(x*x,axis=1))
    z=centered/np.maximum(std[:,None,:],1e-12)
    return np.concatenate([x.mean(axis=1),np.log(np.maximum(std,1e-12)),np.log(np.maximum(rms,1e-12)),np.ptp(x,axis=1),
        np.mean(z**3,axis=1),np.mean(z**4,axis=1),np.max(np.abs(x),axis=1)/np.maximum(rms,1e-12),np.mean(np.abs(x),axis=1)],axis=1)


def spectral_bands(waves,bands=32):
    x=np.asarray(waves,dtype=float);p=np.abs(np.fft.rfft(x-x.mean(axis=1,keepdims=True),axis=1))**2
    # Equal array-frequency bands;use supplied actual sampling rate for physicalHz.
    return np.concatenate([np.log1p(a.mean(axis=1)) for a in np.array_split(p[:,1:],bands,axis=1)],axis=1)
