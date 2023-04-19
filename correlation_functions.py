# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:12:04 2021

@author: RMCGORTY
"""
import numpy as np
import pickle
import scipy
from scipy.interpolate import interp1d
import skimage
from skimage import io, exposure, filters
from numpy.fft import fftshift, fft2, ifft2, fft, ifft
from numpy import real
from numpy import conj

def autocorrelation_2D(image):
    '''
    Computes the 2D autocorrelation
    '''
    data = image-image.mean() 
    data = data / data.std()
    
    temp = real(fftshift(ifft2(fft2(data)*conj(fft2(data)))))
    temp = temp / (data.shape[0]*data.shape[1])
    
    return temp

def autocorrelation_1D(image, ax):
    # This function computes a 1D correlation
    # of a 2D image. 
    # Input parameters:
    #    image -- a 2D numpy array
    #    ax -- the axis to perform the autocorrelation
    #
    # Note about which axes to use:
    #   for ax = 0: this will correlate things along the vertical direction when displayed using matshow
    #   for ax = 1: this will correlate along the horizontal direction when displayed using matshow
    
    means = image.mean(axis=ax)
    temp = np.tile(means, (image.shape[ax], 1))
    if ax==0:
        data = image - temp
    elif ax==1:
        data = image - temp.transpose()
    else:
        print("ax must be 0 or 1")
        return 0
    
    std_devs = data.std(axis=ax)
    #print("shape of std_devs: ", std_devs.shape) #for debugging purposes
    temp = np.tile(std_devs, (image.shape[ax],1))
    if ax==0:
        data = data / temp
    elif ax==1:
        data = data / temp.transpose()
        
    temp = fftshift(ifft(fft(data,axis=ax)*conj(fft(data,axis=ax)),axis=ax),axes=ax)
    temp = real(temp) / data.shape[ax]
       
    if ax==0:
        corr = temp.sum(axis=1)/temp.shape[1]
        return corr[int(temp.shape[0]/2):]
    elif ax==1:
        corr = temp.sum(axis=0)/temp.shape[0]
        return corr[int(temp.shape[1]/2):]
    
def find_where_corr_at_half(corr, val=0.5):
    # This function will actually only find where the correlation function goes to 0.5
    # (which, if properly normalized, means that it dropped 50%) if the "val" is set
    # to 0.5 which is the default. However, you can change that to another value like
    # 0.2 or 1/e or something else. 
    f = interp1d(np.arange(0,len(corr)), corr, kind='linear', fill_value="extrapolate") #create interpolation function
    new_x = np.linspace(0, len(corr), 10*len(corr))  #new x-axis that is 10 times more sampled
    eval_at_new_x = f(new_x) #evaluate the interpolated function over the new range of x-values
    min_index = np.argmin(abs(eval_at_new_x-val)) #find where that is closest to "val" 
    return new_x[min_index]


def find_radial_average(im, mask=None, centralAngle=None, angRange=None,
                        remove_vert_line=False,
                        remove_hor_line=False):
    r"""
    Computes the radial average of a single 2D matrix.

    Parameters
    ----------
    im : array
        Matrix to radially average
    mask : array or None, optional
        If desired, use a mask to avoid radially averaging the whole array. The default is None.
    centralAngle : float
        Central angle of the angular range to average over
    angRange : float
        Range to overage over.

    Returns
    -------
    array
        Radial average of the passed array `im`

    """
    #From https://github.com/MathieuLeocmach/DDM/blob/master/python/DDM.ipynb
    nx,ny = im.shape

    if (centralAngle!=None) and (angRange!=None) and (mask==None):
        mask = generate_mask(im, centralAngle, angRange)
    elif mask==None:
        mask = np.ones_like(im)
        
    if remove_vert_line:
        im[:,int(ny/2)]=0
    if remove_hor_line:
        im[int(nx/2),:]=0

    dists = np.sqrt(np.arange(-1*nx/2, nx/2)[:,None]**2 + np.arange(-1*ny/2, ny/2)[None,:]**2)


    bins = np.arange(max(nx,ny)/2+1)
    histo_of_bins = np.histogram(dists, bins)[0]
    h = np.histogram(dists, bins, weights=im*mask)[0]
    return h/histo_of_bins