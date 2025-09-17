#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from ..idevice import EmptyDevice
import numpy as np
from scipy.interpolate import interp1d
from ...backend import ComputeBackend
import pickle

xp = ComputeBackend()


class SONOS_TID_202312(EmptyDevice):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert xbar normalized conductances to real PCM conductances
        # Max SONOS current to use for weight storage
        self.Imax = 700  # nanoAmps
        if self.on_off_ratio == 0:
            self.Imin = 0
        else:
            self.Imin = self.Imax / self.on_off_ratio

    def _calculate_current(self, input_):
        """Computes matrix of SONOS cell currents that map a matrix of normalized input values
        Current is treated equivalently to conductance here since the drain-source
        voltage bias on the SONOS cell only has one non-zero value.
        """
        I = (
            self.Imin
            + (self.Imax - self.Imin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )
        return I

    def programming_error(self, I):
        """
        For this TID model, do not call this from outside drift_error()!!!
        Input I must be in nanoamps
        """
        
        A = 19.88665
        B = 176.3115

        # Apply random variability
        sigma_I = xp.maximum(A - A * xp.exp(-I / B), 0)

        if sigma_I.any():
            randMat = xp.random.normal(
                scale=1,
                size=I.shape,
            ).astype(I.dtype)
            I = I + sigma_I * randMat

        # Make sure resulting current is non-negative
        I = I.clip(0, None)

        return I

    ######################################
    ######################################
    # In this function, time is TID in rad!!!
    ######################################
    ######################################
    def drift_error(self, input_, time):

        # TID is passed as rad, convert to krad
        TID = time / 1e3

        # First convert input values to current
        I = self._calculate_current(input_)

        # Apply programming error to the currents
        I = self.programming_error(I)

        model_params = pickle.load(open("TID_params_202312.p","rb"))
        TID_vec = model_params['TID_vec']
        Ibins = model_params['Ibins']
        p_means = model_params['params'][:,:,0]
        p_stds = model_params['params'][:,:,1]
        deg = p_means.shape[1]-1

        if TID < 0:
            raise ValueError("TID cannot be negative")
        elif TID < TID_vec[0]:
            p_m = p_means[0,:] * TID/TID_vec[0]
            p_s = p_stds[0,:] * TID/TID_vec[0]
        elif TID > TID_vec[-1]:
            raise ValueError("TID too large: max TID compatible with model is {:.2f} krad".format(xp.max(TID_vec)))
        else:
            # Find index of TID value in model that is closest to queried value
            ind = np.argmin(np.abs(TID_vec-TID))
            if TID_vec[ind] < TID:
                ind1 = ind
                ind2 = ind+1
            else:
                ind1 = ind-1
                ind2 = ind
            p_m = (p_means[ind1,:] + p_means[ind2,:])/2
            p_s = (p_stds[ind1,:] + p_stds[ind2,:])/2

        # Get the array of current shifts based on mean behavior
        Ith_mean = 500
        shift_mean = xp.zeros(I.shape)
        shift_th = 0
        for d in range(deg+1):
            shift_mean += p_m[d]*I**(deg-d)
            shift_th += p_m[d]*Ith_mean**(deg-d)
        # Correct portion where the polynomial fit does poorly
        shift_mean[I>Ith_mean] = shift_th

        # Get the array of the stochastic component of the shift
        Ith_std = 450
        shift_std = xp.zeros(I.shape)
        shift_th = 0
        for d in range(deg+1):
            shift_std += p_s[d]*I**(deg-d)
            shift_th += p_s[d]*Ith_std**(deg-d)
        # Correct portion where the polynomial fit does poorly
        shift_std[I>Ith_std] = shift_th

        # Make sure mean and std of shift are non-negative
        shift_mean[shift_mean < 0] = 0
        shift_std[shift_std < 0] = 0

        # Compute the combined deterministic + stochastic shift for every device
        randMat = xp.random.normal(
                scale=1,
                size=I.shape,
            ).astype(I.dtype)
        shifts = shift_mean + randMat*shift_std

        # Apply the shift
        I = I + shifts

        # Normalize
        input_ = self.Gmin_norm + self.Grange_norm*(I - self.Imin)/(self.Imax - self.Imin)

        # Make sure resulting current is non-negative
        input_ = input_.clip(0, None)

        return input_


    def read_noise(self, input_):
        """Apply read noise."""
        I = self._calculate_current(input_)

        A_noise, B_noise = 12.58037, 215.36557

        # Apply read noise
        sigma_I = xp.maximum(A_noise - A_noise * xp.exp(-I / B_noise), 0)
        sigma_W = sigma_I / (self.Imax - self.Imin)

        if sigma_W.any():
            randMat = xp.random.normal(
                scale=self.Grange_norm,
                size=input_.shape,
            ).astype(input_.dtype)
            input_ = input_ + sigma_W * randMat

        # Make sure resulting current is non-negative
        input_ = input_.clip(0, None)

        return input_
