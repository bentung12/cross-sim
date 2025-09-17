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


class SONOS_TID(EmptyDevice):
    """This is an empirical programming error and drift model for the 40nm SONOS charge trapping memory
    described in:
    V. Agrawal, et al, "Subthreshold operation of SONOS analog memory to enable accurate low-power
    neural network inference", IEEE International Electron Devices Meeting (IEDM) 2022, 21.7.1-21.7.4.
    https://ieeexplore.ieee.org/abstract/document/10019564/.

    SUGGESTED ON/OFF RATIO : 1e7 (~0 to 16 uS)

    Three non-idealities are modeled, based on the statistics of experimentally measured SONOS device
    currents from 128K devices. Both effects are time dependent. For details, see the paper.
    1) Drift in the expected value of the SONOS current, relative to the value at time of programming.
        By definition, this is not applied if time = 0.
    2) Random variability in the SONOS current around the expected value, due to random write errors and
        device-to-device variations.
    3) Read noise in the SONOS current around the programmed value

    The first effect is shown in Fig. 10(b) of the paper. At each point in time, the dependence of the
    drift vs SONOS state is fit using a 10th degree polynomial with 11 coefficients. This is evaluated
    and added to the target currents to get the mean values after drift. The coefficients were extracted
    for measurements taken at t = 1, 2, 3, 4, and 5 days. For values of t between these time steps, the
    coefficients are interpolated.

    The second effect is shown in Fig. 10(a) of the paper. At each point in time, the dependence of the
    random variation vs SONOS state is fit using a saturating exponential function with 2 free parameters
    (A and B). This function is used to scale a 2D matrix of random normal numbers and this is added to the
    SONOS current values, after mean drift is applied. The values of A and B were extracted for measurements
    taken at t = 1, 2, 3, 4, and 5 days. For values of t between these steps, A and B are interpolated.
    If time > 5 days, the polynomial coefficients and A and B will be extrapolated, and are not guaranteed
    to be accurate.

    The third effect is shown in Fig. 11(b) of the paper. The state dependence of the variance of read noise
    is modeled by a saturating exponential function similar to device-to-device variability, but with a
    different pair of coefficients: A_noise and B_noise. This function is used to scale a 2D matrix of random
    normal numbers, and this is added to the SONOS current values every time an MVM is called. These
    coefficients do not vary with time.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert xbar normalized conductances to real PCM conductances
        # Max SONOS current to use for weight storage
        self.Imax = 2000  # nanoAmps
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

    def programming_error(self, input_):
        """See documentation in drift_error()."""
        return self.drift_error(input_, time=-1)

    ######################################
    ######################################
    # In this function, time is TID in rad!!!
    ######################################
    ######################################
    def drift_error(self, input_, time):

        TID = time

        TID_data = pickle.load(open("TID_data_0802_CrossSim.p","rb"))
        TID_vec = TID_data['TID']*1e3
        Imeans_vec = TID_data['I_means']
        Istds_vec = TID_data['I_stds']

        # If TID is negative, use the as-programmed mean and stds, then set
        # TID = 0 for the rest of the calculation
        if TID < 0:
            Imeans_vec[0,:] = TID_data['I_means_prog']
            Istds_vec[0,:] = TID_data['I_stds_prog']
            TID = 0

        #Add this line later
        if TID == 0:
            return input_
        
        # Subtract characterized ADC noise from std measurements
        Istds_vec -= 1.223925

        Id_initial = Imeans_vec[0,:] # Zero TID
        Istd_initial = Istds_vec[0,:]

        input_norm = np.max(Id_initial)/self.Imax
        
        I = self._calculate_current(input_norm*input_)

        if xp.use_gpu:
            I = xp.asnumpy(I)

        # If TID matches a measured data point, no need to interpolate
        if TID in TID_vec:
            ind_TID = np.argmin(np.abs(TID - TID_vec))
            Id_final = Imeans_vec[ind_TID,:]
            Istd_final = Istds_vec[ind_TID,:]

        # Interpolate between the measured TIDs
        else:
            Id_final = np.zeros(len(Id_initial))
            for i in range(len(Id_final)):
                interp_func0 = interp1d(TID_vec,Imeans_vec[:,i],kind='linear',copy=True,fill_value='extrapolate')
                Id_final[i] = interp_func0(TID)

            Istd_final = np.zeros(len(Istd_initial))
            for i in range(len(Istd_final)):
                interp_func0 = interp1d(TID_vec,Istds_vec[:,i],kind='linear',copy=True,fill_value='extrapolate')
                Istd_final[i] = interp_func0(TID)

        # Interpolate TID mean shift vs conductance
        if TID > 0:
            interp_func_mean = interp1d(Id_initial,Id_final,kind='linear',copy=True,fill_value='extrapolate')
            I = interp_func_mean(I)
            # Clip if interpolated outside valid range
            I[I < self.Imin] = self.Imin

        # Get stdev vs conductances
        interp_func_std = interp1d(Id_final,Istd_final,kind='linear',copy=True,fill_value='extrapolate')
        sigma_I = interp_func_std(I)
        sigma_I[sigma_I < 0] = 0

        if xp.use_gpu:
            I = xp.array(I)
            sigma_I = xp.array(sigma_I)

        sigma_W = sigma_I / (self.Imax - self.Imin)
        input_ = self.Gmin_norm + self.Grange_norm*(I - self.Imin)/(self.Imax - self.Imin)
        input_ /= input_norm
        sigma_W /= input_norm

        if sigma_W.any():
            randMat = xp.random.normal(scale=self.Grange_norm, size=input_.shape).astype(input_.dtype)
            input_ = input_ + sigma_W * randMat

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
