import xarray as xr
import numpy as np
import random
import netCDF4 as nc
import logging
import numpy.typing as npt
import pop_tools
from typing import Any
from pathlib import Path
from scipy.stats import norminvgauss

_logger = logging.getLogger(__name__)

class ERA5PCForcingGenerator:
    """A class encapsulating ERA5-PC based model forcing generation.

    This class gather functionalities to generate forcing noise
    on E-P and T@2m using A. Boot analysis published in:
    https://esd.copernicus.org/articles/16/115/2025/
    Two PC-based forcing are implemented: PC(NIG) and the PC(AR)
    autoregressive model only available in an early
    version of the paper.

    Attributes:
        _data_path : the path towards PC-AR data and grid data
        _nr_eofs : total number of EOFs in E-P and T@2m
    """
    def __init__(self,
                 data_path : str,
                 model : str) -> None:
        """Initialize the generator."""
        self._nr_eofs = -1
        self._nr_eofs_ep = -1
        self._nr_eofs_t2m = -1
        self._data_path : str = data_path
        self._rng = None
        self._spinup_file = None
        self._model = model
        self._nig_params = None
        if not Path(data_path).is_dir():
            err_msg = f"Wrong ERA5-PC forcing model data path {data_path}"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        if model not in ["PC-AR", "PC-NIG"]:
            err_msg = f"Wrong ERA5-PC model {model} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)


    def set_rng(self,
                randomgen) -> None:
        """Set the random number generator.

        Args:
            An initialized random number generator
        """
        self._rng = randomgen

    def set_spinup_file(self,
                        ERA5PCARSpinUpfile : str) -> None:
        """Set the spinup PC-AR file.

        Args:
            ERA5PCARSpinUpfile : the path for the NetCDF output spinup file
        """
        self._spinup_file = ERA5PCARSpinUpfile


    def get_spinup_file(self) -> str:
        """Get the spinup AR file.

        Returns:
            the path for the NetCDF output spinup file
        """
        if self._spinup_file is None:
            err_msg = "Unable to retrieve AR spinup file: not specified"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self._spinup_file

    def load_NIG_data(self) -> None:
        """Load the PC-NIG parameters for the EOFs."""
        # Check for NIG distributions parameter files
        if (not Path(f"{self._data_path}/params_era_p_only_PC_NIG.npy").exists() or
            not Path(f"{self._data_path}/params_era_t2m_PC_NIG.npy").exists()):
            err_msg = f"Missing PC-NIG param files in {self._data_path}"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Load the data
        ep_data = np.load(f"{self._data_path}/params_era_p_only_PC_NIG.npy")
        t2m_data = np.load(f"{self._data_path}/params_era_t2m_PC_NIG.npy")

        # Check the number of EOFs
        self._nr_eofs_ep = np.shape(ep_data)[1]
        self._nr_eofs_t2m = np.shape(t2m_data)[1]
        self._nr_eofs = self._nr_eofs_ep + self._nr_eofs_t2m

        # Combine in a single array
        self._nig_params = np.concatenate((ep_data,t2m_data), axis=1)


    def spinup_AR(self) -> None:
        """Spinup data for PC-AR model.

        The autoregressive model requires history data. This function
        generates a history and stores it in a NetCDF file.
        """
        if self._spinup_file is None:
            err_msg = "Unable to generate AR spinup: output file not specified"
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        # E-P data
        lag_re_ep = np.load(f"{self._data_path}/e_p_lags.npy")    # Lags
        rho_ep = np.load(f"{self._data_path}/e_p_yw_rho.npy")     # dim = nr_eofs x max lag
        sig_ep = np.load(f"{self._data_path}/e_p_yw_sigma.npy")   # dime = nr_eofs
        nr_ep = sig_ep.shape[0]
        l_m_ep = int(np.amax(lag_re_ep))                    # maximum lag
        hist_ep = np.zeros([nr_ep,l_m_ep])                  # spinup history

        # T2m data
        lag_re_t2m = np.load(f"{self._data_path}/t2m_lags.npy")    # Lags
        rho_t2m = np.load(f"{self._data_path}/t2m_yw_rho.npy")     # dim = nr_eofs x max lag
        sig_t2m = np.load(f"{self._data_path}/t2m_yw_sigma.npy")   # dime = nr_eofs
        nr_t2m = sig_t2m.shape[0]
        l_m_t2m = int(np.amax(lag_re_t2m))                    # maximum lag
        hist_t2m = np.zeros([nr_t2m,l_m_t2m])                 # spinup history

        # Total number of random number needed in the future
        self._nr_eofs_ep = nr_ep
        self._nr_eofs_t2m = nr_t2m
        self._nr_eofs = nr_ep + nr_t2m

        for nr_i in range(nr_ep): # Loop over all EOFs
            # Select the lag corresponding to the partial autocorrelation function
            lag = int(lag_re_ep[nr_i])
            for spin_it in range(lag):
                # Construct the AR(lag) process where the white noise
                # is scaled with 'sig'
                hist_ep[nr_i,l_m_ep-1] = (np.dot(hist_ep[nr_i,:lag],rho_ep[nr_i,:lag])
                                        + np.random.normal(0,sig_ep[nr_i]))

                # Roll the time series to keep the history:
                # last item (just computed) becomes the first,
                # first becomes second, etc.
                hist_ep[nr_i,:] = np.roll(hist_ep[nr_i,:],1)

        for nr_i in range(nr_t2m): # Loop over all EOFs
            # Select the lag corresponding to the partial autocorrelation function
            lag = int(lag_re_t2m[nr_i])
            for spin_it in range(lag):
                # Construct the AR(lag) process where the white noise
                # is scaled with 'sig'
                hist_t2m[nr_i,l_m_t2m-1] = (np.dot(hist_t2m[nr_i,:lag],rho_t2m[nr_i,:lag])
                                          + np.random.normal(0,sig_t2m[nr_i]))

                # Roll the time series to keep the history:
                # last item (just computed) becomes the first,
                # first becomes second, etc.
                hist_t2m[nr_i,:] = np.roll(hist_t2m[nr_i,:],1)

        # Store AR spinup data
        nc_out = nc.Dataset(self._spinup_file, 'w')
        # dims
        lag_out = nc_out.createDimension('lag_d_ep',l_m_ep)
        eof_out = nc_out.createDimension('eof_d_ep',nr_ep)
        lag_out = nc_out.createDimension('lag_d_t2m',l_m_t2m)
        eof_out = nc_out.createDimension('eof_d_t2m',nr_t2m)
        # lags (as int)
        nc_lags = nc_out.createVariable('lags_ep', 'i4', 'eof_d_ep')
        for i in range(nr_ep):
            nc_lags[i] = int(lag_re_ep[i])
        nc_lags = nc_out.createVariable('lags_t2m', 'i4', 'eof_d_t2m')
        for i in range(nr_t2m):
            nc_lags[i] = int(lag_re_t2m[i])
        # rho
        nc_rho = nc_out.createVariable('rho_ep', 'f8', ['eof_d_ep','lag_d_ep'])
        nc_rho[:,:] = rho_ep[:,:]
        nc_rho = nc_out.createVariable('rho_t2m', 'f8', ['eof_d_t2m','lag_d_t2m'])
        nc_rho[:,:] = rho_t2m[:,:]
        # sig
        nc_sigs = nc_out.createVariable('sig_ep', 'f8', 'eof_d_ep')
        nc_sigs[:] = sig_ep[:]
        nc_sigs = nc_out.createVariable('sig_t2m', 'f8', 'eof_d_t2m')
        nc_sigs[:] = sig_t2m[:]
        # Next set of random number used in POP
        nc_rnd = nc_out.createVariable('rnd_ep', 'f8', 'eof_d_ep')
        nc_rnd[:] = np.random.randn(nr_ep)
        nc_rnd = nc_out.createVariable('rnd_t2m', 'f8', 'eof_d_t2m')
        nc_rnd[:] = np.random.randn(nr_t2m)
        # Spinup history data
        hist_out = nc_out.createVariable("hist_ep", 'f8', ["eof_d_ep","lag_d_ep"])
        hist_out[:,:] = hist_ep[:,:]
        hist_out = nc_out.createVariable("hist_t2m", 'f8', ["eof_d_t2m","lag_d_t2m"])
        hist_out[:,:] = hist_t2m[:,:]
        nc_out.close()

    def generate_normal_noise(self) -> npt.NDArray[np.float64]:
        """Generate a vector of normally distrib. value.

        Returns:
            A numpy array of floats.
        """
        if self._rng is None:
            self._rng = np.random.default_rng()
        return  self._rng.standard_normal(self._nr_eofs)

    def generate_nig_noise(self) -> npt.NDArray[np.float64]:
        """Generate a vector of random values using normal inv. gaussian.

        Returns:
            A numpy array of floats.
        """
        if self._nig_params is None:
            self.load_NIG_data()

        noise = np.zeros(self._nr_eofs)
        for i in range(len(noise)):
            noise[i] = norminvgauss.rvs(*self._nig_params[:,i], size=1)

        return noise

    def generate_noise_init_file(self,
                                 noise : Any,
                                 hist_file : str) -> None:
        """Generate an NetCDF init file of PC-NIG noise.

        Args:
            noise: noise, np array of NIG drawn data
            hist_file: a file to dump the history in.
        """
        # Check that length matches at least
        if len(noise) != self._nr_eofs:
            err_msg = f"Provided init noise length: {len(noise)} does not match {self._nr_eofs}"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        # Store history data
        nc_out = nc.Dataset(hist_file, 'w')
        nc_out.createDimension('nr_eofs_ep',self._nr_eofs_ep)
        nc_out.createDimension('nr_eofs_t2m',self._nr_eofs_t2m)
        nc_noise_ep = nc_out.createVariable('rnd_noise_ep', 'f8', ['nr_eofs_ep'])
        nc_noise_ep[:] = noise[0:self._nr_eofs_ep]
        nc_noise_t2m = nc_out.createVariable('rnd_noise_t2m', 'f8', ['nr_eofs_t2m'])
        nc_noise_t2m[:] = noise[self._nr_eofs_ep:self._nr_eofs]
        nc_out.close()


class ERA5NIGForcingGenerator:
    """A class encapsulating ERA5-NIG model forcing generation.

    This class gather functionalities to generate forcing noise
    on E-P and T@2m using A. Boot analysis published in:
    https://esd.copernicus.org/articles/16/115/2025/
    In particular, the NIG model.

    Attributes:
        _data_path : the path towards NIG fit and grid data
        _month_l : the length of monthly array created
        _gen_year : the year the last batch of data was generated
        _data_loaded : a boolean flag to check if data is loaded
    """
    def __init__(self,
                 data_path : str) -> None:
        """Initialize the generator."""
        self._data_path : str = data_path
        self._data_loaded : bool = False
        self._month_l : int = 12
        self._gen_year : int = -1

    def _load_era5_data(self) -> None:
        """Load the ERA5 fit data."""

        # Meriodional and zonal grid extend
        # and grid coordinates
        self.era_lat_len : int = 721
        self.era_lon_len : int = 521
        self.era_lat = np.load(f"{self._data_path}/lat_era.npy")
        self.era_lon = np.load(f"{self._data_path}/lon_era.npy")

        # ERA5 mask
        self.era_mask = np.load(f"{self._data_path}/mask_era.npy")

        # NIG model fit from
        # https://doi.org/10.5194/egusphere-2024-2431, 2024.
        self.era_params_ep_tot = np.load(f'{self._data_path}/params_era_ep_NIG.npy') # Load in parameters
        self.era_params_t2m_tot = np.load(f'{self._data_path}/params_era_t2m_NIG.npy') # Load in parameters
        assert(self.era_params_ep_tot.shape[1] == self.era_lat_len)
        assert(self.era_params_ep_tot.shape[2] == self.era_lon_len)

    def _load_pop_grid_data(self) -> None:
        """Load the POP grid data."""
        self.lat_pop_o = np.load(f"{self._data_path}/lat_pop.npy")
        self.lon_pop_o = np.load(f"{self._data_path}/lon_pop.npy")
        self.lat_pop = self.lat_pop_o.flatten()
        self.lon_pop = self.lon_pop_o.flatten()
        self.lat_pop_xa = xr.DataArray(self.lat_pop, dims="z")
        self.lon_pop_xa = xr.DataArray(self.lon_pop, dims="z")
        for i in range(len(self.lon_pop)):
            if self.lon_pop[i] > 180:
                self.lon_pop[i] = self.lon_pop[i] - 360

    def generate_forcing_file(self,
                              filebase : str,
                              year : int) -> None:
        """Generate a new set of forcing data."""
        # Check if files already exists
        # and set internal _gen_year to year in that case.
        file_ep_out = f'{filebase}_EP.nc'
        file_t2m_out = f'{filebase}_T2M.nc'
        if (Path(file_ep_out).exists() and
            Path(file_t2m_out).exists()):
            self._gen_year = year

        # Load NIG data if not done already
        if not self._data_loaded:
            self._load_era5_data()
            self._load_pop_grid_data()
            self._data_loaded = True

        # One file per year, keep track of this
        if year == self._gen_year:
            dbg_msg = f"No need for new forcing data files for year {year} !"
            _logger.debug(dbg_msg)
            return

        # Update _gen_year
        self._gen_year = year

        noise_ep = np.zeros([self._month_l, self.era_lat_len,
                             self.era_lon_len])
        noise_t2m = np.zeros([self._month_l, self.era_lat_len,
                              self.era_lon_len])

        # First use NIG on ERA5 grid
        for lat_i in range(self.era_lat_len):
            for lon_i in range(self.era_lon_len):
                # Set noise to Nan (??) on land ERA5 mask
                if self.era_mask[lat_i,lon_i] < 1.0:
                    noise_ep[:,lat_i,lon_i] = np.nan*np.ones((self._month_l))
                    noise_t2m[:,lat_i,lon_i] = np.nan*np.ones((self._month_l))
                else:
                    params_ep = self.era_params_ep_tot[:,lat_i,lon_i]
                    params_t2m = self.era_params_t2m_tot[:,lat_i,lon_i]
                    A_ep = norminvgauss.rvs(*params_ep,size = self._month_l)
                    A_t2m = norminvgauss.rvs(*params_t2m,size = self._month_l)
                    noise_ep[:,lat_i,lon_i] = A_ep
                    noise_t2m[:,lat_i,lon_i] = A_t2m

        # Interpolate on POP grid and write NC file
        file_ep_out = f'{filebase}_EP.nc'
        noise_ep_out = nc.Dataset(file_ep_out, 'w')
        lon_d = noise_ep_out.createDimension('lon_d',self.lat_pop_o.shape[0])
        lat_d = noise_ep_out.createDimension('lat_d',self.lat_pop_o.shape[1])
        month_d = noise_ep_out.createDimension('month_d', self._month_l)
        noise_o = noise_ep_out.createVariable("noise", 'f8', ["month_d","lon_d","lat_d"])
        for m in range(self._month_l):
            interp_data = np.zeros(shape=(1, len(self.lat_pop)))
            datar = xr.DataArray(noise_ep[m,:,:], dims=("lat", "lon"),
                                 coords=dict(lat=self.era_lat,lon=self.era_lon))
            interp_res = datar.interp(lat=self.lat_pop_xa, lon=self.lon_pop_xa)
            interp_data[0,:] = interp_res.values
            interp_data[0,np.isnan(interp_data[0,:])] = 0
            noise_o[m,:,:] = np.reshape(interp_data[0,:], newshape=self.lat_pop.shape)
        noise_ep_out.close()

        file_t2m_out = f'{filebase}_T2M.nc'
        noise_t2m_out = nc.Dataset(file_t2m_out, 'w')
        lon_d = noise_t2m_out.createDimension('lon_d',self.lat_pop_o.shape[0])
        lat_d = noise_t2m_out.createDimension('lat_d',self.lat_pop_o.shape[1])
        month_d = noise_t2m_out.createDimension('month_d', self._month_l)
        noise_o = noise_t2m_out.createVariable("noise", 'f8', ["month_d","lon_d","lat_d"])
        for m in range(self._month_l):
            interp_data = np.zeros(shape=(1, len(self.lat_pop)))
            datar = xr.DataArray(noise_t2m[m,:,:], dims=("lat", "lon"),
                                 coords=dict(lat=self.era_lat,lon=self.era_lon))
            interp_res = datar.interp(lat=self.lat_pop_xa, lon=self.lon_pop_xa)
            interp_data[0,:] = interp_res.values
            interp_data[0,np.isnan(interp_data[0,:])] = 0
            noise_o[m,:,:] = np.reshape(interp_data[0,:], newshape=self.lat_pop.shape)
        noise_t2m_out.close()
