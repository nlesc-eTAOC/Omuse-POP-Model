import xarray as xr  
import numpy as np
import random
import netCDF4 as nc
from pathlib import Path
from scipy.stats import norminvgauss

class ERA5ForcingGenerator:
    """A class encapsulating ERA-Data model forcing generation.

    This class gather functionalities to generate forcing noise
    on E-P and T@2m using A. Boot analysis published in:
    https://esd.copernicus.org/articles/16/115/2025/
    In particular, the NIG model is available.

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
            print("No need for new forcing data files !", flush = True)
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
