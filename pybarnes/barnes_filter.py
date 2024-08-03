from warnings import warn
from psutil import virtual_memory
import xarray as xr
import numpy as np
from tqdm import trange, tqdm
from metpy.calc import lat_lon_grid_deltas

class BarnesFilter2d:

    """
    The Barnes method performs grid point interpolation by selecting appropriate
    filtering parameters *c* and *g* to filter out shortwave noise in the original field,
    making the analysis results stable and smooth. In addition, it can form a bandpass filter
    to separate various sub weather scales that affect weather processes according to actual needs,
    achieving the purpose of scale separation.

    Reference:
        DOI : https://doi.org/10.1175/1520-0493(1980)108<1108:AOTFSM>2.0.CO;2

    Instructions:
    Considering the efficiency of computation and memory usage,
    it is strongly recommended to merge the variables that require filtering and calculate them together,
    and save the results in a timely manner

    For example we have u and v data with 3 levels, whose spatial shape is (61, 71)

    >>> print(u.shape, v.shape)
    (3, 61, 71), (3, 61, 71)
    >>> data = np.stack([u, v], axis=0)
    >>> data.shape
    (2, 3, 61, 71)
    >>> f = BarnesFilter2d(data, lon, lat, radius_degree=10)
    >>> band_data = f.bandpass(g1=0.3, c1=30000, g2=0.3, c2=150000)
    >>> np.save("band_data.npy", band_data)
    """


    def __init__(self, data_arr, lon=None, lat=None, radius_degree=8, **kwargs):
        """
        Initializing the data and caculate the distance.

        Parameters
        ----------
        data_arr : numpy.array (recommended) or xarray.DataArray (not recommended)
            An N-dimensional array which to be filtered.
            Don't support for the wrfout xarray data, please transform it to a numpy.array to this function.

        lon : array
            If the data_arr are numpy.array which has no longitude and latitude infomation,
            then the longitude infomation must be specified.

        lat : array
            If the data_arr are numpy.array which has no longitude and latitude infomation,
            then the latitude infomation must be specified.
            
        radius_degree : int or tuple
            The radius of each point when caculating the distance of each other.
            Units : degree.

            It is recommended to set this with your schemes. 
            For the constant *c*, this parameter is recommended to be:

            for the *c* is [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
            *radius_degree* is recommended for [1, 1.5, 2, 3, 4, 5, 7, 8, 12]


        **kwargs:
        
        dtype : The input data's type (default np.float32)

        check_mask : bool (default True)
            If the data contains NaN, the weights of the NaN grids should be masked.
            This may require additional computing resources.

        max_memory_use_rate : float  (default 0.7)
            In order to accelerate the calculation, this function will use the free memory as much as possible.

        Returns
        -------
        out : object
            A Barnes filter object.
        """
        self.max_memory_use_rate = kwargs.get("max_memory_use_rate", 0.7)
        self.check_mask = kwargs.get("check_mask", True)
        self.dtype = kwargs.get("dtype", np.float32)
            
        if isinstance(data_arr, xr.DataArray):
            self.dims = data_arr.dims
            self.coords = data_arr.coords
            lat_var, lon_var = self.dims[-2:]
            lon, lat = data_arr[lon_var].data, data_arr[lat_var].data
            self.data = data_arr.data.astype(self.dtype)
            self.data_name = data_arr.name
        else:
            if (lat is None) or (lon is None):
                raise KeyError("The longitude or latitude of the data are missing.")
            self.data = data_arr.astype(self.dtype)

        if not np.iterable(radius_degree):
            radius_degree = [radius_degree] * 2
        radius_degree = np.array(radius_degree)
        if (lon.ndim < 2) & (lat.ndim < 2):
            lon, lat = np.meshgrid(lon, lat)
        self.ny, self.nx = lon.shape
        dx, dy = lat_lon_grid_deltas(lon, lat)
        self.xcoord = np.array(np.concatenate([np.zeros(self.ny,)[:, None], np.cumsum(dx, 1)], 1), dtype=self.dtype)/1000
        self.ycoord = np.array(np.concatenate([np.zeros(self.nx,)[None], np.cumsum(dy, 0)], 0, dtype=self.dtype))/1000
        del dx, dy

        self.whether_data_contains_nan = False
        self.sum = np.sum
        if self.check_mask :
            data0 = self.data.reshape(-1, self.ny, self.nx)
            mask_data = np.isnan(data0).any(axis=0)
            self.whether_data_contains_nan = mask_data.any()
            if self.whether_data_contains_nan:
                self.mask_data = np.where(mask_data, np.nan, 1).astype(np.float16)
                warn("The input data contains NaNs which may result in unexpected result.\nMaybe reshape the data into 1-dimesino and deop the NaN data will help.")
                self.sum = np.nansum

        mean_delta_lon = np.abs(np.mean(lon[:, :-1] - lon[:, 1:]))
        mean_delta_lat = np.abs(np.mean(lat[:-1] - lat[1:]))
        radius_grid = (radius_degree//np.array([mean_delta_lon, mean_delta_lat])).astype(int)
        grids = (radius_grid * 2 + 1)[::-1].tolist()
        self.grids = [min([self.ny, self.nx][i], grids[i]) for i in range(2)]

        self.x_window_start_points = np.clip(np.arange(self.nx) - (self.grids[-1] // 2), 0, self.nx - self.grids[-1]).astype(np.int32)
        self.y_window_start_points = np.clip(np.arange(self.ny) - (self.grids[-2] // 2), 0, self.ny - self.grids[-2]).astype(np.int32)
        avail_memory_mb = virtual_memory().available/(1024 ** 2) - np.prod(self.data.shape) * 8/(1024 ** 2)
        memory4use_mb = avail_memory_mb * self.max_memory_use_rate / (4 if not self.whether_data_contains_nan else 8)
        base_shape = tuple(self.grids) + self.data.shape[:-2]
        xcoord_batch_size = int(memory4use_mb//(np.prod(base_shape) * 4/(1024 ** 2)))
        self.xcoord_batch_size = max(min(xcoord_batch_size, self.nx), 1)
        self.xcoord_epochs = int(np.ceil(self.nx/self.xcoord_batch_size))
        if self.xcoord_epochs < 2:
            ycoord_batch_size = int(memory4use_mb//(np.prod(base_shape + (self.xcoord_batch_size, )) * 4/(1024 ** 2)))
            self.ycoord_batch_size = max(min(ycoord_batch_size, self.ny), 1) + 1
            self.ycoord_epochs = int(np.ceil(self.ny/self.ycoord_batch_size))
        else:
            self.ycoord_batch_size, self.ycoord_epochs = 1, self.ny

    def __split_filed(self, data=None):
        assert (data is None) or (data.shape[-2:] == (self.ny, self.nx))
        for i in range(self.ycoord_epochs):
            slicey = slice(i * self.ycoord_batch_size, (i + 1) * self.ycoord_batch_size)
            y_window_slice = self.y_window_start_points[slicey, None, None, None] + np.arange(self.grids[-2], dtype=np.int32)[:, None]
            for j in range(self.xcoord_epochs):
                slicex = slice(j * self.xcoord_batch_size, (j + 1) * self.xcoord_batch_size)
                x_window_slice = self.x_window_start_points[slicex, None, None] + np.arange(self.grids[-1], dtype=np.int32)
                if data is None:
                    dx2 = (self.xcoord[slicey, slicex, None, None] - self.xcoord[y_window_slice, x_window_slice]) ** 2
                    dy2 = (self.ycoord[slicey, slicex, None, None] - self.ycoord[y_window_slice, x_window_slice]) ** 2
                    kilometer_distance2 = dx2 + dy2
                    del dx2, dy2
                    if self.whether_data_contains_nan : 
                        kilometer_distance2 = kilometer_distance2 * self.mask_data[y_window_slice, x_window_slice]
                    yield kilometer_distance2
                else:
                    yield data[..., y_window_slice, x_window_slice]

    def lowpass(self, g=0.3, c=150000, print_progress=True):
        """
        Selecting different parameters *g* and *c*
        will result in different filtering characteristics.

        Reference:
        DOI : https://doi.org/10.1175/1520-0493(1980)108<1108:AOTFSM>2.0.CO;2

        Parameters
        ----------
        g : float, generally between (0, 1]
            Constant parameter.
        c : int
            Constant parameter. When *c* takes a larger value, the filter function converges
            at a larger wavelength, and the response function slowly approaches the maximum value,
            which means that high-frequency fluctuations have been filtered out.
        print_progress : bool
            Whether to print the progress bar when executing computation.

        Returns
        -------
        data_vars : array
            Data field after filtering out high-frequency fluctuations

        """
        meta = self.__split_filed()
        meta_data = self.__split_filed(self.data)
        revised_data = np.zeros_like(self.data, dtype=self.dtype)
        yiter = range if ((self.ycoord_epochs == 1) or not print_progress) else trange
        print("Caculating the first revision...") if print_progress else None
        for i in yiter(self.ycoord_epochs):
            slicey = slice(i * self.ycoord_batch_size, (i + 1) * self.ycoord_batch_size)
            for j in range(self.xcoord_epochs):
                slicex = slice(j * self.xcoord_batch_size, (j + 1) * self.xcoord_batch_size)
                kilometer_distance2, data_part = next(meta), next(meta_data)
                normed_weights = self.__caculate_normed_weights(kilometer_distance2, c)
                revised_data[..., slicey, slicex] = self.sum(data_part * normed_weights, (-1, -2))
                del normed_weights, kilometer_distance2, data_part

        meta = self.__split_filed()
        meta_data = self.__split_filed(self.data - revised_data)
        print("Caculating the second revision...") if print_progress else None
        for i in yiter(self.ycoord_epochs):
            slicey = slice(i * self.ycoord_batch_size, (i + 1) * self.ycoord_batch_size)
            for j in range(self.xcoord_epochs):
                slicex = slice(j * self.xcoord_batch_size, (j + 1) * self.xcoord_batch_size)
                kilometer_distance2, diff = next(meta), next(meta_data)
                normed_weights = self.__caculate_normed_weights(kilometer_distance2, g * c)
                weighted_diff = self.sum(diff * normed_weights, (-1, -2))
                del normed_weights, kilometer_distance2, diff
                revised_data[..., slicey, slicex] = revised_data[..., slicey, slicex] + weighted_diff
        return self.__convert_data(revised_data)

    def bandpass(self, g1=0.3, c1=30000, g2=0.3, c2=150000, r=1.2, print_progress=True):
        """
        Select two different filtering schemes 1 and 2, and perform the filtering separately.
        And then perform the difference, that means *scheme1 - scheme2*.
        The mesoscale fluctuations are thus preserved.

        Parameters
        ----------
        g1 : float, generally between (0, 1]
            Constant parameter of scheme1.
        c1 : int
            Constant parameterof scheme1.
        g2 : float, generally between (0, 1]
            Constant parameter of scheme2.
        c2 : int
            Constant parameterof scheme2.
        r :  float
            The inverse of the maximum response differenc.
            It is prevented from being unduly large and very small difference fields are not greatly amplified.
        print_progress : bool
            Whether to print the progress bar when executing computation.

        Returns
        -------
        data_vars : array
            Mesoscale wave field filtered out from raw data
        """
        meta = self.__split_filed()
        meta_data = self.__split_filed(self.data)
        revised_data = np.zeros((2, ) + self.data.shape, dtype=self.dtype)
        yiter = range if ((self.ycoord_epochs == 1) or not print_progress) else trange
        print("Caculating the first twice revision...") if print_progress else None
        for i in yiter(self.ycoord_epochs):
            slicey = slice(i * self.ycoord_batch_size, (i + 1) * self.ycoord_batch_size)
            for j in range(self.xcoord_epochs):
                slicex = slice(j * self.xcoord_batch_size, (j + 1) * self.xcoord_batch_size)
                kilometer_distance2, data_part = next(meta), next(meta_data)
                normed_weights = self.__caculate_normed_weights(kilometer_distance2, c1)
                revised_data[0, ..., slicey, slicex] = self.sum(data_part * normed_weights, (-1, -2))
                if c1 != c2:
                    normed_weights = self.__caculate_normed_weights(kilometer_distance2, c2)
                    revised_data[1, ..., slicey, slicex] = self.sum(data_part * normed_weights, (-1, -2))
        if c1 == c2:
            revised_data[1] = revised_data[0]

        meta = self.__split_filed()
        meta_data = self.__split_filed(self.data - revised_data)
        print("Caculating the second twice revision...") if print_progress else None
        for i in yiter(self.ycoord_epochs):
            slicey = slice(i * self.ycoord_batch_size, (i + 1) * self.ycoord_batch_size)
            for j in range(self.xcoord_epochs):
                slicex = slice(j * self.xcoord_batch_size, (j + 1) * self.xcoord_batch_size)
                kilometer_distance2, diff = next(meta), next(meta_data)
                normed_weights = self.__caculate_normed_weights(kilometer_distance2, g1 * c1)
                weighted_diff = self.sum(diff[0] * normed_weights, (-1, -2))
                revised_data[0, ..., slicey, slicex] = revised_data[0, ..., slicey, slicex] + weighted_diff
                normed_weights = self.__caculate_normed_weights(kilometer_distance2, g2 * c2)
                weighted_diff = self.sum(diff[1] * normed_weights, (-1, -2))
                revised_data[1, ..., slicey, slicex] = revised_data[1, ..., slicey, slicex] + weighted_diff
                del normed_weights, kilometer_distance2, diff
        return self.__convert_data(r * (revised_data[0] - revised_data[1]))
    
    def __caculate_normed_weights(self, distance, width):
        weights = np.exp(-distance/(4 * width))
        sum_weights = self.sum(weights, (-1, -2), keepdims=True)
        return weights/sum_weights

    def __convert_data(self, data):
        if hasattr(self, "dims"):
            data = xr.DataArray(data, coords=self.coords, dims=self.dims, name=self.data_name)
        return data


class BarnesFilter1d:

    """
    The Barnes method performs grid point interpolation by selecting appropriate
    filtering parameters *c* and *g* to filter out shortwave noise in the original field,
    making the analysis results stable and smooth. In addition, it can form a bandpass filter
    to separate various sub weather scales that affect weather processes according to actual needs,
    achieving the purpose of scale separation.

    Reference:
        DOI : https://doi.org/10.1175/1520-0493(1980)108<1108:AOTFSM>2.0.CO;2

    Instructions:
    Considering the efficiency of computation and memory usage,
    it is strongly recommended to merge the variables that require filtering and calculate them together,
    and save the results in a timely manner

    For example we have u and v data with 3 levels, whose spatial shape is (61)

    >>> print(u.shape, v.shape)
    (3, 61), (3, 61)
    >>> data = np.stack([u, v], axis=0)
    >>> data.shape
    (2, 3, 61)
    >>> f = BarnesFilter1d(lon, lat, data, radius_degree=10)
    >>> band_data = f.bandpass(g1=0.3, c1=30000, g2=0.3, c2=150000)
    >>> lon, lat = f.get_lon_lat()
    >>> np.savez("band_data.npz", **dict(band_data=band_data, lon=lon, lat=lat))
    """

    def __init__(self, lon, lat, data_arr, radius_degree=8, **kwargs):
        """
        Initializing the data and caculate the distance.

        Parameters
        ----------
        lon : array
            If the data_arr are numpy.array which has no longitude and latitude infomation,
            then the longitude infomation must be specified.

        lat : array
            If the data_arr are numpy.array which has no longitude and latitude infomation,
            then the latitude infomation must be specified.

        data_arr : numpy.array (recommended) or xarray.DataArray (not recommended)
            An N-dimensional array which to be filtered.
            Don't support for the wrfout xarray data, please transform it to a numpy.array to this function.

        radius_degree : int or tuple
            The radius of each point when caculating the distance of each other.
            Units : degree.

            It is recommended to set this with your schemes.
            For the constant *c*, this parameter is recommended to be:

            for the *c* is [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
            *radius_degree* is recommended for [1, 1.5, 2, 3, 4, 5, 7, 8, 12]


        **kwargs:
        neighbor_dots : int (default : the number of the points within the *radius_degree* for the center point)
            Only caculate this number of neighbor points for each points.

        center_lat : int of float (default : the mean latitude of the input data)
            To use the radius of rhe Earth to caculate the distance between two points,
            it needs to refer to a standard latitude.

        dtype : The input data's type (default np.float32)

        check_mask : bool (default True)
            If the data contains NaN, the weights of the NaN grids should be masked.
            This may require additional computing resources.

        max_memory_use_rate : float  (default 0.7)
            In order to accelerate the calculation, this function will use the free memory as much as possible.

        Returns
        -------
        out : object
            A Barnes filter object.
        """
        assert lon.ndim == lat.ndim == 1
        assert data_arr.shape[-1] == len(lon) == len(lat)
        self.max_memory_use_rate = kwargs.get("max_memory_use_rate", 0.8)
        self.dtype = kwargs.get("dtype", np.float32)
        center_lat = kwargs.get("center_lat", None)
        neighbor_dots = kwargs.get("neighbor_dots", None)
        idx = np.isnan(data_arr) | np.isnan(lon) | np.isnan(lat)
        if idx.sum() > 0:
            data_arr = data_arr[~idx]
            lon, lat = lon[~idx], lat[~idx]
        ind = np.argsort(lat)
        self.data = data_arr[..., ind].astype(self.dtype)
        self.lon, self.lat = lon[ind].astype(self.dtype), lat[ind].astype(self.dtype)
        mean_lat_delta = np.mean(self.lat[1:] - self.lat[:-1])
        self.lat_neighbors = int(radius_degree//mean_lat_delta)
        self.ndots = data_arr.shape[-1]

        lat0 = np.mean(self.lat) if center_lat is None else center_lat
        lon0 = np.mean(self.lon)
        self.factor = 6370 * np.cos(np.deg2rad(lat0)) * np.pi/180
        if neighbor_dots is None:
            deg2center = np.sqrt((self.lon - lon0) ** 2 + (self.lat - lat0) ** 2)
            self.dots = max((deg2center <= radius_degree).sum(), 1)
        else:
            self.dots = neighbor_dots
        self.dots = min(self.dots, self.ndots)
        self.lat_neighbors = max(self.dots, self.lat_neighbors)

        avail_memory_mb = virtual_memory().available/(1024 ** 2) - np.prod(self.data.shape) * 8/(1024 **2)
        memory4use_mb = avail_memory_mb * self.max_memory_use_rate * 0.24
        base_shape = self.data.shape[:-1] + (self.dots, )
        batch_size = int(memory4use_mb//(np.prod(base_shape) * 4/(1024 **2)))
        tmp = int(memory4use_mb//(self.lat_neighbors * 12/(1024 **2)))
        batch_size = min(tmp, batch_size)
        self.batch_size = max(min(batch_size, self.ndots), 1)
        self.epochs = int(np.ceil(self.ndots/self.batch_size))

    def __split_filed(self, data=None):
        assert (data is None) or (data.shape[-1] == self.ndots)
        for i in range(self.epochs):
            sli = slice(i * self.batch_size, (i + 1) * self.batch_size)
            seq = np.arange(self.ndots)
            start_ind = np.clip(seq - self.lat_neighbors//2, 0, self.ndots - self.lat_neighbors).astype(int)
            window = start_ind[sli, None] + np.arange(self.lat_neighbors, dtype=int)
            dlon, dlat = self.__get_lon_distance(self.lon[sli, None], self.lon[window]), self.lat[sli, None] - self.lat[window]
            k2 = (dlon ** 2 + dlat ** 2) * self.factor ** 2
            del dlon, dlat
            idx = np.argsort(k2, axis=-1)[:, :self.dots].astype(np.int32)
            tmp = np.arange(len(k2))[:, None]
            kilometer_distance2 = k2[tmp, idx]
            if data is None:
                yield kilometer_distance2
            else:
                yield kilometer_distance2, data[..., window[tmp, idx]]
    
    def lowpass(self, g=0.3, c=150000, print_progress=True):
        """
        Selecting different parameters *g* and *c*
        will result in different filtering characteristics.

        Reference:
        DOI : https://doi.org/10.1175/1520-0493(1980)108<1108:AOTFSM>2.0.CO;2

        Parameters
        ----------
        g : float, generally between (0, 1]
            Constant parameter.
        c : int
            Constant parameter. When *c* takes a larger value, the filter function converges
            at a larger wavelength, and the response function slowly approaches the maximum value,
            which means that high-frequency fluctuations have been filtered out.
        print_progress : bool
            Whether to print the progress bar when executing computation.

        Returns
        -------
        data_vars : array
            Data field after filtering out high-frequency fluctuations

        """
        meta = self.__split_filed(self.data)
        revised_data = np.zeros_like(self.data, dtype=self.dtype)
        data_iter = range if ((self.epochs == 1) or not print_progress) else trange
        print("Caculating the first revision...") if print_progress else None
        for i in data_iter(self.epochs):
            sli = slice(i * self.batch_size, (i + 1) * self.batch_size)
            kilometer_distance2, data_part = next(meta)
            normed_weights = self.__caculate_normed_weights(kilometer_distance2, c)
            revised_data[..., sli] = np.sum(data_part * normed_weights, -1)
            del normed_weights, kilometer_distance2, data_part

        meta = self.__split_filed(self.data - revised_data)
        print("Caculating the second revision...") if print_progress else None
        for i in data_iter(self.epochs):
            sli = slice(i * self.batch_size, (i + 1) * self.batch_size)
            kilometer_distance2, diff = next(meta)
            normed_weights = self.__caculate_normed_weights(kilometer_distance2, g * c)
            weighted_diff = np.sum(diff * normed_weights, -1)
            del normed_weights, kilometer_distance2, diff
            revised_data[..., sli] = revised_data[..., sli] + weighted_diff
        return revised_data
    
    def bandpass(self, g1=0.3, c1=30000, g2=0.3, c2=150000, r=1.2, print_progress=True):
        """
        Select two different filtering schemes 1 and 2, and perform the filtering separately.
        And then perform the difference, that means *scheme1 - scheme2*.
        The mesoscale fluctuations are thus preserved.

        Parameters
        ----------
        g1 : float, generally between (0, 1]
            Constant parameter of scheme1.
        c1 : int
            Constant parameterof scheme1.
        g2 : float, generally between (0, 1]
            Constant parameter of scheme2.
        c2 : int
            Constant parameterof scheme2.
        r :  float
            The inverse of the maximum response differenc.
            It is prevented from being unduly large and very small difference fields are not greatly amplified.
        print_progress : bool
            Whether to print the progress bar when executing computation.

        Returns
        -------
        data_vars : array
            Mesoscale wave field filtered out from raw data
        """
        meta = self.__split_filed(self.data)
        revised_data = np.zeros((2, ) + self.data.shape, dtype=self.dtype)
        data_iter = range if ((self.epochs == 1) or not print_progress) else trange
        print("Caculating the first twice revision...") if print_progress else None
        for i in data_iter(self.epochs):
            sli = slice(i * self.batch_size, (i + 1) * self.batch_size)
            kilometer_distance2, data_part = next(meta)
            normed_weights = self.__caculate_normed_weights(kilometer_distance2, c1)
            revised_data[0, ..., sli] = np.sum(data_part * normed_weights, -1)
            if c1 != c2:
                normed_weights = self.__caculate_normed_weights(kilometer_distance2, c2)
                revised_data[1, ..., sli] = np.sum(data_part * normed_weights, -1)
        if c1 == c2:
            revised_data[1] = revised_data[0]

        meta = self.__split_filed(self.data - revised_data)
        print("Caculating the second twice revision...") if print_progress else None
        for i in data_iter(self.epochs):
            sli = slice(i * self.batch_size, (i + 1) * self.batch_size)
            kilometer_distance2, diff = next(meta)
            normed_weights = self.__caculate_normed_weights(kilometer_distance2, g1 * c1)
            weighted_diff = np.sum(diff[0] * normed_weights, -1)
            revised_data[0, ..., sli] = revised_data[0, ..., sli] + weighted_diff
            normed_weights = self.__caculate_normed_weights(kilometer_distance2, g2 * c2)
            weighted_diff = np.sum(diff[1] * normed_weights, -1)
            revised_data[1, ..., sli] = revised_data[1, ..., sli] + weighted_diff
            del normed_weights, kilometer_distance2, diff
        return r * (revised_data[0] - revised_data[1])

    def get_lon_lat(self):
        """
        Get the longtitude and latitude of the outout data.
        """
        return self.lon, self.lat

    def __caculate_normed_weights(self, distance, width):
        weights = np.exp(-distance/(4 * width))
        sum_weights = np.sum(weights, -1, keepdims=True)
        return weights/sum_weights

    @staticmethod
    def __get_lon_distance(x, y):
        dlon1 = np.abs(x - y)
        dlon2 = np.abs(x - y - 360)
        return np.min(np.stack((dlon1, dlon2), axis=0), axis=0)

if __name__ == "__main__":
    ngrid = 200
    d = np.random.uniform(-10, 10, (ngrid, ))
    x, y = np.random.uniform(0, 90, (2, ngrid, ))
    f = BarnesFilter1d(x, y, d)
    a = f.lowpass()
    import matplotlib.pyplot as plt
    plt.figure(1, (14, 4.8))
    plt.subplot(131)
    plt.colorbar(plt.tricontourf(x, y, d, cmap=plt.cm.RdYlBu_r))
    plt.subplot(132)
    plt.colorbar(plt.tricontourf(x, y, a.data, cmap=plt.cm.RdYlBu_r))
    plt.subplot(133)
    plt.colorbar(plt.tricontourf(x, y, d - a.data, cmap=plt.cm.RdYlBu_r))
    plt.show()

