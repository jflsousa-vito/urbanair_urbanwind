import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy import stats
#import pygrib
import warnings
#import seaborn as sns
#sns.set_theme()
#import netCDF4
import rasterio 
from rasterio.transform import xy
from rasterio.windows import from_bounds
from affine import Affine
import os
from src.green_pont import reproject_tiff
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from src.green_pont import open_raster_file, save_raster_file

warnings.filterwarnings("ignore")

def read_cfd_nox(path_cfd,angles, cfd_height, crop_bounds):

    print('Reading CFD NOx files from ', path_cfd) 
    cfd_ratio=dict()
    for ag in angles:
        tif_file=f"{path_cfd}{ag}/NOX_floor_175m_{ag}.tiff"
        """
        
        
        with rasterio.open(tif_file) as src:
            band1 = src.read(1)
            transform = src.transform
            crs = src.crs

            # ---- Flip horizontally before cropping ----
            if ag == 3000:
                print()
                band1 = np.fliplr(band1)
                width = band1.shape[1]
                transform = transform * Affine.translation(-width, 0) * Affine.scale(-1, 1)

                
            # --------------------------------------------

            if crop_bounds != []:
                # Crop using the *updated* transform
                window = from_bounds(*crop_bounds, transform=transform)
                print(window)
                band1 = band1[
                    int(window.row_off):int(window.row_off + window.height),
                    int(window.col_off):int(window.col_off + window.width)
                ]
                transform = transform * Affine.translation(window.col_off, window.row_off)
            
            res = src.res
            rows, cols = np.indices(band1.shape)
            xs, ys = xy(transform, rows, cols, offset="center")
            xs = np.asarray(xs)
            ys = np.asarray(ys)

            cfd_ratio[ag] = {
                "data": band1,
                "transform": transform,
                "crs": crs,
                "res": res,
                "xs": xs,
                "ys": ys,
            }


        
        """
        src = open_raster_file(tif_file)            
        profile = src.profile          # full metadata (useful if you’ll write a new GeoTIFF)

        if crop_bounds==[]:

            band1 = src.read(1)
            transform =src.transform
            crs = src.crs
        else:

            
            window = from_bounds(*crop_bounds, transform=src.transform)
            
            band1 = src.read(1, window=window)   # only cropped area loaded
            
            profile.update({
                "height": band1.shape[0],
                "width": band1.shape[1],
                "transform": src.window_transform(window)
            })

            
            crs = src.crs                  # coordinate reference system
            transform = src.window_transform(window)
            nodata = src.nodata
            bounds = rasterio.windows.bounds(window, transform)

        res = src.res                  # (pixel_width, pixel_height)
        rows, cols = np.indices(band1.shape)
        height, width = band1.shape
        
        xs, ys = xy(src.transform, rows, cols, offset="center")  # 2D lists/arrays of coords
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        dtype = band1.dtype          # data type of the raster values
        
        src.close()
        
        cfd_ratio[ag]=band1
        
    cfd_ratio['x']=xs
    cfd_ratio['y']=ys
    cfd_ratio['profile']=profile
    cfd_ratio['transform']=transform
    cfd_ratio['crs']=crs
    cfd_ratio['height']=height
    cfd_ratio['width']=width
    cfd_ratio['dtype']=dtype

    return cfd_ratio
            

    

def scale_cfd_nox(wind_meteo, cfd_ratio):
    print('Scaling  NOx to local urban scale using CFD ratios')
    ref_height_cfd=1.75
    ref_height_meteo=30
    z0=0.1

        
    U_30tp175=(np.log(ref_height_cfd/z0)/np.log(ref_height_meteo/z0))

    nox_local=dict()

    time_factor = pd.read_csv('time_factors_hourly_mean.csv').set_index('hour')
    print(time_factor)
    bck= 0
    
    ii=0
    for index, row in wind_meteo.iterrows():
        wind_s=row['wind_speed']
        wind_d=row['wind_dir']
        time_stamp=pd.to_datetime(index.strftime('%Y%m%d_%H%M'),format='%Y%m%d_%H%M')
        

        tf=time_factor.iloc[int(time_stamp.hour)].values

        
        angle = int(np.round(wind_d/ 30) * 30) % 360
        
        #U_175=wind_s*U_30tp175
        
        
        # coversion to NO2
        
        nox_local[time_stamp]=cfd_ratio[angle] *3.8/wind_s * tf + bck
        
        ii=ii+1

        #scaled_wind = U_ratio * ratio

    return nox_local


def save_local_aq(aq_local, cfd_ratio, path, reproject=True, mask_frames=None):
    print('Saving local wind maps to ', path)
    saved_files=[]
    
    for time_stamp, nox_local in aq_local.items():
        output_file=f"{path}/NOx_175_{time_stamp}.tif"
        meta = {
            "driver": "GTiff",
            "height": cfd_ratio["height"],
            "width": cfd_ratio["width"],
            "count": 1,
            "dtype": cfd_ratio["dtype"],
            "crs": cfd_ratio["crs"],
            "transform": cfd_ratio["transform"]
        }
        save_raster_file(output_file, nox_local, meta, indexes=1)
        
        saved_files.append(output_file)

        if reproject:
            reproject_tiff(
                output_file, 
                output_file[:-1*len(".tif")] + '_4326.tif', 
                dst_crs ="EPSG:4326",
                mask=False,
            )
            
    return saved_files