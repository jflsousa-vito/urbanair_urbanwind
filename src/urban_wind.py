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



warnings.filterwarnings("ignore")

    

def wind_speed_grib(height, path,grib_pattern):
    
    print('Reading GRIB files from ', path)
    
    target_lat = 51.1954 # Antwerp
    target_lon = 4.3615  

    hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '11', '13', '14', '15', '17', '18', '23', '24']
    wind_s = []
    wind_d = []
    time_stamp=[]
    idx=None
    for i in hours:
        grib_file=path+grib_pattern.replace('$hour$', str(i).zfill(4))
       
        grib = xr.open_dataset(grib_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround'})
        time_stamp.append(pd.to_datetime(grib['time'].values)+pd.Timedelta(hours=int(i)))
        
        if idx==None:
            latitude = grib['v'].latitude.values
            longitude = grib['v'].longitude.values

            dist = np.sqrt((latitude - target_lat)**2 + (longitude - target_lon)**2)
            idx = dist.argmin()
        
        U = grib['u'].sel(heightAboveGround=height)
        V = grib['v'].sel(heightAboveGround=height)
        wind_speed = np.sqrt(U**2 + V**2)
        wind_dir = (np.degrees(np.arctan2(U, V)) + 180 ) % 360

        w_speed = wind_speed.values[idx]
        w_dir = wind_dir.values[idx]
        wind_s = np.append(wind_s, w_speed)
        wind_d = np.append(wind_d, w_dir)
        grib.close()
    df = pd.DataFrame({'wind_speed': wind_s, 'wind_dir': wind_d}, index=time_stamp)
    
    return df

def get_wind_extreme_dt(LOCATION,date='-14'):
    import earthkit.plots
    from polytope.api import Client
    import numpy as np
    import pandas as pd

    LIVE_REQUEST = True

    request = {
        "dataset": "extremes-dt",
        "class": "d1",
        "stream": "oper",
        "type": "fc",
        "date": date,
        "time": "0000",
        "levtype": "sfc",
        "expver": "0001",
        "param": "166/165",  # <-- V component of 10 m wind
        "feature": {
            "type": "timeseries",
            "points": [[LOCATION[0], LOCATION[1]]],
            "axes": "step",
            "range": {"start": 0, "end": 24}
        },
    }

    data_file = "data/extremes-dt-earthkit-example-fe-timeseries.covjson"
    if LIVE_REQUEST:
        data = earthkit.data.from_source("polytope", "destination-earth", request, address="polytope.lumi.apps.dte.destination-earth.eu", stream=False)
        data.to_target("file", data_file)
    else:
        data = earthkit.data.from_source("file", data_file) 

    da = data.to_xarray()

    U = da['10u']
    V = da['10v']
    wind_speed_10 = np.sqrt(U**2 + V**2)
    wind_dir = (np.degrees(np.arctan2(U, V)) + 180 ) % 360

    z0 = 0.03  # grassland, change for urban/forest
    wind_speed_30 = wind_speed_10 * np.log(30/z0) / np.log(10/z0)

    wind_speed_10=np.array(wind_speed_10).flatten().tolist()
    wind_speed_30=np.array(wind_speed_30).flatten().tolist()
    wind_dir=np.array(wind_dir).flatten().tolist()
    time_stamp= da.t.values
    
    df = pd.DataFrame({'wind_speed': wind_speed_30, 'wind_dir': wind_dir}, index=time_stamp)
            
    return df
    
def read_cfd_wind(path_cfd,angles, cfd_height, crop_bounds):



    print('Reading CFD wind files from ', path_cfd) 
    cfd_ratio=dict()
    for ag in angles:
        tif_file=f'{path_cfd}/Wind_ratio_merge_{ag}_{cfd_height}_fix.tiff'

        
        with rasterio.open(tif_file) as src:
            
            profile = src.profile          # full metadata (useful if you’ll write a new GeoTIFF)
            

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
            


def scale_cfd_wind(wind_meteo, cfd_ratio):
    print('Scaling meso-scale wind to local urban scale using CFD ratios')
    ref_height_cfd=1.75
    ref_height_meteo=30
    z0=0.1

        
    U_30tp175=(np.log(ref_height_cfd/z0)/np.log(ref_height_meteo/z0))

    wind_local=dict()

    for index, row in wind_meteo.iterrows():
        wind_s=row['wind_speed']
        wind_d=row['wind_dir']
        time_stamp=index
        angle = int(np.round(wind_d/ 30) * 30) % 360
        U_175=wind_s*U_30tp175
        U_local = cfd_ratio[angle] *U_175
        
        wind_local[time_stamp]=U_local

        #scaled_wind = U_ratio * ratio

    return wind_local
    
def save_local_wind(wind_local, cfd_ratio, path, reproject=True, mask_frames=None):
    print('Saving local wind maps to ', path)
    saved_files=[]
    
    for time_stamp, U_local in wind_local.items():
        output_file=path+'/wind_175_'+time_stamp.strftime('%Y%m%d_%H%M')+'.tif'
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=cfd_ratio['height'],
            width=cfd_ratio['width'],
            count=1,
            dtype=cfd_ratio['dtype'],
            crs=cfd_ratio['crs'],
            transform=cfd_ratio['transform'],
        ) as dst:
            dst.write(U_local, 1)    
            #dst.write_mask(mask.astype(np.uint8) * 255)
        saved_files.append(output_file)

        
        
        if reproject:
            
            
            
            reproject_tiff(output_file, output_file.split('.')[0] + '_4326.tif', dst_crs ="EPSG:4326", mask_frames=mask_frames)

    return saved_files

def plot_maps(saved_files, path):
    
    import rasterio as rio
    from rasterio.plot import show
    import matplotlib.pyplot as plt

    for file in saved_files:

        with rio.open(file) as src:
            arr = src.read(1, masked=True)          # Band 1, honor NoData
            left, top, right, bottom = src.bounds
            extent = (left, right, bottom, top)

        print(bottom, top, left, right)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        im = ax.imshow(arr, extent=extent, origin="upper", vmin=0, vmax=4,cmap = plt.cm.get_cmap("jet"))  # geospatial extent
        title=file.split('_')[3] + file.split('_')[4]
        ax.set_title(title)
        ax.invert_yaxis()
        plt.axis('off')
        #ax.set_xlabel(f"X ({src.crs.to_string()})")
        #ax.set_ylabel(f"Y ({src.crs.to_string()})")
        ax.set_aspect("equal")  # square pixels
        fig.colorbar(im, ax=ax)
        ff=file.split('/')[-1].replace('.tif', '.png')
        fig.savefig(f'{path}/{ff}')
                
        
    
        """
        with rio.open(file) as src:
            fig, ax = plt.subplots()
            show(src.read(1, masked=True), transform=src.transform, ax=ax)
            ax.set_title("Band 1")
            ax.set_aspect("equal")

        ff=file.split('/')[-1].replace('.tif', '.png')
        fig.savefig(f'{path}/{ff}')
        """

    print("Creating animation...")
    cmd = "convert -delay 40 -loop 0 *.png animation.gif"
    os.chdir(path)
    os.system(cmd)
    
        
    