import numpy as np
import pandas as pd
from src.urban_wind import get_wind_extreme_dt, read_cfd_wind, scale_cfd_wind, save_local_wind
from src.green_pont import load_json_with_comments
from src.process_geotiff import fix_geotiff_transform


cf=load_json_with_comments('etc/settings_wind_maps.json')

import rasterio
from rasterio.transform import from_origin
xc=153300
yc=211343
#xc=211644
#yc=153300
L=100
crop_bounds = (xc-L, yc-L, xc+L, yc+L)

fix_geotiff=False

if fix_geotiff:
    angles=[0,30,60,90,120,150,180,210,240,270,300,330]

    for ag in angles:
        for cfd_height in [175]:
            tiff_test=f'/projects/urbanair/DATA/DATA_AntwerpWindStudy/wind_ratios/Wind_ratio_merge_{ag}_{cfd_height}.tiff'
            tiff_test_out=f'/projects/urbanair/DATA/DATA_AntwerpWindStudy/wind_ratios/Wind_ratio_merge_{ag}_{cfd_height}_fix.tiff'
            fix_geotiff_transform(input_path=tiff_test, output_path=tiff_test_out)



# read pre-computed normalized CFD wind ratios
path_cfd=cf['path_cfd'] 
angles=cf["angles"]
height=cf["height"]

xc=153300
yc=211644
#xc=211644
#yc=153300
L=100
crop_bounds = (xc-L, yc-L, xc+L, yc+L)

cfd_ratio=read_cfd_wind(path_cfd,angles, height,crop_bounds)


