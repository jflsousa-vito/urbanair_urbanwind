import numpy as np
import pandas as pd
from src.urban_wind import get_wind_extreme_dt, read_cfd_wind, scale_cfd_wind, save_local_wind
from src.green_pont import load_json_with_comments

cf=load_json_with_comments('etc/settings_wind_maps.json')

# read pre-computed normalized CFD wind ratios
path_cfd=cf['path_cfd'] 
angles=cf["angles"]
height=cf["height"]

xc=153300
yc=201644
#xc=211644
#yc=153300
L=10
crop_bounds = (xc-L, yc-L, xc+L, yc+L)

cfd_ratio=read_cfd_wind(path_cfd,angles, height,crop_bounds)


