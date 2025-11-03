
import warnings
import os
from src.green_pont import load_json_with_comments
from src.urban_wind import wind_speed_grib, read_cfd_nox, save_local_aq, plot_maps, scale_cfd_nox


warnings.filterwarnings("ignore")



def main():
    
    """ Create urban wind maps at 175m height based on meso-scale model data and CFD pre-computed ratios
            1. Read meso-scale model data from GRIB files (e.g. from ECMWF or other sources)
            2. Read pre-computed CFD wind ratios for different wind directions
            3. Scale meso-scale wind data to local urban scale using CFD ratios
            4. Save the resulting urban wind maps as GeoTIFF files
            5. Plot the urban wind maps and save as PNG files
            
        usage: python create_urban_wind_maps.py
    """
    cf=load_json_with_comments('etc/settings_air_quality.json')
    
    path_cfd=cf['path_cfd'] #'/projects/urbanair/DATA/DATA_AntwerpWindStudy/wind_ratios'
    angles=cf["angles"]
    height=cf["height"]
    path = cf['path_meteo'] #'data/meteo_input/'
    grib_file= cf['grib_file'] # 'GRIBPFDEOD+$hour$h00m00s'
    output_path=cf["output_path"]
    
    # read meteo file from meso-scale model
    wind_meteo = wind_speed_grib(30, path, grib_file)

    # read pre-computed normalized CFD wind ratios
    
        
    xc=153300
    yc=211644
    L=1000


    crop_bounds = (xc-L, yc-L, xc+L, yc+L)
    cfd_ratio=read_cfd_nox(path_cfd,angles, height, crop_bounds)

    print(wind_meteo)
   
    # scale meso-scale wind to local urban scale using CFD ratios
    nox_local=scale_cfd_nox(wind_meteo, cfd_ratio)
    
    os.makedirs(output_path, exist_ok=True)
    
    print(nox_local)
    
    
    saved_files=save_local_aq(nox_local, cfd_ratio, output_path)
    
    #plot_maps(saved_files, path=output_path)

    
    

if __name__ == "__main__":
    main()