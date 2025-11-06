# urbanair_urbanwind
All notebook can work with the default polytope kernel.


# Create a green potential map
greenPotential.ipynb


# create a wind map:
create_urban_wind.ipynb

Major setttings are define under: 'etc/settings_wind_maps.json'

The user can chnage to request meteo data from:
    ECMWF -> For current meteo forecast
    On demand dt -> For selected period tailored for UrbanAIR 


# create air quality map:
Still working on it


# If you want to run more interactive notebooks you need to make a new conda env:

# Need to install env in local path in this case "conda/" otherise conda env appears to be gone after each log out
conda env create --prefix conda/ --file=urbanair_urbanwind/conda/environment.yml

# source bashrc might be required each time -> not sure why

# register conda env as a jupyter kernel
/home/jovyan/conda/urbanair/bin/python -m ipykernel install --user \
  --name urbanair \
  --display-name "Python (urbanair)"
