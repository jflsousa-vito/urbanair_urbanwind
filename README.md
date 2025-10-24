# urbanair_urbanwind


# Install conda env:
# Need to install env in local path in this case "conda/" otherise conda env appears to be gone after each log out
conda env create --prefix conda/ --file=urbanair_urbanwind/conda/environment.yml
# source bashrc might be required each time -> not sure why


# register conda env as a jupyter kernel
/home/jovyan/conda/urbanair/bin/python -m ipykernel install --user \
  --name urbanair \
  --display-name "Python (urbanair)"

# Create a green potential map

create_urban_wind_maps.ipynb


# create a wind map:

greenPotential.ipynb
