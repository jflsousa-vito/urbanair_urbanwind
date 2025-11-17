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

To use the "On demand dt", the Polytope API key for the `jovyan` user must be created.
This can be done by running the program in `/home/jovyan/polytope-lab/desp-authentication.py`
This requires your DestinationEarth username and password.

# create air quality map:
Still working on it

# Reading files from S3
To read files from S3, two changes are needed

1.  A file called "access_keys.json" in the `/home/jovyan` directory is required.  This has the format:
```
{
  "accessKey":"<access key redacted>",
  "secretKey":"<secret key redacted>"
}
```
Fill in the `access key` and `secret key` that you got from the MyDataLake service.

2. File paths in `/etc/settings_wind_maps.json`, `settings_air_quality.json` and `settings_greenPotential.json` can be changed to read files from S3.
For example:
```
"path_cfd" : "s3.central.data.destination-earth.eu/",
```
If the string `"s3."` is contained within a file path in the `/etc/settings_*` files, then the notebook will attempt to read the file from s3, using the
URL provided in the path, and the accessKey and secretKey in `/home/jovyan/access_keys.json`.


# Installing a custom kernel
To run the notebooks with s3 support, and to write more interactive notebooks with sliders and other UI components, a custom kernel can be installed.
This contains all the packages in the Polytope kernel, plus extra packages for S3 and UI component support.
To do this, follow these instructions:

1. Create and activate new virtual environment 
`python -m venv /home/jovyan/urbanair-venv`
`source /home/jovyan/urbanair-venv/bin/activate`

2. Install the Polytope packages
It’s important that we retain access to the Polytope packages, and specific versions of them, that are in the Polytope custom kernel.  To ensure this, I have copied the list of Polytope packages from the Polytope kernel into requirements.txt, which we now install 

`pip install –r requirements.txt`

Install extra packages 
`pip install ipywidgets`
`pip install s3fs`

3. Create new kernel
`pip install ipykernel`
`ipython kernel install --user --name=urbanair-venv`

4. In use 
The kernel can now be selected from the drop-down Kernel menu in the Jupyter notebook (top right of the actual notebook). 

5. Note – updating the packages 
If you add packages to a virtual env that is being used as a kernel, then it seems you have to remove the kernel and then reinstall it: 

`jupyter kernelspec remove urbanair-venv `
`ipython kernel install –-user –-name=urbanair-venv `
 
You may also have to select another kernel and then select the urbanair-venv kernel to refresh it to the new version. 