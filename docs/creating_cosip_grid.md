# Creating a CoCIP Grid Environment for Contrail Modeling

This document describes how to generate a CoCIP grid environment for contrail modeling using the `generate_cosip_grid_environment` script.
The process involves specifying spatial and temporal bounds, pressure levels, and downloading meteorological data from the Copernicus Climate Data Store (CDS) using the CDS API.

## Required Inputs

To generate a CoCIP grid, you need to specify the following inputs:

- **Time bounds**: Start and end times for the model run, e.g., `(2024-01-01 00:00:00, 2024-01-01 23:00:00)`
- **Longitude bounds**: Tuple specifying the minimum and maximum longitude, e.g., `(-8, 3)`
- **Latitude bounds**: Tuple specifying the minimum and maximum latitude, e.g., `(49, 62)`
- **Pressure levels**: Tuple of pressure levels in hPa, e.g., `(200, 225, 250, 300, 350, 400, 450)`
- **Save filename**: Name for the output NetCDF file (without extension)

## Example Usage

The main script to generate the grid is `generate_cosip_grid_environment.py`.
This will download the required meteorological data, run the CoCIP grid model, and save the results as a NetCDF file in the folder `data/energy_forcing_data`.

## Weather Data Download (CDS API)

The script uses the pycontrails library, which relies on the CDS API to download ERA5 meteorological data.
To use this API:

### 1. Register for a CDS account

Go to [CDS Registration](https://cds.climate.copernicus.eu/user/register) and create an account.

### 2. Find your API key

After logging in, visit [Your CDS API key](https://cds.climate.copernicus.eu/how-to-api) and copy the key.

### 3. Save your API key locally

Create a file at `~/.cdsapirc` with your credentials.

### 4. Install the CDS API client

Run `pip install cdsapi` or `uv pip install cdsapi` in your environment.

## Output

The output is a NetCDF file containing gridded meteorological and contrail-relevant variables for further contrail modeling and analysis.

## References

- [pycontrails CoCiPGrid Documentation](https://py.contrails.org/)
- [CDS API How-To](https://cds.climate.copernicus.eu/how-to-api)
