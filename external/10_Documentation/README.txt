AquaSense AI – Dataset Repository

Study Area:
Phelps County, Nebraska, USA

Project:
Groundwater visualization, multimodal environmental data integration and AI-based groundwater analysis.

Dataset Categories:

1. Groundwater observations
   Source: USGS National Water Information System (NWIS)

2. Sentinel-1
   Source: Google Earth Engine
   Dataset ID: COPERNICUS/S1_GRD

3. Sentinel-2
   Source: Google Earth Engine
   Dataset ID: COPERNICUS/S2_SR_HARMONIZED

4. Landsat 8
   Source: Google Earth Engine / USGS
   Dataset ID: LANDSAT/LC08/C02/T1_L2

5. SRTM DEM
   Source: Google Earth Engine / NASA / USGS / JPL-Caltech
   Dataset ID: USGS/SRTMGL1_003

6. NASA POWER
   Source: NASA POWER API

7. SoilGrids
   Source: ISRIC – World Soil Information
   Dataset ID: ISRIC/SoilGrids250m/v2_0

Groundwater Processing:

Groundwater observations were cleaned and yearly groundwater surfaces were generated using Ordinary Kriging.

The annual groundwater GeoTIFFs were generated on a common spatial grid using EPSG:4326 so that the yearly rasters have consistent spatial geometry for temporal analysis.

Important:

Satellite, DEM and SoilGrids datasets accessed through Google Earth Engine are documented by their official dataset IDs rather than duplicated as local files.

This repository is intended to provide the dataset organization and source information required for the AquaSense AI research project.