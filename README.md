# Satellite-based Vegetation Optical Depth as an Indicator of Drought-driven Tree Mortality

This repository contains `python` and `R` scripts to estimate drought-driven tree mortality in California as part of research article in the Journal Remote Sensing of Environment- [Satellite-based vegetation optical depth as an indicator of drought-driven tree mortality](https://www.sciencedirect.com/science/article/pii/S0034425719301208)

## Repository details

### Data:
1. All data files are under `data` folder.
   1. The input data files containing 32 variables for the random forest model are named as `rf_data_` followed by the name of the respective model setup (base model, trimmed model, etc.)
   1. The predicted fractional area of mortality from the random forest analysis is provided in `rf_predicted.csv`
   1. The feature importance from random forest analysis is provided in files `rf_sensitivity` followed by the model name. 


### Scripts:
The repository consists of scripts to perform the following-

1. Download and regrid the remote sensing data (scripts available in the folder `download_and_regrid`)
1. Perform statistical analysis such as breakpoint threshold identification and random forests regressions
   - Breakpoint analysis is performed right before producing the scatter plot in `plot_rwc_cwd_all()` in `plot_functions.py`
   - Random forest analysis is performed in `R` using the files in the folder `random_forest_analysis` - 
     -  The `analysis_random_forest.py` script is used to compile all the downloaded and regridded data into uniform rows and columns. The output is saved in the folder `random_forest_data`.
     - The `rf_model_tuning.rmd` file is used to perform the random forest analysis. The output is saved in the folder `random_forest_data`.
1. Plot the data to reproduce the figures presented in the research article using `plot_functions.py`

All data used were obtained from public sources.

## Data Sources

The data compiled and used the analysis were obtained from the following sources (refer last column):

|	Category	|	Variable Name	|	Description	|	Season	|	Time Dependence	|	Source	|
|	:---	|	:---	|	:---	|	:---	|	:---	|	:---	|
|	Vegetation	|	Canopy height	|	Height of canopy	|	All-year	|	Constant	|	Simard et al., 2011	|
|		|	Forest cover	|	Fraction of woody cover	|	All-year	|	Constant	|	Arino et al., 2010	|
|		|	LAIsum	|	Leaf area index	|	Summer	|	Variable	|	Yuan et al., 2011	|
|		|	LAIwin	|	Leaf area index	|	Winter	|	Variable	|	Yuan et al., 2011	|
|		|	NDWIsum	|	Normalized Difference Water Index	|	Summer	|	Variable	|	Schaaf and Wang, 2015	|
|		|	NDWIwin	|	Normalized Difference Water Index	|	Winter	|	Variable	|	Schaaf and Wang, 2015	|
|		|	RWC	|	Relative water content	|	Summer	|	Variable	|	Du et al., 2016	|
|		|	Tree density	|	Basal area of live trees per unit area	|	All-year	|	Variable	|	Young et al., 2017	|
|	Topography	|	Aspect	|	Mean aspect	|	All-year	|	Constant	|	USGS, 2011b	|
|		|	Aspectsd	|	Standard deviation of aspect	|	All-year	|	Constant	|	USGS, 2011b	|
|		|	Elevation	|	Mean altitude	|	All-year	|	Constant	|	USGS, 2011a	|
|		|	Elevationsd	|	Standard deviation of altitude	|	All-year	|	Constant	|	USGS, 2011a	|
|		|	Location	|	Factored Latitude-Longitude of pixel center	|	All-year	|	Constant	|	-	|
|		|	Sand	|	Mean sand fraction of top-soil	|	All-year	|	Constant	|	Liu et al., 2014	|
|		|	Silt	|	Mean silt fraction of top-soil	|	All-year	|	Constant	|	Liu et al., 2014	|
|		|	TWI	|	Mean topographic wetness index	|	All-year	|	Constant	|	USGS, 2000	|
|		|	TWIsd	|	Standard deviation of  topographic wetness index	|	All-year	|	Constant	|	USGS, 2000	|
|	Climate	|	AETsum	|	Actual evapotranspiration	|	Summer	|	Variable	|	Xia et al., 2012	|
|		|	AETwin	|	Actual evapotranspiration	|	Winter	|	Variable	|	Xia et al., 2012	|
|		|	CWD	|	Annually accumulated climatic water deficit	|	All-year	|	Variable	|	Young et al., 2017	|
|		|	PETsum	|	Potential evapotranspiration	|	Summer	|	Variable	|	Xia et al., 2012	|
|		|	PETwin	|	Potential evapotranspiration	|	Winter	|	Variable	|	Xia et al., 2012	|
|		|	Psum	|	Precipitation	|	Summer	|	Variable	|	PRISM, 2016	|
|		|	Pwin	|	Precipitation	|	Winter	|	Variable	|	PRISM, 2016	|
|		|	SMsum	|	Soil moisture	|	Summer	|	Variable	|	Du et al., 2016	|
|		|	SMwin	|	Soil moisture	|	Winter	|	Variable	|	Du et al., 2016	|
|		|	Tmax, sum	|	Maximum temperature	|	Summer	|	Variable	|	PRISM, 2016	|
|		|	Tmax, win	|	Maximum temperature	|	Winter	|	Variable	|	PRISM, 2016	|
|		|	Tsum	|	Mean temperature	|	Summer	|	Variable	|	PRISM, 2016	|
|		|	Twin	|	Mean temperature	|	Winter	|	Variable	|	PRISM, 2016	|
|		|	VPDmax, sum	|	Maximum vapor pressure deficit	|	Summer	|	Variable	|	PRISM, 2016	|
|		|	VPDmax, win	|	Maximum vapor pressure deficit	|	Winter	|	Variable	|	PRISM, 2016	|


## Prerequisites

1. `Python 2.7`
1. `arcpy` package from ARC GIS
1. `R v. 3.2.2 `
1. `ranger` package in `R`

## Reproducibility guide

1. Clone the repository using `git clone https://github.com/kkraoj/tree_mortality_from_vod.git`
1. Open plot_functions.py and change `CA_Dir` variable to point to the folder where `random_forest_data` folder is located
1. Run `plot_functions.py` by uncommenting any of the functions at the end of the script to reproduce the figures you wish

## License
Please cite the following paper if you with to use any data or analyses from this study:

**Rao, K., Anderegg, W.R.L., Sala, A., Martínez-Vilalta, J. & Konings, A.G. (2019). Satellite-based vegetation optical depth as an indicator of drought-driven tree mortality. Remote Sens. Environ., 227, 125–136.**

## Acknowledgments

Thanks to the anonymous reviewers for helping us improve the manuscript

## Issues?

Check the `Issues` tab for troubleshooting or create a new issue.

## References

1. Simard, M., Pinto, N., Fisher, J.B. & Baccini, A. (2011). Mapping forest canopy height globally with spaceborne lidar. J. Geophys. Res. Biogeosciences, 116, 1–12.
1. Arino, O., Perez, J.R., Kalogirou, V., Defourny, P. & Achard, F. (2010). Globcover 2009. ESA Living Planet Symp., 1–3.
1. Yuan, H., Dai, Y., Xiao, Z., Ji, D. & Shangguan, W. (2011). Reprocessing the MODIS Leaf Area Index products for land surface and climate modelling. Remote Sens. Environ., 115, 1171–1187.
1. Schaaf, C., Wang, Z. (2015). MCD43A4 V006 | LP DAAC :: NASA Land Data Products and Services. Available at: https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mcd43a4_v006. Last accessed 29 January 2019.
1. Du, J., Kimball, J.S. & Jones, L.A. (2016). Passive Microwave Remote Sensing of Soil Moisture Based on Dynamic Vegetation Scattering Properties for AMSR-E. IEEE Trans. Geosci. Remote Sens., 54, 597–608.
1. Young, D.J.N., Stevens, J.T., Earles, J.M., Moore, J., Ellis, A., Jirka, A.L., et al. (2017). Long-term climate and competition explain forest mortality patterns under extreme drought. Ecol. Lett., 20, 78–86.
1. USGS. (2011b). National Gap Analysis Project (GAP). Available at: https://gapanalysis.usgs.gov/data/web-services/. Last accessed 14 August 2016.
1. USGS. (2011a). National Elevation Dataset. Available at: https://nationalmap.gov/elevation.html. Last accessed 14 August 2016.
1. Liu, S., Wei, Y., Post, W.M., Cook, R.B., Schaefer, K. & Thorton, M.M. (2014). NACP MsTMIP: Unified North American Soil Map.
1. USGS. (2000). HYDRO1k | The Long Term Archive. Available at: https://lta.cr.usgs.gov/HYDRO1KReadMe. Last accessed 13 November 2018.
1. Xia, Y., Mitchell, K., Ek, M., Sheffield, J., Cosgrove, B., Wood, E., et al. (2012). Continental-scale water and energy flux analysis and validation for the North American Land Data Assimilation System project phase 2 (NLDAS-2): 1. Intercomparison and application of model products. J. Geophys. Res. Atmos., 117 
1. PRISM Climate Group Oregon State University. (2016). PRISM Climate Data. Available at: http://prism.oregonstate.edu/. Last accessed 1 March 2018.
