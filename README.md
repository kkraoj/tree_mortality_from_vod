# Satellite-based Vegetation Optical Depth as an Indicator of Drought-driven Tree Mortality

This repository contains `python` and `R` scripts to estimate drought-driven tree mortality in California as part of research article in  in Journal Remote Sensing of Environment [Satellite-based vegetation optical depth as an indicator of drought-driven tree mortality](https://www.sciencedirect.com/science/article/pii/S0034425719301208)

## Repository details

The repository consists of scripts to perform the following-

1. Download and regrid the remote sensing data (scripts available in the folder `download_and_regrid`)
1. Perform statistical analysis such as breakpoint threshold identification and random forests regressions
  1. Breakpoint analysis is performed right before producing the scatter plot in `plot_rwc_cwd_all()` in `plot_functions.py`
  1. Random forest analysis is performed in `R` using the files in the folder `random_forest_analysis` - 
    1. The `analysis_random_forest.py` script is used to compile all the downloaded and regridded data into uniform rows and columns. The output is saved in the folder `random_forest_data`.
    1. The `rf_model_tuning.rmd` file is used to perform the random forest analysis. The output is saved in the folder `random_forest_data`.
1. Plot the data to reproduce the figures presented in the research article using `plot_functions.py`

All data used were obtained from public sources. 

## Prerequisites

1. `Python 2.7`
1. `arcpy` package from ARC GIS
1. `R v. 3.2.2 `
1. `ranger` package in `R`

## Reproducibility guide

1. Clone the repository using `git clone https://github.com/kkraoj/tree_mortality_from_vod.git`
1. Open plot_functions.py and change `CA_Dir` variable to point to the folder where random_forest_data folder is installed
1. Run `plot_functions.py` by uncommenting any of the functions at the end of the script to reproduce the figures you wish.


## License
Please cite the following paper if you use any data or analysis from this study:
_Rao, K., Anderegg, W.R.L., Sala, A., Martínez-Vilalta, J. & Konings, A.G. (2019). Satellite-based vegetation optical depth as an indicator of drought-driven tree mortality. Remote Sens. Environ., 227, 125–136._

## Acknowledgments

Thanks to the anonymous reviewers for helping us improve the manuscript
