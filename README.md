# Satellite-based Vegetation Optical Depth as an Indicator of Drought-driven Tree Mortality

This repository contains `python` and `R` scripts to estimate drought-driven tree mortality in California as part of paper in Journal <to be filled after publication>. 
  
## Repository details

The repository consists of scripts to download and regrid the data, perform statistical analysis such as breakpoint threshold identification and random forests regressions. All data used were obtained from public sources. 

## Prerequisites

1. `Python 2.7`
2. `arcpy`
3. `ranger`
4. `R v. 3.2.2 `

## Reproducibility guide

1. `git clone https://github.com/kkraoj/tree_mortality_from_vod.git`
2. Open analysis_random_forest.py and change `MyDir` variable to point to the folder where data.csv is installed.
3. Run the analysis_random_forest.py
4. From the same home directory, open and run rf_model_tuning.rmd to perform the random forest analysis
5. Go back to python IDE and run plot_functions.py by passing any of the listed arguments at the end of the script to reproduce the figure you wish.


## License
Please cite the following paper if you use any data or analysis from this study: <add citation key after publishing>.

## Acknowledgments

Thanks to the anonymous reviewers for helping us improve the manuscript
