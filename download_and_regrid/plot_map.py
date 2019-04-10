# -*- coding: utf-8 -*-
"""
plot_map function for plotting maps using basemap
helper function get_marker_size used to control
marker size inside plot_map

@author: kkrao
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def get_marker_size(ax,fig,loncorners,grid_size=0.25,marker_factor=1.):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    width *= fig.dpi
    marker_size=width*grid_size/np.diff(loncorners)[0]*marker_factor
    return marker_size

def plot_map(lats,lons, var =None,\
             latcorners = [-90,90], loncorners = [-180, 180],\
             enlarge = 1, marker_factor = 1, \
             cmap = 'YlGnBu', markercolor = 'r',\
             fill = 'papayawhip', background = 'lightcyan',\
             height = 3, width = 5,\
             drawcoast = True, drawcountries = False,\
             drawstates = False, drawcounties = False):
    """
    usage:
    fig, ax = plot_map(lats,lons,var)
    Above inputs are required. 
    
    To add color bar:
        cax = fig.add_axes([0.17, 0.3, 0.03, 0.15])
        fig.colorbar(plot,ax=ax,cax=cax)
    """
    fig, ax = plt.subplots(figsize=(width*enlarge,height*enlarge))
    marker_size=get_marker_size(ax,fig,loncorners, marker_factor)
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
                    llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                    llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                    ax=ax)
    if drawcoast:
        m.drawcoastlines()
    if drawcountries:
        m.drawcountries()
    if drawstates:
        m.drawstates()
    if drawcounties:
        m.drawcounties()
    m.drawmapboundary(fill_color=background)
    m.fillcontinents(color=fill,zorder=0)
    if var is not None:
        plot=m.scatter(lons, lats, s=marker_size,c=var,cmap=cmap,\
                        marker='s')
    else:
        m.scatter(lons, lats, s=marker_size,c=markercolor,\
                        marker='s')
    return fig, ax