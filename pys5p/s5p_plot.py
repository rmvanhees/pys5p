"""
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

The class ICMplot contains generic plot functions to display S5p Tropomi data

-- generate figures --
 Public functions a page in the output PDF
 * draw_signal
 * draw_hist
 * draw_quality
 * draw_geo

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from collections import OrderedDict

import numpy as np

import matplotlib as mpl

mpl.use('TkAgg')

#
# Suggestion for the name of the report/pdf-file
#    <identifier>_<yyyymmdd>_<orbit>.pdf
# where
#  identifier : name of L1B/ICM/OCM product or monitoring database
#  yyyymmdd   : coverage start-date or start-date of monitoring entry
#  orbit      : reference orbit
#
# Suggestion for info-structure to be displayed in figure
# * Versions used to generate the data:
#   algo_version : version of monitoring algorithm (SRON)
#   db_version   : version of the monitoring database (SRON)
#   l1b_version  : L1b processor version
#   icm_version  : ICM processor version
#   msm_date     : Datetime of first frame
# * Data in detector coordinates or column/row averaged as function of time
#   sign_median  : (biweight) median of signals
#   sign_spread  : (biweight) spread of signals
#   error_median : (biweight) median of errors
#   error_spread : (biweight) spread of errors
# * Detector quality data:
#   dpqf_01      : nummer of good pixels with threshold at 0.1
#   dpqf_08      : nummer of good pixels with threshold at 0.8
#
# To force the sequence of the displayed information it is adviced to use
# collections.OrderedDict:
#
# > from collections import OrderedDict
# > dict_info = OrderedDict({key1 : value1})
# > dict_info.update({key2 : value2})
# etc.
#

# pylint: disable=too-many-arguments, too-many-locals
class S5Pplot(object):
    """
    Generate figure(s) for the SRON Tropomi SWIR monitor website or MPC reports

    The PDF will have the following name:
        <dbname>_<startDateTime of monitor entry>_<orbit of monitor entry>.pdf
    """
    def __init__( self, figname, cmap="Rainbow", mode='frame' ):
        from matplotlib.backends.backend_pdf import PdfPages

        self.__pdf  = PdfPages( figname )
        self.__cmap = cmap
        self.__mode = mode

    def __repr__( self ):
        pass

    def __del__( self ):
        self.__pdf.close()

    # --------------------------------------------------
    @staticmethod
    def __fig_info( fig, dict_info, aspect=4 ):
        """
        Add meta-information in the current figure

        Parameters
        ----------
        fig       :  Matplotlib Figure instance
        dict_info :  dictionary or sortedDict
           legenda parameters to be displayed in the figure
        """
        from datetime import datetime

        copy_str = r'$\copyright$ SRON Netherlands Institute for Space Research'
        info_str = 'date : ' + datetime.utcnow().isoformat(' ')[0:19]
        for key in dict_info:
            if isinstance(dict_info[key], float) \
               or isinstance(dict_info[key], np.float32):
                info_str += '\n{} : {:.5g}'.format(key, dict_info[key])
            else:
                info_str += '\n{} : {}'.format(key, dict_info[key])

        if aspect == 1:
            fig.text(0.2, 0.18, info_str,
                     fontsize=12, style='italic',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':10})
            fig.text(0.975, 0.025, copy_str,
                     fontsize=10, family='monospace',
                     rotation='vertical',
                     verticalalignment='bottom',
                     horizontalalignment='left')
        elif aspect == 2:
            fig.text(0.225, 0.35, info_str,
                     fontsize=12, style='italic',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':10})
            fig.text(0.02, 0.875, copy_str,
                     fontsize=9, family='monospace',
                     rotation='horizontal',
                     verticalalignment='bottom',
                     horizontalalignment='left')
        else:
            fig.text(0.15, 0.35, info_str,
                     fontsize=12, style='italic',
                     verticalalignment='top',
                     horizontalalignment='right',
                     multialignment='left',
                     bbox={'facecolor':'white', 'pad':10})
            fig.text(0.02, 0.875, copy_str,
                     fontsize=9, family='monospace',
                     rotation='horizontal',
                     verticalalignment='bottom',
                     horizontalalignment='left')
                
            # --------------------------------------------------
    def draw_signal( self, data, data_col=None, data_row=None,
                     data_label='signal', data_unit=None, time_axis=None,
                     title=None, sub_title=None, fig_info=None,
                     vperc=[1., 99.], vrange=None ):
        """
        Display 2D array data as image and averaged column/row signal plots

        Parameters
        ----------
        data       :  ndarray
           Numpy array (2D) holding the data to be displayed
        data_col   :  ndarray
           Numpy array (1D) column averaged values of data
           Default is calculated as biweight(data, axis=1)
        data_row   :  ndarray
           Numpy array (1D) row averaged values of data
           Default is calculated as biweight(data, axis=0)
        vrange     :  float in range of [vmin,vmax]
           Range to normalize luminance data between min and max. Note that is 
           you pass a vrange instance then vperc wil be ignored
        vperc      :  float in range of [0,100]
           Range to normalize luminance data between percentiles min and max of 
           array data. Default is [1., 99.]
        data_label :  string
           Name of dataset. Default is 'signal'
        data_unit  :  string
           Units of dataset. Default is None
        time_axis  :  tuple {None, ('x', t_intg), ('y', t_intg)}
           Defines if one of the axis is a time-axis. Default is None
           Where t_intg is the time between succesive readouts (in seconds)
        title      :  string
           Title of the figure. Default is None
           Suggestion: use attribute "title" of data-product
        sub_title  :  string
           Sub-title of the figure. Default is None
           Suggestion: use attribute "comment" of data-product
        fig_info   :  dictionary
           OrderedDict holding meta-data to be displayed in the figure

        """
        from matplotlib import pyplot as plt
        from matplotlib import gridspec

        from . import sron_colorschemes

        sron_colorschemes.register_cmap_rainbow()
        line_colors = sron_colorschemes.get_line_colors()

        # calculate column/row medians (if required)
        if data_col is None:
            data_col = np.nanmedian( data, axis=1 )

        if data_row is None:
            data_row = np.nanmedian( data, axis=0 )

        # determine aspect-ratio of data and set sizes of figure and sub-plots
        dims = data.shape
        if time_axis is None:                      # band or detector lay-out
            aspect = int(np.round(dims[1] / dims[0]))
        else:                                      # trend data
            aspect = min(4, max(1, int(np.round(dims[1] / dims[0]))))

        if aspect == 1:
            figsize = (16.5001, 15)
            gdim = np.array([( 1, 12, 2, 8), ( 1, 12, 8, 28),
                             ( 12, 15, 8, 28), (1, 15, 28, 31)],
                            dtype=[('xmn','u2'),('xmx','u2'),
                                   ('ymn','u2'),('ymx','u2')])
        elif aspect == 2:
            figsize = (13.5001, 8)
            gdim = np.array([( 1, 5, 2, 8), ( 1, 5, 8, 22),
                             ( 5, 8, 8, 22), (1, 8, 22, 25)],
                            dtype=[('xmn','u2'),('xmx','u2'),
                                   ('ymn','u2'),('ymx','u2')])
        elif aspect == 4:
            figsize = (20.5001, 8)
            gdim = np.array([( 1, 5, 2, 8), ( 1, 5, 8, 36),
                             ( 5, 8, 8, 36), (1, 8, 36, 39)],
                            dtype=[('xmn','u2'),('xmx','u2'),
                                   ('ymn','u2'),('ymx','u2')])
        else:
            print( '*** FATAL: aspect ratio not implemented, exit' )
            return
        #print( 'aspect: {}'.format(aspect) )

        # set label and range of X/Y axis
        xlabel = 'column'
        ylabel = 'row'
        xdata = np.arange(data_row.size, dtype=float)
        xmax = data_row.size
        ydata = np.arange(data_col.size, dtype=float)
        ymax = data_col.size
        if time_axis is not None:
            if time_axis[0] == 'y':
                ylabel = 'time(s)'
                ydata *= time_axis[1]
                ymax *= time_axis[1]
            elif time_axis[0] == 'x':
                xlabel = 'time(s)'
                xdata *= time_axis[1]
                xmax *= time_axis[1]
        extent = [0, xmax, 0, ymax]

        # scale data to keep reduce number of significant digits small to
        # the axis-label and tickmarks readable
        dscale = 1.0
        if vrange is None:
            assert len(vperc) == 2
            (vmin, vmax) = np.percentile( data[np.isfinite(data)], vperc )
        else:
            assert len(vrange) == 2
            (vmin, vmax) = vrange
        
        if data_unit is None:
            zunit = None
            zlabel = '{}'.format(data_label)
        elif data_unit.find( 'electron' ) >= 0:
            max_value = max(abs(vmin), abs(vmax))

            if max_value > 1000000000:
                dscale = 1e9
                zunit = data_unit.replace('electron', 'Ge')
            elif max_value > 1000000:
                dscale = 1e6
                zunit = data_unit.replace('electron', 'Me')
            elif max_value > 1000:
                dscale = 1e3
                zunit = data_unit.replace('electron', 'ke')
            else:
                zunit = data_unit.replace('electron', 'e')
            zlabel = '{} [{}]'.format(data_label, zunit)
        else:
            zunit = data_unit
            zlabel = '{} [{}]'.format(data_label, data_unit)

        # inititalize figure
        fig = plt.figure(figsize=figsize)
        if title is not None:
            fig.suptitle( title, fontsize=24 )
        gspec = gridspec.GridSpec(np.rint(figsize[1]).astype(int),
                                  np.rint(2 * figsize[0]).astype(int))

        # draw sub-plot 1
        axx = plt.subplot(gspec[gdim[0]['xmn']:gdim[0]['xmx'],
                                gdim[0]['ymn']:gdim[0]['ymx']])
        axx.plot(data_col / dscale, ydata, lw=0.5, color=line_colors[0])
        axx.set_ylim([0, ymax])
        axx.grid(linestyle=':')
        axx.locator_params(axis='x', nbins=4)
        axx.set_xlabel(zlabel)
        axx.set_ylabel(ylabel)

        # draw sub-plot 2
        axx = plt.subplot(gspec[gdim[1]['xmn']:gdim[1]['xmx'],
                                gdim[1]['ymn']:gdim[1]['ymx']])
        axx.imshow( data / dscale, cmap=self.__cmap, extent=extent,
                    vmin=vmin / dscale, vmax=vmax / dscale,
                    aspect='auto', interpolation='none', origin='lower' )
        if sub_title is not None:
            axx.set_title( sub_title )

        # draw sub-plot 3
        axx = plt.subplot(gspec[gdim[2]['xmn']:gdim[2]['xmx'],
                                gdim[2]['ymn']:gdim[2]['ymx']])
        axx.plot(xdata, data_row / dscale, lw=0.5, color=line_colors[0])
        axx.set_xlim([0, xmax])
        axx.grid(linestyle=':')
        axx.locator_params(axis='y', nbins=6)
        axx.set_xlabel(xlabel)
        axx.set_ylabel(zlabel)

        # draw sub-plot 4
        axx = plt.subplot(gspec[gdim[3]['xmn']:gdim[3]['xmx'],
                                gdim[3]['ymn']:gdim[3]['ymx']])
        norm = mpl.colors.Normalize(vmin=vmin / dscale, vmax=vmax / dscale)
        cb1 = mpl.colorbar.ColorbarBase( axx, cmap=self.__cmap, norm=norm,
                                         orientation='vertical' )
        cb1.set_label(zlabel)

        # add annotation
        if fig_info is None:
            from .biweight import biweight

            (median, spread) = biweight( data, spread=True)
            if zunit is not None:
                median_str = '{:.5g} {}'.format(median / dscale, zunit)
                spread_str = '{:.5g} {}'.format(spread / dscale, zunit)
            else:
                median_str = '{:.5g}'.format(median)
                spread_str = '{:.5g}'.format(spread)

            fig_info = OrderedDict({'median' : median_str})
            fig_info.update({'spread' : spread_str})

        # save and close figure
        self.__fig_info( fig, fig_info, aspect )
        plt.tight_layout()
        self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_hist( self, data_in, error_in,
                   data_label=None, data_unit=None,
                   error_label=None, error_unit=None,
                   title=None, sub_title=None, fig_info=None ):
        """
        Display signal & its errors as histograms

        Parameters
        ----------
        data        :  ndarray
           Numpy array (2D) holding the data to be displayed
        error       :  ndarray
           Numpy array (2D) holding the data to be displayed
        data_label  :  string
           Name of dataset
        data_unit   :  string
           Units of dataset
        error_label :  string
           Name of dataset
        error_unit  :  string
           Units of dataset
        title       :  string
           Title of the figure (use attribute "title" of product)
        sub_title   :  string
           Sub-title of the figure (use attribute "comment" of product)
        fig_info    :  dictionary
           Dictionary holding meta-data to be displayed in the figure

        """
        from matplotlib import pyplot as plt
        from matplotlib import gridspec

        from .biweight import biweight
        from .sron_colorschemes import get_line_colors

        line_colors = get_line_colors()

        if fig_info is None:
            fig_info = OrderedDict({'num_sigma' : 3})
        else:
            fig_info.update({'num_sigma' : 3})

        data = data_in[np.isfinite(data_in)].reshape(-1)
        if 'sign_median' not in fig_info \
            or 'sign_spread' not in fig_info:
            (median, spread) = biweight(data, spread=True)
            fig_info.update({'sign_median' : median})
            fig_info.update({'sign_spread' : spread})
        data -= fig_info['sign_median']

        error = error_in[np.isfinite(error_in)].reshape(-1)
        if 'error_median' not in fig_info \
            or 'error_spread' not in fig_info:
            (median, spread) = biweight(error, spread=True)
            fig_info.update({'error_median' : median})
            fig_info.update({'error_spread' : spread})
        error -= fig_info['error_median']

        if data_unit is not None:
            d_label = '{} [{}]'.format(data_label, data_unit)
        else:
            d_label = data_label

        if error_unit is not None:
            e_label = '{} [{}]'.format(error_label, error_unit)
        else:
            e_label = error_label

        fig = plt.figure(figsize=(18, 7.875))
        if title is not None:
            fig.suptitle( title, fontsize=24 )
        gspec = gridspec.GridSpec(6,8)

        axx = plt.subplot(gspec[1:3,1:])
        axx.hist( data,
                  range=[-fig_info['num_sigma'] * fig_info['sign_spread'],
                         fig_info['num_sigma'] * fig_info['sign_spread']],
                  bins=15, color=line_colors[0] )
        axx.set_title( r'Histogram is centered at the median with range of ' \
                       r'$\pm 3 \sigma$' )
        axx.set_xlabel( d_label )
        axx.set_ylabel( 'count' )
        if sub_title is not None:
            axx.set_title( sub_title )

        axx = plt.subplot(gspec[3:5,1:])
        axx.hist( error,
                  range=[-fig_info['num_sigma'] * fig_info['error_spread'],
                         fig_info['num_sigma'] * fig_info['error_spread']],
                  bins=15, color=line_colors[0] )
        axx.set_xlabel( e_label )
        axx.set_ylabel( 'count' )

        self.__fig_info( fig, fig_info )
        plt.tight_layout()
        self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_quality( self, dpqm, low_thres=0.1, high_thres=0.8,
                      title=None, sub_title=None, fig_info=None ):
        """
        Display pixel quality data

        Parameters
        ----------
        dpqm        :  ndarray
           Numpy array (2D) holding pixel-quality data
        title       :  string
           Title of the figure (use attribute "title" of product)
        sub_title   :  string
           Sub-title of the figure (use attribute "comment" of product)

        """
        from matplotlib import pyplot as plt
        from matplotlib import gridspec

        thres_min = 10 * low_thres
        thres_max = 10 * high_thres
        dpqf = (dpqm * 10).astype(np.int8)
        unused_cols = np.where(np.sum(dpqm, axis=0) < (256 // 4))
        if unused_cols[0].size > 0:
            dpqf[:,unused_cols[0]] = -1
        unused_rows = np.where(np.sum(dpqm, axis=1) < (1000 // 4))
        if unused_rows[0].size > 0:
            dpqf[:,unused_rows[0]] = -1

        fig = plt.figure(figsize=(18, 7.875))
        if title is not None:
            fig.suptitle( title, fontsize=24 )
        gspec = gridspec.GridSpec(6,16)

        clist = ['#BBBBBB', '#EE6677','#CCBB44','w']
        cmap = mpl.colors.ListedColormap(clist)
        bounds=[-1, 0, thres_min, thres_max, 10]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        dpqf_col_01 = np.sum( ((dpqf >= 0) & (dpqf < thres_min)), axis=1 )
        dpqf_col_08 = np.sum( ((dpqf >= 0) & (dpqf < thres_max)), axis=1 )

        axx = plt.subplot(gspec[1:4,0:2])
        axx.step(dpqf_col_08, np.arange(dpqf_col_08.size),
                 lw=0.5, color=clist[2] )
        axx.step(dpqf_col_01, np.arange(dpqf_col_01.size),
                 lw=0.6, color=clist[1] )
        axx.set_xlim( [0, 30] )
        axx.set_xlabel( 'bad (count)' )
        axx.set_ylim( [0, dpqf_col_01.size-1] )
        axx.set_ylabel( 'row' )
        axx.grid(True)

        axx = plt.subplot(gspec[1:4,2:14])
        axx.imshow( dpqf, cmap=cmap, norm=norm,
                    aspect=1, vmin=-1, vmax=10,
                    interpolation='none', origin='lower' )
        if sub_title is not None:
            axx.set_title( sub_title )

        dpqf_row_01 = np.sum( ((dpqf >= 0) & (dpqf < thres_min)), axis=0 )
        dpqf_row_08 = np.sum( ((dpqf >= 0) & (dpqf < thres_max)), axis=0 )

        axx = plt.subplot(gspec[4:6,2:14])
        axx.step(np.arange(dpqf_row_08.size), dpqf_row_08,
                 lw=0.5, color=clist[2] )
        axx.step(np.arange(dpqf_row_01.size), dpqf_row_01,
                 lw=0.6, color=clist[1] )
        axx.set_ylim( [0, 10] )
        axx.set_ylabel( 'bad (count)' )
        axx.set_xlim( [0, dpqf_row_01.size-1] )
        axx.set_xlabel( 'column' )
        axx.grid(True)

        if fig_info is None:
            fig_info = OrderedDict({'thres_01': low_thres})
        else:
            fig_info.update({'thres_01': low_thres})
        fig_info.update({'dpqf_01': np.sum(((dpqf >= 0) & (dpqf < thres_min)))})
        fig_info.update({'thres_08': high_thres})
        fig_info.update({'dpqf_08': np.sum(((dpqf >= 0) & (dpqf < thres_max)))})

        self.__fig_info( fig, fig_info )
        plt.tight_layout()
        self.__pdf.savefig()
        plt.close()

    # --------------------------------------------------
    def draw_geolocation( self, lats, lons, sequence=None,
                          subsatellite=False, proj='hammer',
                          title=None, sub_title=None, fig_info=None ):
        """
        Display footprint of sub-satellite coordinates project on the globe

        Parameters
        ----------
        lats         :  ndarray
           Latitude coordinates
        lons         :  ndarray
           Longitude coordinates
        subsatellite :  boolean
           Coordinates are given for sub-satellite point. Default is False
        proj         :  string
           Projection, see toolkit basemap. Default is 'hammer'
        title        :  string
           Title of the figure (use attribute "title" of product)
        sub_title    :  string
           Sub-title of the figure (use attribute "comment" of product)

        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon

        from mpl_toolkits.basemap import Basemap

        from . import sron_colorschemes

        sron_colorschemes.register_cmap_rainbow()
        line_colors = sron_colorschemes.get_line_colors()

        # determine central longitude
        if lons.max() - lons.min() > 180:
            if np.sum( lons > 0 ) > np.sum( lons < 0 ):
                lons[lons < 0] += 360
            else:
                lons[lons > 0] -= 360
        lon_0 = np.around(np.mean(lons), decimals=-1)

        # inititalize figure
        fig = plt.figure(figsize=(13, 8))
        if title is not None:
            fig.suptitle( title, fontsize=24 )

        # draw worldmap
        m = Basemap(projection=proj, lon_0=lon_0, resolution='l')
        m.drawparallels(np.arange(-90.,120.,30.))
        m.drawmeridians(np.arange(0.,420.,60.))
        #m.drawcoastlines()
        m.drawmapboundary(fill_color=line_colors[1])
        m.fillcontinents(color=line_colors[2], lake_color=line_colors[1])

        # draw footprint
        if not subsatellite:
            if sequence is None:
                lat = np.concatenate((lats[0, :], lats[1:-1, -1],
                                      lats[-1, ::-1], lats[1:-1:-1, 0]))
                lon = np.concatenate((lons[0, :], lons[1:-1, -1],
                                      lons[-1, ::-1], lons[1:-1:-1, 0]))

                (x, y) = m(lon, lat)
                xy = np.squeeze(np.dstack((x,y)))
                poly = Polygon(xy, facecolor=line_colors[3], alpha=0.6)
                plt.gca().add_patch(poly)
            else:
                for ii in np.unique(sequence):
                    indx = np.unique(np.where(sequence == ii)[0])
                    indx_rev = indx[::-1]
                    lat = np.concatenate((lats[indx[0], :],
                                          lats[indx, -1],
                                          lats[indx[-1], ::-1],
                                          lats[indx_rev, 0]))
                    lon = np.concatenate((lons[indx[0], :],
                                          lons[indx, -1],
                                          lons[indx[-1], ::-1],
                                          lons[indx_rev, 0]))

                    (x, y) = m(lon, lat)
                    xy = np.squeeze(np.dstack((x,y)))
                    poly = Polygon(xy, facecolor=line_colors[3], alpha=0.6)
                    plt.gca().add_patch(poly)
                
        # draw sub-satellite coordinates
        if subsatellite:
            (x, y) = m(lons, lats)
            m.scatter(x, y, 4, marker='o', color=line_colors[4])

        if fig_info is None:
            fig_info = OrderedDict({'lon0': lon_0})

        self.__fig_info( fig, fig_info )
        plt.tight_layout()
        self.__pdf.savefig()
        plt.close()
##
## --------------------------------------------------
##
def test_geo():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from l1b_io import L1BioCAL, L1BioRAD

    plot = S5Pplot( 'test_geo.pdf' )

    # test footprint mode
    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data/L1B-RADIANCE'
    else:
        data_dir = '/stage/EPSstorage/jochen/TROPOMI_ODA_DATA/data_set_05102016/OFFL/L1B/2015/08/21/00058/L1B-RADIANCE'
    fl_name = 'S5P_OFFL_L1B_RA_BD7_20150821T012540_20150821T030710_00058_01_000000_20160721T173643.nc'
    l1b = L1BioRAD( os.path.join(data_dir, fl_name) )
    l1b.select()
    geo = l1b.get_geo_data( icid=4 )
    del l1b

    plot.draw_geolocation( geo['latitude'], geo['longitude'],
                           sequence=geo['sequence'] )

    # test subsatellite mode
    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data/L1B-CALIBRATION'
    else:
        data_dir = '/stage/EPSstorage/jochen/TROPOMI_ODA_DATA/data_set_05102016/OFFL/L1B/2015/08/21/00058/L1B-CALIBRATION'
    fl_name = 'S5P_OFFL_L1B_CA_SIR_20150821T012540_20150821T030710_00058_01_000000_20160721T173643.nc'
    l1b = L1BioCAL( os.path.join(data_dir, fl_name) )
    l1b.select('BACKGROUND_RADIANCE_MODE_0005')
    geo = l1b.get_geo_data()
    del l1b

    plot.draw_geolocation( geo['satellite_latitude'],
                           geo['satellite_longitude'],
                           sequence=geo['sequence'],
                           subsatellite=True )

    del plot
    
#-------------------------
def test_frame():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from ocm_io import OCMio

    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data/proc_knmi/2015_02_23T01_36_51_svn4709_CellEarth_CH4'
    else:
        data_dir ='/nfs/TROPOMI/ocal/proc_knmi/2015_02_23T01_36_51_svn4709_CellEarth_CH4'

    icid = 31523

    # Read BAND7 product
    product_b7 = 'trl1brb7g.lx.nc'
    ocm_product = os.path.join( data_dir, 'before_strayl_l1b_val_SWIR_2',
                                product_b7 )
    # open OCAL Lx poduct
    ocm7 = OCMio( ocm_product )

    # select data of measurement(s) with given ICID
    if ocm7.select( icid ) > 0:
        dict_b7 = ocm7.get_msm_data( 'signal' )
        dict_b7_std = ocm7.get_msm_data( 'signal_error_vals' )

    # Read BAND8 product
    product_b8 = 'trl1brb8g.lx.nc'
    ocm_product = os.path.join( data_dir, 'before_strayl_l1b_val_SWIR_2',
                                product_b8 )
    # open OCAL Lx poduct
    ocm8 = OCMio( ocm_product )

    # select data of measurement(s) with given ICID
    if ocm8.select( icid ) > 0:
        dict_b8 = ocm8.get_msm_data( 'signal' )
        dict_b8_std = ocm8.get_msm_data( 'signal_error_vals' )

    # Combine band 7 & 8 data
    # del dict_b7['ICID_31524_GROUP_00000']
    # del dict_b8['ICID_31524_GROUP_00000']
    for key in dict_b7:
        print( key, dict_b7[key].shape )
    data = ocm7.band2channel( dict_b7, dict_b8,
                              mode=['combine', 'median'],
                              skip_first=True, skip_last=True )
    error = ocm7.band2channel( dict_b7_std, dict_b8_std,
                               mode=['combine', 'median'],
                               skip_first=True, skip_last=True )

    # Generate figure
    figname = os.path.basename(data_dir) + '.pdf'
    plot = S5Pplot( figname )
    plot.draw_signal( data,
                      data_label='signal',
                      data_unit=ocm7.get_msm_attr('signal', 'units'),
                      title=ocm7.get_attr('title'),
                      sub_title='ICID={}'.format(icid),
                      fig_info=None )
    plot.draw_signal( error,
                      data_label='signal_error_vals',
                      data_unit=ocm7.get_msm_attr('signal_error_vals', 'units'),
                      title=ocm7.get_attr('title'),
                      sub_title='ICID={}'.format(icid),
                      fig_info=None )
    plot.draw_hist( data, error,
                    data_label='signal',
                    data_unit=ocm7.get_msm_attr('signal', 'units'),
                    error_label='signal_error_vals',
                    error_unit=ocm7.get_msm_attr('signal_error_vals', 'units'),
                    title=ocm7.get_attr('title'),
                    sub_title='ICID={}'.format(icid),
                    fig_info=None )
    del plot
    del ocm7
    del ocm8

#-------------------------
def test_ckd_dpqf():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    import h5py

    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data'
    else:
        data_dir ='/nfs/TROPOMI/ocal/ckd/ckd_release_swir/dpqf'
    dpqm_fl = os.path.join(data_dir, 'ckd.dpqf.detector4.nc')

    with h5py.File( dpqm_fl, 'r' ) as fid:
        band7 = fid['BAND7/dpqf_map'][:-1,:]
        band8 = fid['BAND8/dpqf_map'][:-1,:]
        dpqm = np.hstack( (band7, band8) )

    # generate figure
    figname = 'ckd.dpqf.detector4.pdf'
    plot = S5Pplot( figname )
    plot.draw_quality( dpqm,
                       title='ckd.dpqf.detector4.nc',
                       sub_title='dpqf_map' )
    del plot

#-------------------------
def test_icm_dpqf():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from icm_io import ICMio

    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data/S5P_ICM_CA_SIR/001000/2012/09/18'
    elif os.path.isdir('/nfs/TROPOMI/ical/'):
        data_dir = '/nfs/TROPOMI/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    else:
        data_dir = '/data/richardh/Tropomi/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    fl_name = 'S5P_TEST_ICM_CA_SIR_20120918T131651_20120918T145629_01890_01_001100_20151002T140000.h5'
    icm_product = os.path.join(data_dir, fl_name)

    # open ICM product
    icm = ICMio( icm_product )
    if len(icm.select('DPQF_MAP')) > 0:
        dpqm = icm.get_msm_data( 'dpqf_map', band='78' )

        dpqm_dark = icm.get_msm_data( 'dpqm_dark_flux', band='78' )

        dpqm_noise = icm.get_msm_data( 'dpqm_noise', band='78' )

    # generate figure
    figname = fl_name + '.pdf'
    plot = S5Pplot( figname )
    plot.draw_quality( dpqm,
                       title=fl_name,
                       sub_title='dpqf_map' )
    plot.draw_quality( dpqm_dark,
                       title=fl_name,
                       sub_title='dpqm_dark_flux' )
    plot.draw_quality( dpqm_noise,
                       title=fl_name,
                       sub_title='dpqm_noise' )
    del plot
    del icm

#--------------------------------------------------
if __name__ == '__main__':
    print( '*** Info: call function test_geo()')
    test_geo()
    print( '*** Info: call function test_frame()')
    test_frame()
    print( '*** Info: call function test_ckd_dpqf()')
    test_ckd_dpqf()
    print( '*** Info: call function test_icm_dpqf()')
    test_icm_dpqf()
